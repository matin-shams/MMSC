from firedrake import *
from petsc4py.PETSc import DMPlexTransform, DMPlexTransformType
import numpy as np
from scipy import special



'''
Parameters
'''
# Problem parameters
Re = Constant(2**16)  # Reynolds number

# Spatial discretisation
N = 2**2  # Mesh number
k = 3  # Order (>=3)

# Time discretisation
final_t = 2**0  # Final time
dt = Constant(2**-10)  # Timestep



'''
Mesh
'''
# Alfeld split
def alfeld_split(mesh):
    dm = mesh.topology_dm
    tr = DMPlexTransform().create(comm=mesh.comm)
    tr.setType(DMPlexTransformType.REFINEALFELD)
    tr.setDM(dm)
    tr.setUp()
    return Mesh(tr.apply(dm))

# Create
msh = alfeld_split(UnitCubeMesh(N, N, N))
x, y, z = SpatialCoordinate(msh)



'''
Function spaces
'''
# Component
R = FunctionSpace(msh, "CG", k+2)
W = FunctionSpace(msh, "N2curl", k+1)
V = FunctionSpace(msh, "CG", k)
Q = FunctionSpace(msh, "DG", k-1)

# Mixed
VQVQWR = MixedFunctionSpace([V, V, V, Q, V, V, V, Q, W, R])  # General
VQ_ic = MixedFunctionSpace([V, V, V, Q])  # For setting up ICs
V_prev = MixedFunctionSpace([V, V, V])  # For tracking u
print(RED % f"Degrees of freedom: {VQVQWR.dim()} {[VQVQWR_.dim() for VQVQWR_ in VQVQWR]}")



'''
Functions
'''
# General
upabor = Function(VQVQWR)
u_x, u_y, u_z, p, alpha_x, alpha_y, alpha_z, beta, omega, r = split(upabor)
u = as_vector([u_x, u_y, u_z]); alpha = as_vector([alpha_x, alpha_y, alpha_z])
u_x_out, u_y_out, u_z_out, p_out, alpha_x_out, alpha_y_out, alpha_z_out, beta_out, omega_out, r_out = upabor.subfunctions

vqgdcs = TestFunction(VQVQWR)
v_x, v_y, v_z, q, gamma_x, gamma_y, gamma_z, delta, chi, s = split(vqgdcs)
v = as_vector([v_x, v_y, v_z]); gamma = as_vector([gamma_x, gamma_y, gamma_z])

# IC setup
up_ic = Function(VQ_ic)
u_x_ic, u_y_ic, u_z_ic, p_ic = split(up_ic)
u_ic = as_vector([u_x_ic, u_y_ic, u_z_ic])

vq_ic = TestFunction(VQ_ic)
v_x_ic, v_y_ic, v_z_ic, q_ic = split(vq_ic)
v_ic = as_vector([v_x_ic, v_y_ic, v_z_ic])

# Tracking
u_prev = Function(V_prev)
u_x_prev, u_y_prev, u_z_prev = split(u_prev)



'''
Hill vortex functions
'''
# Bessel function parameters
bessel_J_root = 5.7634591968945506
bessel_J_root_threehalves = bessel_J(3/2, bessel_J_root)

# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
      - bessel_J_root_threehalves
    ) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        bessel_J_root * bessel_J(5/2, bessel_J_root*rho) / rho**(1/2)
      + 2 * bessel_J_root_threehalves
      - 2 * bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return bessel_J_root * (
        bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
      - bessel_J_root_threehalves
    ) * rho * sin(theta)

# Hill vortex (Cartesian)
def hill(vec, radius):
    (x, y, z) = vec

    # Cylindrical/spherical coordinates
    r_cyl = sqrt(x**2 + y**2)  # Cylindrical radius
    r_sph = sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = conditional(  # Spherical angle
        le(r_cyl, 1e-13),
        0,
        pi/2 - atan(z/r_cyl)
    )

    return conditional(  # If we're outside the vortex...
        ge(r_sph, radius),
        as_vector([0, 0, 0]),
        conditional(  # If we're at the origin...
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((bessel_J_root/2)**(3/2)/special.gamma(5/2) - bessel_J_root_threehalves)]),
            conditional(  # If we're on the z axis...
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                as_vector(  # Else...
                    hill_r(r_sph, theta, radius) * np.array([x, y, z]) / r_sph
                  + hill_theta(r_sph, theta, radius) * np.array([x*z, y*z, -r_cyl**2]) / r_sph / r_cyl
                  + hill_phi(r_sph, theta, radius) * np.array([-y, x, 0]) / r_cyl
                )
            )
        )
    )



'''
Paraview setup
'''
# Files
pvd_cts = VTKFile("output/boris_ie/continuous_data.pvd")
pvd_discts = VTKFile("output/boris_ie/discontinuous_data.pvd")

# Functions
u_x_out.rename("Velocity (x)"); u_y_out.rename("Velocity (y)"); u_z_out.rename("Velocity (z)")
p_out.rename("Pressure")
alpha_x_out.rename("Alpha (x)"); alpha_y_out.rename("Alpha (y)"); alpha_z_out.rename("Alpha (z)")
beta_out.rename("Beta")
omega_out.rename("Vorticity")
r_out.rename("Lagrange multiplier")



'''
Data setup
'''
# QoIs
qois_cts = [
    {"Name": "Energy",                                "File": "energy",           "Operator": 1/2 * inner(u, u) * dx},
    {"Name": "Enstrophy",                             "File": "enstrophy",        "Operator": 1/2 * inner(curl(u), curl(u)) * dx},
    {"Name": "Helicity",                              "File": "helicity",         "Operator": 1/2 * inner(u, curl(u)) * dx},
    {"Name": "Divergence of u (L2 norm)",             "File": "divergence_u",     "Operator": inner(div(u), div(u)) * dx},
]
qois_discts = [
    {"Name": "Energy dissipation",                    "File": "energy_diss",      "Operator": dt/4/Re * inner(curl(u + u_prev), curl(u + u_prev)) * dx},
    {"Name": "Enstrophy dissipation",                 "File": "enstrophy_diss",   "Operator": dt/Re * inner(curl(omega), curl(omega)) * dx},
    {"Name": "Enstrophy convective generation",       "File": "enstrophy_gen",    "Operator": dt/2 * inner(cross(u + u_prev, omega), curl(omega)) * dx},
    {"Name": "Divergence of alpha (L2 norm)",         "File": "divergence_alpha", "Operator": inner(div(alpha), div(alpha)) * dx},
    {"Name": "Error in curl omega = alpha (L2 norm)", "File": "omega_error",      "Operator": inner(curl(omega) - alpha, curl(omega) - alpha) * dx},
    {"Name": "Lagrange multiplier (L2 norm)",         "File": "lagrange_mult",    "Operator": inner(r, r) * dx}
]

# Write functions
def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = assemble(qoi_operator)
    print(BLUE % f"{qoi_name}: {qoi}")
    open("output/boris_ie/" + qoi_file + ".txt", write_type).write(str(qoi) + "\n")

def print_write(write_type):
    for qoi in qois_cts:
        print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)
    for qoi in qois_discts:
        if write_type == "w":
            open("output/boris_ie/" + qoi["File"] + ".txt", "w").write("No data for discontinuous QoI at initial condition")
        else:
            print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)
            



'''
IC setup (divergence-free!)
'''
# Target function
u_vortex = hill([x-0.5, y-0.5, z-0.5], 0.25)

# Residual
F_ic = (
    inner(u_ic - u_vortex, v_ic)
  - inner(p_ic, div(v_ic))
  - inner(div(u_ic), q_ic)
) * dx

# Boundary conditions
index_surface = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)
]
bcs_ic = [DirichletBC(VQ_ic.sub(index), 0, surface) for (index, surface) in index_surface]

# Solver paramters
sp_ic = {
    # Outer (nonlinear) solver
    # "snes_atol": 1.0e-11,
    # "snes_rtol": 1.0e-11,

    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,

    # Inner (linear) solver
    # "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    # "pc_type"                   : "lu",
    # "pc_factor_mat_solver_type" : "mumps",
    # "ksp_atol"                  : 1e-8,
    # "ksp_rtol"                  : 1e-8,
    # "ksp_max_it"                : 100,

    "ksp_monitor" : None,
    "ksp_converged_reason" : None,
    "ksp_monitor_true_residual" : None,
}

# Solve
print(GREEN % f"Setting up ICs:")
solve(F_ic==0, up_ic, bcs=bcs_ic, solver_parameters=sp_ic)

# Extract and normalise u component
sqrt_energy_ = sqrt(assemble(1/2 * inner(u_ic, u_ic) * dx))
for i in range(3):
    u_prev.sub(i).assign(up_ic.sub(i) / sqrt_energy_)



'''
Collect initial data
'''
for i in range(3):
    upabor.sub(i).assign(u_prev.sub(i))
pvd_cts.write(u_x_out, u_y_out, u_z_out)
print_write("w")



'''
Full solve loop
'''
# Residual
F = (
    (  # Momentum
        1/dt * inner(u - u_prev, v)
      - inner(cross(u, omega), v)
      + 1/Re * inner(alpha, v)
      - inner(p, div(v))
    )
  + (  # Incompressiblity
      - inner(div(u), q)
    )
  + (  # alpha (i.e. curl curl u)
        inner(alpha, gamma)
      - inner(curl(u), curl(gamma))
      - inner(beta, div(gamma))
    )
  + (  # alpha divergence-free constraint
      - inner(div(alpha), delta)
    )
  + (  # curl omega = alpha
        inner(curl(omega), curl(chi))
      - inner(alpha, curl(chi))
      + inner(grad(r), chi)
    )
  + (  # omega (adjoint) divergence-free constraint
        inner(omega, grad(s))
    )
) * dx

# Boundary conditions
index_surface = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
    (4, 1), (4, 2), (5, 3), (5, 4), (6, 5), (6, 6),
    (8, "on_boundary"),
    (9, "on_boundary")
]
bcs = [DirichletBC(VQVQWR.sub(index), 0, surface) for (index, surface) in index_surface]

# Solver paramters
sp = {
    # Outer (nonlinear) solver
    # "snes_atol": 1.0e-11,
    # "snes_rtol": 1.0e-11,

    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,

    # Inner (linear) solver
    # "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    # "pc_type"                   : "lu",
    # "pc_factor_mat_solver_type" : "mumps",
    # "ksp_atol"                  : 1e-8,
    # "ksp_rtol"                  : 1e-8,
    # "ksp_max_it"                : 100,

    "ksp_monitor" : None,
    "ksp_converged_reason" : None,
    "ksp_monitor_true_residual" : None,
}

# Solve loop
t = 0
while t <= final_t - float(dt)/2:
    t += float(dt)
    print(GREEN % f"Solving for time t = {t}:")
    for i in range(3):
        u_prev.sub(i).assign(upabor.sub(i))
    solve(F==0, upabor, bcs=bcs, solver_parameters=sp)
    pvd_cts.write(u_x_out, u_y_out, u_z_out)
    pvd_discts.write(p_out, alpha_x_out, alpha_y_out, alpha_z_out, beta_out, omega_out, r_out)
    print_write("a")
