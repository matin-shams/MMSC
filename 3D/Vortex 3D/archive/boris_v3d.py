from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from firedrake.petsc import PETSc
import numpy as np
from scipy import special
import math

# Parameters
final_t = 1.0 # Final time
N = 2**3 # Mesh number
dt = Constant(2**-12)
Re = Constant(2**-2)
# Set up Runge–Kutta implicit midpoint method (1-stage Gauss–Legendre)
t = Constant(0)
butcher_tableau = GaussLegendre(1)

def alfeld_split(mesh):
    dm = mesh.topology_dm
    tr = PETSc.DMPlexTransform().create(comm=mesh.comm)
    tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    tr.setDM(dm)
    tr.setUp()
    newplex = tr.apply(dm)
    return Mesh(newplex)


# Mesh and function space
msh = UnitCubeMesh(N, N, N, hexahedral=False)
msh = alfeld_split(msh) 
x, y, z = SpatialCoordinate(msh)

V = FunctionSpace(msh, "CG", 2)
Q = FunctionSpace(msh, "DG", 1)
W = FunctionSpace(msh, "N2curl", 3)
R = FunctionSpace(msh, "CG", 4)
UP = MixedFunctionSpace([V, V, V, Q, V, V, V, Q, W, R])


'''
Hill vortex functions
'''
# Firedrake-compatible Bessel function
def besselJ(x, alpha, layers=10):
    return sum([
        (-1)**m / math.factorial(m) / special.gamma(m + alpha + 1)
      * (x/2)**(2*m+alpha)
        for m in range(layers)
    ])



# Bessel function parameters
besselJ_root = 5.7634591968945506
besselJ_root_threehalves = besselJ(besselJ_root, 3/2)



# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
    ) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        besselJ_root * besselJ(besselJ_root*rho, 5/2) / rho**(1/2)
      + 2 * besselJ_root_threehalves
      - 2 * besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return besselJ_root * (
        besselJ(besselJ_root*rho, 3/2) / rho**(3/2)
      - besselJ_root_threehalves
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
            as_vector([0, 0, 2*((besselJ_root/2)**(3/2)/special.gamma(5/2) - besselJ_root_threehalves)]),
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

# The project is divergence-free!
# set up and solve a problem like (u - u_vortex, v) - (p, div(v)) - (div(u), q).
# Divergence-free projection of the vortex
u_vortex = 2 * hill([x-0.5, y-0.5, z-0.5], 0.25)

U_proj = FunctionSpace(msh, "CG", 2)
P_proj = FunctionSpace(msh, "DG", 1)
UP_proj = U_proj * U_proj * U_proj * P_proj

u_ic = Function(UP_proj)
(u_x_ic, u_y_ic, u_z_ic, p_ic) = split(u_ic)
uproj = as_vector([u_x_ic, u_y_ic, u_z_ic])
(v_x_ic, v_y_ic, v_z_ic, q_ic) = TestFunctions(UP_proj)
vproj = as_vector([v_x_ic, v_y_ic, v_z_ic])

F_proj = (
    inner(uproj - u_vortex, vproj)
    - p_ic * div(vproj)
    - q_ic * div(uproj)
) * dx

bcs_proj = [
    DirichletBC(UP_proj.sub(0), 0, 1),
    DirichletBC(UP_proj.sub(0), 0, 2),
    DirichletBC(UP_proj.sub(1), 0, 3),
    DirichletBC(UP_proj.sub(1), 0, 4),
    DirichletBC(UP_proj.sub(2), 0, 5),
    DirichletBC(UP_proj.sub(2), 0, 6),
]

sp_proj = {"ksp_monitor_true_residual": None}
solve(F_proj == 0, u_ic, bcs=bcs_proj, solver_parameters=sp_proj)

u_x0, u_y0, u_z0, _ = u_ic.subfunctions


'''

our 3D scheme from Boris_SV_3D.py (https://github.com/matin-shams/MMSC/blob/main/3D/Boris_SV_3D.py)

'''


up = Function(UP)
(u_x, u_y, u_z, p, alpha_x, alpha_y, alpha_z, beta, omega, r) = split(up)
u = as_vector([u_x, u_y, u_z])
alpha = as_vector([alpha_x, alpha_y, alpha_z])
(v_x, v_y, v_z, q, gamma_x, gamma_y, gamma_z, delta, chi, s) = TestFunctions(UP)
v = as_vector([v_x, v_y, v_z])
gamma = as_vector([gamma_x, gamma_y, gamma_z])


F = (
    (
        inner(Dt(u), v) * dx
        - inner(cross(u, omega), v) * dx
        + 1/Re * inner(alpha, v) * dx
        - p * div(v) * dx
    )
    - div(u) * q * dx
    + (
        inner(alpha, gamma) * dx
        - inner(curl(u), curl(gamma)) * dx
        - beta * div(gamma) * dx
    )
    - div(alpha) * delta * dx
    + (
        inner(curl(omega), curl(chi)) * dx
        - inner(alpha, curl(chi)) * dx
        + inner(grad(r), chi) * dx
    )
    + inner(omega, grad(s)) * dx
)

bc = [
    DirichletBC(UP.sub(0), 0, 1),
    DirichletBC(UP.sub(0), 0, 2),
    DirichletBC(UP.sub(1), 0, 3),
    DirichletBC(UP.sub(1), 0, 4),
    DirichletBC(UP.sub(2), 0, 5),
    DirichletBC(UP.sub(2), 0, 6),
    DirichletBC(UP.sub(4), 0, 1),
    DirichletBC(UP.sub(4), 0, 2),
    DirichletBC(UP.sub(5), 0, 3),
    DirichletBC(UP.sub(5), 0, 4),
    DirichletBC(UP.sub(6), 0, 5),
    DirichletBC(UP.sub(6), 0, 6),
    DirichletBC(UP.sub(8), as_vector([0.0, 0.0, 0.0]), "on_boundary"),
    DirichletBC(UP.sub(9), 0, "on_boundary"),
]

# Initialise with projected vortex
u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n = up.subfunctions
u_x_n.assign(u_x0)
u_y_n.assign(u_y0)
u_z_n.assign(u_z0)

sp = {"ksp_monitor_true_residual": None}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

pvd = VTKFile("vortex_3d.pvd")

# Write initial conditions
t.assign(0.0)
pvd.write(u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n)

while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))
    pvd.write(u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n)