from firedrake import *
from petsc4py.PETSc import DMPlexTransform, DMPlexTransformType
from pathlib import Path
import numpy as np
from scipy import special

# Output directory
output_base = Path(__file__).parent / "output_classical"
(output_base / "qois").mkdir(parents=True, exist_ok=True)

# Colour prints
GREEN = "\033[92m%s\033[0m"
BLUE = "\033[94m%s\033[0m"

# Parameters
Re = Constant(2**16)  # Reynolds number
N = 2**2              # Mesh number
k = 3                 # Polynomial order (>=3)
final_t = 2**-4       # Final time
dt = Constant(2**-10) # Timestep

# Mesh (Alfeld split)
def alfeld_split(mesh):
    dm = mesh.topology_dm
    tr = DMPlexTransform().create(comm=mesh.comm)
    tr.setType(DMPlexTransformType.REFINEALFELD)
    tr.setDM(dm)
    tr.setUp()
    return Mesh(tr.apply(dm))

msh = alfeld_split(UnitCubeMesh(N, N, N))
x, y, z = SpatialCoordinate(msh)

# Parallel-friendly print
print_ = print
def print(*args, sep=" ", end="\n"):
    if msh.comm.rank == 0:
        print_(*args, sep=sep, end=end, flush=True)

# Function spaces (Stokes complex)
V = FunctionSpace(msh, "CG", k)       # H^1 scalar
Q = FunctionSpace(msh, "DG", k-1)     # L^2 pressure

VVVQ = V*V*V*Q   # u_x,u_y,u_z,p
VVV_prev = V*V*V   # to track u_x,u_y,u_z

# Mixed functions
up = Function(VVVQ)
u_x, u_y, u_z, p = split(up)
u = as_vector([u_x, u_y, u_z])

vtest = TestFunction(VVVQ)
v_x, v_y, v_z, q = split(vtest)
v = as_vector([v_x, v_y, v_z])

# Subfunctions for output
u_x_out, u_y_out, u_z_out, p_out = up.subfunctions

# IC setup functions
up_ic = Function(VVVQ)
u_x_ic, u_y_ic, u_z_ic, p_ic = split(up_ic)
u_ic = as_vector([u_x_ic, u_y_ic, u_z_ic])

vq_ic = TestFunction(VVVQ)
v_x_ic, v_y_ic, v_z_ic, q_ic = split(vq_ic)
v_ic = as_vector([v_x_ic, v_y_ic, v_z_ic])

# Previous solution holder
u_prev = Function(VVV_prev)

# Hill vortex functions
bessel_J_root = 5.7634591968945506
bessel_J_root_threehalves = bessel_J(3/2, bessel_J_root)

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

def hill(vec, radius):
    (x, y, z) = vec
    r_cyl = sqrt(x**2 + y**2)
    r_sph = sqrt(x**2 + y**2 + z**2)
    theta = conditional(le(r_cyl, 1e-13), 0, pi/2 - atan(z/r_cyl))
    return conditional(
        ge(r_sph, radius),
        as_vector([0, 0, 0]),
        conditional(
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((bessel_J_root/2)**(3/2)/special.gamma(5/2) - bessel_J_root_threehalves)]),
            conditional(
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                as_vector(
                    hill_r(r_sph, theta, radius) * np.array([x, y, z]) / r_sph
                  + hill_theta(r_sph, theta, radius) * np.array([x*z, y*z, -r_cyl**2]) / r_sph / r_cyl
                  + hill_phi(r_sph, theta, radius) * np.array([-y, x, 0]) / r_cyl
                )
            )
        )
    )

# Paraview setup
pvd_cts = VTKFile(str(output_base / "classical_vortex_3d_cts.pvd"))

u_x_out.rename("Velocity (x)"); u_y_out.rename("Velocity (y)"); u_z_out.rename("Velocity (z)")
p_out.rename("Pressure")

# QoIs
def qoi_energy(): return assemble(0.5 * inner(u, u) * dx)
def qoi_enstrophy(): return assemble(0.5 * inner(omega, omega) * dx)
def qoi_helicity(): return assemble(0.5 * inner(u, omega) * dx)
def qoi_div2(): return assemble(inner(div(u), div(u)) * dx)

qois = [
    {"Name": "Energy",    "File": "energy",    "Operator": qoi_energy},
    {"Name": "Enstrophy", "File": "enstrophy", "Operator": qoi_enstrophy},
    {"Name": "Helicity",  "File": "helicity",  "Operator": qoi_helicity},
    {"Name": "DivL2",     "File": "divu_l2",   "Operator": qoi_div2},
]

def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = float(qoi_operator())
    print(BLUE % f"{qoi_name}: {qoi}")
    if msh.comm.rank == 0:
        path = output_base / "qois" / f"{qoi_file}.txt"
        with path.open(write_type) as f:
            f.write(str(qoi) + "\n")

def print_write(write_type):
    for q in qois:
        print_write_qoi(q["Name"], q["File"], q["Operator"], write_type)

# -----------------------------------------------------------------------------
# IC setup (divergence-free!)
# -----------------------------------------------------------------------------

u_vortex = hill([x-0.5, y-0.5, z-0.5], 0.25)

F_ic = (
    inner(u_ic - u_vortex, v_ic)
  - inner(p_ic, div(v_ic))
  - inner(div(u_ic), q_ic)
) * dx


index_surface = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

bcs_ic = [DirichletBC(VVVQ.sub(index), 0, surface) for (index, surface) in index_surface]

sp_ic = {"ksp_monitor_true_residual": None}

print(GREEN % f"Setting up ICs:")
solve(F_ic == 0, up_ic, bcs=bcs_ic, solver_parameters=sp_ic)

sqrt_energy_ = sqrt(assemble(0.5 * inner(u_ic, u_ic) * dx))
for i in range(3): u_prev.sub(i).assign(up_ic.sub(i) / sqrt_energy_)
for i in range(3): up.sub(i).assign(u_prev.sub(i))

# Output initial data
pvd_cts.write(u_x_out, u_y_out, u_z_out)
print_write("w")

# -----------------------------------------------------------------------------
# Full solve loop
# -----------------------------------------------------------------------------

u_mid = 0.5 * (u + u_prev)
F = (
    inner((u - u_prev) / dt, v)
  + inner(curl(u_mid), cross(u_mid, v))
  + (1/Re) * inner(curl(u_mid), curl(v))
  - inner(p, div(v))
  - inner(q, div(u))
) * dx

bcs = [DirichletBC(VVVQ.sub(index), 0, surface) for (index, surface) in index_surface]
sp = {"ksp_monitor_true_residual": None}

# Time stepping
t = 0.0
while t <= float(final_t) - float(dt)/2:
    t += float(dt)
    print(GREEN % f"Solving for time t = {t}:")
    solve(F == 0, up, bcs=bcs, solver_parameters=sp)
    pvd_cts.write(u_x_out, u_y_out, u_z_out)
    for i in range(3):
        u_prev.sub(i).assign(up.sub(i))
    print_write("a")
