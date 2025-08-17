from firedrake import *
import gc
from pathlib import Path

# Parameters
nx = 2**5  # Mesh number
k = 3      # Spatial degree (Must be >=3, I think)
sigma = Constant(2**5)  # IP parameter

# Temporal discretisation
timestep = Constant(2**(-8))
duration = 2**3

# Setting
Re = Constant(2**24)

# Other
save_int = 1  # How regularly to save output

# Output directories
output_base = Path(__file__).parent / "classical_output"
(output_base / "qois").mkdir(parents=True, exist_ok=True)
(output_base / "continuous_data").mkdir(parents=True, exist_ok=True)

# General purpose functions
print_ = print

def print(x):
    if mesh.comm.rank == 0:
        print_(x, flush=True)

# Mesh (and properties)
mesh = PeriodicUnitSquareMesh(nx, nx, direction="x", quadrilateral=True)
(x, y) = SpatialCoordinate(mesh)
n = FacetNormal(mesh)
h = CellDiameter(mesh)

# Function space
S = FunctionSpace(mesh, "Q", k)
print(RED % f"Degrees of freedom: {S.dim()}")

# Functions
s = Function(S)
s_out = s
v = TestFunction(S)
s_out.interpolate(
    conditional(le(y, 0.5), y, 1-y)
  - 1e-10 * cos(2*pi*x) * sin(2*pi*y)
)

# Residual definition
L2 = lambda a, b: inner(a, b) * dx
L2_interior = lambda a, b: inner(a, b) * dS
H10 = lambda a, b: L2(grad(a), grad(b))
Hdiv0_broken = lambda a, b: (
    L2(div(a), div(b))
  - L2_interior(avg(inner(a, n)), avg(div(b)))
  - L2_interior(avg(div(a)), avg(inner(b, n)))
  + L2_interior(sigma * k**2 / avg(h) * avg(inner(a, n)), avg(inner(b, n)))
)
H20_broken = lambda a, b: Hdiv0_broken(grad(a), grad(b))
H20 = lambda a, b: L2(div(grad(a)), div(grad(b)))

cross_2D = lambda a, b: a[0]*b[1] - a[1]*b[0]

# Classical non-conforming convective term (your supervisor’s pattern)
def convective_classical(sA, v):
    return (
        - div(grad(sA)) * cross_2D(grad(sA), grad(v)) * dx
        + 2 * avg(inner(grad(sA), n)) * cross_2D(avg(grad(sA)), avg(grad(v))) * ds
    )

s_ = Function(S)
s_mid = 0.5 * (s + s_)

F = (
    1/timestep * H10(s - s_, v)                 # time derivative in H1
  + convective_classical(s_mid, v)              # classical non-conforming convection
  + 1/Re * H20_broken(s_mid, v)                 # C0-IP biharmonic viscosity  ✅
)

# Solver parameters
sp = {
    # "ksp_monitor_true_residual": None,
}

def compute_auxiliary_enstrophy(s_ae):
    w_ae = Function(S)
    v_ae = TestFunction(S)
    F_aux = H20_broken(s_ae, v_ae) - H10(w_ae, v_ae)
    solve(F_aux == 0, w_ae, bcs=[DirichletBC(S, 0, "on_boundary")])
    return assemble(0.5 * inner(w_ae, w_ae) * dx)

# Write text outputs
def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = qoi_operator(s_out)
    print(GREEN % f"{qoi_name}: {qoi}")
    if mesh.comm.rank == 0:
        path = output_base / "qois" / f"{qoi_file}.txt"
        with path.open(write_type) as f:
            f.write(str(qoi) + "\n")

qois = [
    {"Name": "Energy", "File": "energy", "Operator": lambda s_ref: assemble(1/2 * H10(s_ref, s_ref))},
    {"Name": "Broken Enstrophy", "File": "broken_enstrophy", "Operator": lambda s_ref: assemble(1/2 * H20_broken(s_ref, s_ref))},
    {"Name": "Internal Enstrophy", "File": "internal_enstrophy", "Operator": lambda s_ref: assemble(1/2 * H20(s_ref, s_ref))},
    {"Name": "Auxiliary Enstrophy", "File": "auxiliary_enstrophy", "Operator": compute_auxiliary_enstrophy},
]

def print_write(write_type):
    for qoi in qois:
        print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)

# Output setup
s_out.rename("Stream function")
pvd_cts = VTKFile(str(output_base / "continuous_data/solution.pvd"))
pvd_cts.write(s_out)

# Print and write initial QoIs
print_write("w")

# Solve
time = 0.0
i = 0
while time < duration - float(timestep)/2:
    print(RED % f"Solving for t = {float(time) + float(timestep)}:")
    s_.assign(s_out)
    solve(F == 0, s, bcs=DirichletBC(S, 0, "on_boundary"), solver_parameters=sp)
    gc.collect()
    i += 1
    if i % save_int == 0:
        pvd_cts.write(s_out)
    print_write("a")
    time += float(timestep)