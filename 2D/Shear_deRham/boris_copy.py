'''
Imports
'''
from firedrake import *
import gc
from pathlib import Path


'''
Parameters
'''
# Spatial discretisation
nx = 2**5  # Mesh number
k = 3  # Spatial degree (Must be >=3, I think)
sigma = Constant(2**5)  # IP parameter

# Temporal discretisation
timestep = Constant(2**(-8))
duration = 2**3

# Setting
Re = Constant(2**30)
amp = 1e-10
# Other
save_int = 1  # How regularly to save output

output_base = Path(__file__).parent / "boris_output"
(output_base / "qois").mkdir(parents=True, exist_ok=True)
(output_base / "continuous_data").mkdir(parents=True, exist_ok=True)
(output_base / "discontinuous_data").mkdir(parents=True, exist_ok=True)
'''
General purpose functions
'''
# Parallelised "print"
print_ = print
def print(x):
    if mesh.comm.rank == 0:
        print_(x, flush = True)



'''
Mesh (and properties)
'''
# Create mesh
mesh = PeriodicUnitSquareMesh(nx, nx, direction="x", quadrilateral=True)

# Get properties
(x, y) = SpatialCoordinate(mesh)  # Cartesian coordinates
n = FacetNormal(mesh)  # Facet normal
h = CellDiameter(mesh)  # Cell diameter



'''
Function spaces
'''
S = FunctionSpace(mesh, "Q", k)
SS = S*S
print(RED % f"Degrees of freedom: {SS.dim()} {[SS_.dim() for SS_ in SS]}")




'''
Functions
'''
# Trial functions
sw = Function(SS)
s_out, w_out = sw.subfunctions
s_out.interpolate(
    conditional(le(y, 0.5), y, 1-y)
  - amp * cos(2*pi*x) * sin(2*pi*y)
)
(s, w) = split(sw)

# Test functions
v_sw = TestFunction(SS)
(v_w, v_s) = split(v_sw)  # Switching the order to make the assembled matrix more symmetric



'''
Residual definition
'''
# Inner products
L2 = lambda a, b : inner(a, b) * dx
L2_interior = lambda a, b : inner(a, b) * dS
H10 = lambda a, b : L2(grad(a), grad(b))
Hdiv0_broken = lambda a, b : (
    L2(div(a), div(b))
  - L2_interior(avg(inner(a, n)), avg(div(b)))
  - L2_interior(avg(div(a)), avg(inner(b, n)))
  + L2_interior(sigma * k**2 / avg(h) * avg(inner(a, n)), avg(inner(b, n)))
)
H20_broken = lambda a, b : Hdiv0_broken(grad(a), grad(b))
H20 = lambda a, b : L2(div(grad(a)), div(grad(b)))
# Other utilities
cross_2D = lambda a, b : a[0]*b[1] - a[1]*b[0]

# Initialise residual
s_ = Function(S)
s_mid = 1/2 * (s + s_)

F = (
    (  # Momentum(/stream function) equation
        1/timestep * H10(s - s_, v_s)
      + L2(w, cross_2D(grad(s_mid), grad(v_s)))
      + 1/Re * H10(w, v_s)
    )
  + (  # (Auxiliary) vorticity equation
        1/Re * H10(w, v_w)
      - 1/Re * H20_broken(s_mid, v_w)
    )
)




'''
Solver parameters
'''
sp = {
    # # Outer (nonlinear) solver
    # # "snes_atol": 1.0e-11,
    # # "snes_rtol": 1.0e-11,

    # "snes_converged_reason"     : None,
    # "snes_linesearch_monitor"   : None,
    # "snes_monitor"              : None,

    # # Inner (linear) solver
    # # "ksp_type"                  : "preonly",  # Krylov subspace = GMRes
    # # "pc_type"                   : "lu",
    # # "pc_factor_mat_solver_type" : "mumps",
    # # "ksp_atol"                  : 1e-8,
    # # "ksp_rtol"                  : 1e-8,
    # # "ksp_max_it"                : 100,

    # # "ksp_monitor" : None,
    # # "ksp_converged_reason" : None,
    # "ksp_monitor_true_residual" : None,
}

def compute_auxiliary_enstrophy(s_ae):
    w_ae = Function(S)
    v_ae = TestFunction(S)
    F_aux = H20_broken(s_ae, v_ae) - H10(w_ae, v_ae)
    solve(F_aux == 0, w_ae, bcs=[DirichletBC(S, 0, "on_boundary")])
    return assemble(0.5 * inner(w_ae, w_ae) * dx)

'''
Write text outputs
'''
# Print and write QoIs
def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = qoi_operator(s_out)
    print(GREEN % f"{qoi_name}: {qoi}")
    if mesh.comm.rank == 0:
        path = output_base / "qois" / f"{qoi_file}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(write_type) as f:
            f.write(str(qoi) + "\n")

qois = [
    {"Name": "Energy", "File": "energy", "Operator": lambda s_ref : assemble(1/2 * H10(s_ref, s_ref))},
    {"Name": "Broken Enstrophy", "File": "broken_enstrophy", "Operator": lambda s_ref : assemble(1/2 * H20_broken(s_ref, s_ref))},
    {"Name": "Internal Enstrophy", "File": "internal_enstrophy", "Operator": lambda s_ref : assemble(1/2 * H20(s_ref, s_ref))},
    {"Name": "Auxiliary Enstrophy", "File": "auxiliary_enstrophy", "Operator": compute_auxiliary_enstrophy},
]

def print_write(write_type):
    for qoi in qois:
        print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)



'''
Output setup
'''
# Set up Paraview
s_out.rename("Stream function")
w_out.rename("Vorticity")
pvd_cts    = VTKFile(str(output_base / "continuous_data/solution.pvd"))
pvd_discts = VTKFile(str(output_base / "discontinuous_data/solution.pvd"))
pvd_cts.write(s_out)

# Print and write initial QoIs
print_write("w")



'''
Solve
'''
time = 0.0
i = 0
while (time < duration - float(timestep)/2):
    # Print timestep
    print(RED % f"Solving for t = {float(time) + float(timestep)}:")

    # Solve
    s_.assign(s_out)
    print(sp)
    solve(F==0, sw, bcs=[DirichletBC(SS_, 0, "on_boundary") for SS_ in SS], solver_parameters=sp)

    # Collect garbage
    gc.collect()

    # Write to Paraview (at intervals to save on memory!)
    i += 1
    if i % save_int == 0:
        pvd_cts.write(s_out)
        pvd_discts.write(w_out)

    # Write text outputs
    print_write("a")

    # Increment time
    time += float(timestep)
