from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
import os

# Parameters (matching boris.py)
nx = 2**5
k = 3
sigma = Constant(2**5)  # unused but kept for comparison
dt = Constant(2**-8)
duration = 2**3
Re = Constant(2**24)

# Time-stepping setup
butcher_tableau = GaussLegendre(1)
t = Constant(0)

# Mesh
mesh = PeriodicUnitSquareMesh(nx, nx, direction="x", quadrilateral=True)
x, y = SpatialCoordinate(mesh)

# Function spaces
test_deg = 2
V = VectorFunctionSpace(mesh, "CG", test_deg)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Initial stream function and velocity
psi_expr = conditional(le(y, 0.5), y, 1 - y) - 1e-10*cos(2*pi*x)*sin(2*pi*y)
u_expr = as_vector((psi_expr.dx(1), -psi_expr.dx(0)))

up = Function(W)
up.subfunctions[0].interpolate(u_expr)
up.subfunctions[1].assign(0)

u, p = split(up)
v, q = TestFunctions(W)

# Boundary conditions
bc = DirichletBC(W.sub(0), u_expr, "on_boundary")

# Navier--Stokes residual
F = (
    inner(Dt(u), v) * dx
    + inner(dot(grad(u), u), v) * dx
    - inner(dot(grad(u), v), u) * dx
    + 1/Re * inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    - q * div(u) * dx
)

sp = {"ksp_monitor_true_residual": None}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc, solver_parameters=sp)

# Output setup
os.makedirs("output/continuous_data", exist_ok=True)
os.makedirs("output/discontinuous_data", exist_ok=True)
Vpsi = FunctionSpace(mesh, "CG", k)
psi = Function(Vpsi, name="Stream function")
Wvort = FunctionSpace(mesh, "CG", 1)
w = Function(Wvort, name="Vorticity")
phi = TestFunction(Vpsi)

pvd_cts = VTKFile("output/continuous_data/solution.pvd")
pvd_discts = VTKFile("output/discontinuous_data/solution.pvd")


def update_outputs():
    u_now, _ = up.subfunctions
    project(curl(u_now), w, solver_parameters=sp)
    solve(inner(grad(psi), grad(phi)) * dx == w * phi * dx,
          psi,
          bcs=DirichletBC(Vpsi, Constant(0), "on_boundary"),
          solver_parameters=sp)
    pvd_cts.write(psi)
    pvd_discts.write(w)


update_outputs()
while float(t) < duration:
    if float(t) + float(dt) > duration:
        dt.assign(duration - float(t))
    stepper.advance()
    t.assign(float(t) + float(dt))
    update_outputs()
    print(float(t))
