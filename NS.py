from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper

# Parameters
final_t = 1.0 # Final time
N = 2**7 # Mesh number
dt = Constant(2**-12)
Re = Constant(2**3)
# Set up Runge–Kutta implicit midpoint method (1-stage Gauss–Legendre)
t = Constant(0)
butcher_tableau = GaussLegendre(1)

# Mesh and function space
msh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(msh, "CG", 2)
Q = FunctionSpace(msh, "CG", 1)
W = MixedFunctionSpace([V, Q])


# Manufactured solution for forcing term
x, y = SpatialCoordinate(msh)

# Initial condition
up = Function(W)
u, p = split(up)
u0 = Function(V)
v, q = TestFunctions(W)

# Semi-discrete variational form
f = as_vector([
    exp((y - 0.75)**2 / 0.25**2),
    0
])

F = (
    inner(Dt(u), v)*dx 
    + inner(dot(grad(u), u), v) * dx
    - inner(dot(grad(u), v), u) * dx
    + 1/Re*inner(grad(u), grad(v))*dx
    - div(v)*p*dx
    - q*div(u)*dx
    - inner(f, v)*dx
)
bc = DirichletBC(W.sub(0), Constant((0, 0)), "on_boundary")

# Create time stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

# Time-stepping loop
# time step n 
pvd = VTKFile("NS.pvd")
u_n, p_n = up.subfunctions
pvd.write(u_n, p_n)
# pvd = VTKFile("stokes.pvd")
# pvd.write(u)

while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))
    u_n, p_n = up.subfunctions
    pvd.write(u_n, p_n)

