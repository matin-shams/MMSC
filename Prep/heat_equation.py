from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from ufl.algorithms.ad import expand_derivatives

# Parameters
final_t = 10.0

# Set up Runge–Kutta implicit midpoint method (1-stage Gauss–Legendre)
butcher_tableau = GaussLegendre(1)
ns = butcher_tableau.num_stages

# Mesh and function space
N = 100
x0, x1, y0, y1 = 0.0, 10.0, 0.0, 10.0
msh = RectangleMesh(N, N, x1, y1)
V = FunctionSpace(msh, "CG", 1)

# Time variables
MC = MeshConstant(msh)
dt = MC.Constant(10.0 / N)
t = MC.Constant(0.0)

# Manufactured solution for forcing term
x, y = SpatialCoordinate(msh)
S, C = Constant(2.0), Constant(1000.0)
B = (x - x0)*(x - x1)*(y - y0)*(y - y1)/C
R = sqrt(x*x + y*y)
uexact = B * atan(t)*(pi/2 - atan(S*(R - t)))
rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

# Initial condition
u = Function(V)
u.interpolate(uexact)

# Semi-discrete variational form
v = TestFunction(V)
F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx - inner(rhs, v)*dx
bc = DirichletBC(V, 0, "on_boundary")

# Solver parameters (direct LU)
luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

# Create time stepper
stepper = TimeStepper(F, butcher_tableau, t, dt, u, bcs=bc,
                      solver_parameters=luparams)

# Time-stepping loop
pvd = VTKFile("heat_equation.pvd")
pvd.write(u)
while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t))
    t.assign(float(t) + float(dt))
    pvd.write(u)

# Compute relative L² error
print()
print(norm(u - uexact)/norm(uexact))
