from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper

# Parameters
final_t = 1.0 # Final time
N = 2**2 # Mesh number
dt = Constant(2**-12)
Re = Constant(2**-2)
# Set up Runge–Kutta implicit midpoint method (1-stage Gauss–Legendre)
t = Constant(0)
butcher_tableau = GaussLegendre(1)

# Mesh and function space
msh = UnitCubeMesh(N, N, N, hexahedral=False)
V = FunctionSpace(msh, "CG", 5)
Q = FunctionSpace(msh, "DG", 4)
W = MixedFunctionSpace([V, V, V, Q])


# Manufactured solution for forcing term
x, y, z = SpatialCoordinate(msh)

# Initial condition
up = Function(W)
u_x, u_y, u_z, p = split(up)
u = as_vector([u_x, u_y, u_z])
v_x, v_y, v_z, q = TestFunctions(W)
v = as_vector([v_x, v_y, v_z])

# Semi-discrete variational form
f = as_vector([
    exp(((y - 0.5)**2 + (z - 0.5)**2) / 0.25**2),
    0,
    0
])

F = (
    inner(Dt(u), v)*dx 
    - inner(cross(u, curl(u)), v) * dx
    + 1/Re*inner(grad(u), grad(v))*dx
    - div(v)*p*dx
    - q*div(u)*dx
    - inner(f, v)*dx
)
bc = [
    DirichletBC(W.sub(0), 0, 1),
    DirichletBC(W.sub(0), 0, 2),
    DirichletBC(W.sub(1), 0, 3),
    DirichletBC(W.sub(1), 0, 4),
    DirichletBC(W.sub(2), 0, 5),
    DirichletBC(W.sub(2), 0, 6)
]

# Create time stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

# Time-stepping loop
# time step n 
pvd = VTKFile("NS_SV_3D.pvd")
u_x_n, u_y_n, u_z_n, p_n = up.subfunctions
pvd.write(u_x_n, u_y_n, u_z_n, p_n)
# pvd = VTKFile("stokes.pvd")
# pvd.write(u)

while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t))
    print(assemble(inner(div(u),div(u))*dx))
    t.assign(float(t) + float(dt))
    u_x_n, u_y_n, u_z_n, p_n = up.subfunctions
    pvd.write(u_x_n, u_y_n, u_z_n, p_n)

