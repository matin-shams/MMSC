from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper

# Parameters
final_t = 1.0 # Final time
N = 2**5 # Mesh number
dt = Constant(2**-12)
Re = Constant(2**2)
# Set up Runge–Kutta implicit midpoint method (1-stage Gauss–Legendre)
t = Constant(0)
butcher_tableau = GaussLegendre(1)

# Mesh and function space
msh = UnitSquareMesh(N, N, quadrilateral=False)
V = FunctionSpace(msh, "CG", 2, variant="alfeld")
Q = FunctionSpace(msh, "DG", 1, variant="alfeld")
W = FunctionSpace(msh, "CG", 3, variant="alfeld")
UP = MixedFunctionSpace([V, V, Q, V, V, Q, W])


# Manufactured solution for forcing term
x, y = SpatialCoordinate(msh)

# Initial condition
up = Function(UP)
u_x, u_y, p, alpha_x, alpha_y, beta, omega = split(up)
u = as_vector([u_x, u_y])
alpha = as_vector([alpha_x, alpha_y])
v_x, v_y, q, gamma_x, gamma_y, delta, chi = TestFunctions(UP)
v = as_vector([v_x, v_y])
gamma = as_vector([gamma_x, gamma_y])

# Semi-discrete variational form
f = as_vector([
    exp((y - 0.75)**2 / 0.25**2),
    0
])

def curl(vec):
    return as_vector([vec.dx(1), - vec.dx(0)])
F = (
    (
        inner(Dt(u), v) * dx
        + inner(omega * as_vector([- u[1], u[0]]), v) * dx
        + 1/Re * inner(alpha, v) * dx
        - p * div(v) * dx
        - inner(f, v) * dx
    )
    - div(u) * q * dx
    + (
        inner(alpha, gamma) * dx
        - inner(rot(u), rot(gamma)) * dx
        - beta * div(gamma) * dx
    )
    - div(alpha) * delta * dx
    + (
        inner(curl(omega), curl(chi)) * dx
        - inner(alpha, curl(chi)) * dx
    )
)

bc = [
    DirichletBC(UP.sub(0), 0, 1),
    DirichletBC(UP.sub(0), 0, 2),
    DirichletBC(UP.sub(1), 0, 3),
    DirichletBC(UP.sub(1), 0, 4),
    DirichletBC(UP.sub(3), 0, 1),
    DirichletBC(UP.sub(3), 0, 2),
    DirichletBC(UP.sub(4), 0, 3),
    DirichletBC(UP.sub(4), 0, 4),
    DirichletBC(UP.sub(6), 0, "on_boundary")
]

# Create time stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

# Time-stepping loop
# time step n 
pvd = VTKFile("Boris_SV.pvd")
u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n = up.subfunctions
pvd.write(u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n)
# pvd = VTKFile("stokes.pvd")
# pvd.write(u)

while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t))
    print(assemble((div(u)**2)*dx))
    print(assemble((div(alpha)**2)*dx))
    print(assemble(inner(curl(omega) - alpha, curl(omega) - alpha)*dx))
    t.assign(float(t) + float(dt))
    u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n = up.subfunctions
    pvd.write(u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n)

