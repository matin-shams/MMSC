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
msh = UnitSquareMesh(N, N)

W = FunctionSpace(msh, "CG", 2)          # ω ∈ CG_{p+1}
V = FunctionSpace(msh, "BDM", 2)         # u ∈ BDM_{p+1}
Q = FunctionSpace(msh, "DG", 1)              # p ∈ DG_p
Z = MixedFunctionSpace([W, V, Q])            # (ω, u, p)


# Manufactured solution for forcing term
x, y = SpatialCoordinate(msh)

# Initial condition
z = Function(Z)
w, u, p = split(z)
u0 = Function(V)
psi, v, q = TestFunctions(Z)

# Semi-discrete variational form
f = as_vector([
    exp((y - 0.75)**2 / 0.25**2),
    0
])

F = (
    inner(Dt(u), v) * dx
    + inner(w * as_vector([u[1], -u[0]]), v) * dx
    + (1 / Re) * inner(rot(w), v) * dx
    + p * div(v) * dx
    - inner(f, v) * dx

    + inner(w, psi) * dx - inner(u, rot(psi)) * dx

    + q * div(u) * dx
)


bcs = [
    DirichletBC(Z.sub(0), 0.0, "on_boundary"),         # ω = 0
    DirichletBC(Z.sub(1), Constant((0.0, 0.0)), "on_boundary")  # u = 0
]
# Create time stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, z, bcs=bcs,
                      solver_parameters=sp)

# Time-stepping loop
# time step n 

pvd = VTKFile("MEEVC.pvd")
w_, u_, p_ = z.subfunctions
pvd.write(w_, u_, p_)

# Time loop
while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(f"t = {float(t):.5f}")
    t.assign(float(t) + float(dt))
    pvd.write(w_, u_, p_)