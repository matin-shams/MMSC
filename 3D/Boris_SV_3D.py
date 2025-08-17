from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from firedrake.petsc import PETSc
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

V = FunctionSpace(msh, "CG", 2)
Q = FunctionSpace(msh, "DG", 1)
W = FunctionSpace(msh, "N2curl", 3)
R = FunctionSpace(msh, "CG", 4)
UP = MixedFunctionSpace([V, V, V, Q, V, V, V, Q, W, R])


# Manufactured solution for forcing term
x, y, z = SpatialCoordinate(msh)

# Initial condition
up = Function(UP)
u_x, u_y, u_z, p, alpha_x, alpha_y, alpha_z, beta, omega, r = split(up)
u = as_vector([u_x, u_y, u_z])
alpha = as_vector([alpha_x, alpha_y, alpha_z])
v_x, v_y, v_z, q, gamma_x, gamma_y, gamma_z, delta, chi, s = TestFunctions(UP)
v = as_vector([v_x, v_y, v_z])
gamma = as_vector([gamma_x, gamma_y, gamma_z])

# Semi-discrete variational form
f = as_vector([
    exp(((y - 0.5)**2 + (z - 0.5)**2) / 0.25**2),
    0,
    0
])

F = (
    (
        inner(Dt(u), v) * dx
        - inner(cross(u, omega), v) * dx
        + 1/Re * inner(alpha, v) * dx
        - p * div(v) * dx
        - inner(f, v) * dx
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
    DirichletBC(UP.sub(9), 0, "on_boundary")
]

# Create time stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

# Time-stepping loop
# time step n 
pvd = VTKFile("Boris_SV_3D.pvd")
u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n = up.subfunctions
pvd.write(u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n)
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
    u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n = up.subfunctions
    pvd.write(u_x_n, u_y_n, u_z_n, p_n, alpha_x_n, alpha_y_n, alpha_z_n, beta_n, omega_n, r_n)

