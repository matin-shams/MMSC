from firedrake import *
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
import gc

# ─── Parameters ───────────────────────────────────────────────────────────────
final_t       = 1.0            # Final time
N             = 2**5           # Mesh resolution
dt            = Constant(2**-12)
Re            = Constant(2**2)
t             = Constant(0)    
butcher_tableau = GaussLegendre(1)

# ─── Mesh & Spaces ─────────────────────────────────────────────────────────────
msh = UnitSquareMesh(N, N, quadrilateral=False)
V   = FunctionSpace(msh, "CG", 2, variant="alfeld")   # Scalar CG₂
Q   = FunctionSpace(msh, "DG", 1, variant="alfeld")   # Scalar DG₁
W   = FunctionSpace(msh, "CG", 3, variant="alfeld")   # Scalar CG₃
UP  = MixedFunctionSpace([V, V, Q, V, V, Q, W])       # uₓ,u_y,p,αₓ,α_y,β,ω

# Coordinates
x, y = SpatialCoordinate(msh)

# ─── Stream‐function utilities (IC only) , analytical ψ ──────────────────────────────────────
layers = 10
def layer_summands(x, y, f, layers=10):
    base = f(x, y, 0, 0)
    out  = base
    for m in range(-layers, layers+1):
        for n in range(-layers, layers+1):
            if not (m==0 and n==0):
                out += (f(x,y,m,n) - f(0,0,m,n))
    return out

den = lambda x,y,m,n: ((x-2*m)**2 + (y-2*n)**2)**2
real_part = lambda x,y: layer_summands(x,y,
    lambda x,y,m,n: ((x-2*m)**2 - (y-2*n)**2)/den(x,y,m,n), layers)
imag_part = lambda x,y: layer_summands(x,y,
    lambda x,y,m,n: -2*(x-2*m)*(y-2*n)/den(x,y,m,n), layers)

ln_f     = lambda x,y,X,Y: 0.5*ln((real_part(x,y)-real_part(X,Y))**2 + (imag_part(x,y)-imag_part(X,Y))**2)
stream_f = lambda x,y,X,Y: ln_f(x,y,X,Y) - ln_f(x,y,X,-Y)

def stream_func_tidy(x,y,X,Y):
    return conditional(
        le(x,0.5),
        conditional(le(y,0.5),
            stream_f(x,y,X,Y),
            stream_f(x,1-y,X,1-Y)
        ),
        conditional(le(y,0.5),
            stream_f(1-x,y,1-X,Y),
            stream_f(1-x,1-y,1-X,1-Y)
        ),
    )

# ─── Build analytical ψ and recover (uₓ,u_y) IC ──────────────────────────────
# dipole centres
v1 = (0.381966, 0.763932)
v2 = (0.618034, 0.236068)
psi_expr = stream_func_tidy(x, y, *v1) - stream_func_tidy(x, y, *v2)

# small mixed space just for IC
UV  = FunctionSpace(msh, "CG", 2, variant="alfeld")
PQ  = FunctionSpace(msh, "DG", 1, variant="alfeld")
UVP = MixedFunctionSpace([UV, UV, PQ])
uvp = Function(UVP)
u_x_, u_y_, p_ = split(uvp)
v_x_, v_y_, q_ = TestFunctions(UVP)
u_ = as_vector([u_x_, u_y_])
rot_v_ = v_y_.dx(0) - v_x_.dx(1)

#  curl u = -Δψ
F_ic = (
    inner(u_, as_vector([v_x_, v_y_]))*dx
  - p_*div(as_vector([v_x_, v_y_]))*dx
  - psi_expr*rot_v_*dx
  - q_*div(u_)*dx
)
bc_ic = [
    DirichletBC(UVP.sub(0), 0.0, 1),
    DirichletBC(UVP.sub(0), 0.0, 2),
    DirichletBC(UVP.sub(1), 0.0, 3),
    DirichletBC(UVP.sub(1), 0.0, 4),
]

#sp = {"pc_type":"lu"}
sp = {
    "ksp_monitor_true_residual": None,
}
solve(F_ic == 0, uvp, bcs=bc_ic, solver_parameters=sp)

# extract and build vector u₀
u_x0, u_y0, _ = split(uvp)
u0 = Function(VectorFunctionSpace(msh, "CG", 2, variant="alfeld"))
u0.interpolate(as_vector([u_x0, u_y0]))
u0.assign(u0 / sqrt(assemble(inner(u0,u0)*dx)))  # normalise energy

# project into big UP (zeros for all other p,α,β,ω)
up_init = project(as_vector([
    u0[0], u0[1],          # uₓ, u_y
    0,                     # p
    0, 0,                  # αₓ, α_y
    0,                     # β
    0                      # ω
]), UP)
# assign into the solution function
up = Function(UP).assign(up_init)


# ─── Now our 2d existing scheme, unchanged as before ───────────────────────────────────────
u_x, u_y, p, alpha_x, alpha_y, beta, omega = split(up)
u     = as_vector([u_x, u_y])
alpha = as_vector([alpha_x, alpha_y])
v_x, v_y, q, gamma_x, gamma_y, delta, chi = TestFunctions(UP)
v     = as_vector([v_x, v_y])
gamma = as_vector([gamma_x, gamma_y])

# manufactured forcing
f = as_vector([0,0])

def curl(vec):
    return as_vector([vec.dx(1), - vec.dx(0)])
F = (
    (
        inner(Dt(u), v) * dx
        + inner(omega * as_vector([- u[1], u[0]]), v) * dx
        + 1/Re * inner(alpha, v) * dx
        - p * div(v) * dx
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

# BCs (unchanged)
bc = [
    DirichletBC(UP.sub(0), 0, 1),
    DirichletBC(UP.sub(0), 0, 2),
    DirichletBC(UP.sub(1), 0, 3),
    DirichletBC(UP.sub(1), 0, 4),
    DirichletBC(UP.sub(3), 0, 1),
    DirichletBC(UP.sub(3), 0, 2),
    DirichletBC(UP.sub(4), 0, 3),
    DirichletBC(UP.sub(4), 0, 4),
    DirichletBC(UP.sub(6), 0, "on_boundary"),
]

# Time‐stepper
sp = {
    "ksp_monitor_true_residual": None,
}
stepper = TimeStepper(F, butcher_tableau, t, dt, up, bcs=bc,
                      solver_parameters=sp)

# Output & time‐loop
pvd = VTKFile("vortex_2d.pvd")
u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n = up.subfunctions
pvd.write(u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n)

while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(float(t),
          assemble(div(u)**2*dx),
          assemble(div(alpha)**2*dx),
          assemble(inner(curl2(omega)-alpha, curl2(omega)-alpha)*dx))
    t.assign(float(t) + float(dt))
    u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n = up.subfunctions
    pvd.write(u_x_n, u_y_n, p_n, alpha_x_n, alpha_y_n, beta_n, omega_n)

