from firedrake import *



# ─── Parameters ───────────────────────────────────────────────────────────────
final_t = 2**0  # Final time
N = 2**3  # Mesh resolution
dt = Constant(2**-12)  # Timestep
Re = Constant(2**2)  # Reynolds number
S = 1  # Order
centres_amplitudes = [  # Vortex centres and relative amplitudes
    ((0.381966, 0.763932), 1),
    ((0.618034, 0.236068), -1)
]



# ─── Mesh & spaces ─────────────────────────────────────────────────────────────
# Mesh
msh = UnitSquareMesh(N, N, quadrilateral=False)
x, y = SpatialCoordinate(msh)

# Spaces
W = FunctionSpace(msh, "CG", 3, variant="alfeld")  # Scalar CG₃
U = FunctionSpace(msh, "CG", 2, variant="alfeld")  # Scalar CG₂
P = FunctionSpace(msh, "DG", 1, variant="alfeld")  # Scalar DG₁
UUPUUPW = U*U*P*U*U*P*W  # u_x, u_y, p, α_x, α_y, β, ω



# ─── Setting up analytical IC in ψ ──────────────────────────────────────
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

psi_expr = sum([amplitude * stream_func_tidy(x, y, *centre) for (centre, amplitude) in centres_amplitudes])



# ─── Recover discrete IC in u ──────────────────────────────
# Spaces
UUP = U*U*P

# Functions
uup_ = Function(UUP)
u_x_, u_y_, p_ = split(uup_)
u_ = as_vector([u_x_, u_y_])
v_x_, v_y_, q_ = TestFunctions(UUP)
v_ = as_vector([v_x_, v_y_])

# Form (curl u = -Δψ)
rot = lambda vec : vec[0].dx(1) - vec[1].dx(0)
F_ = (
    inner(u_, v_)
  - inner(p_, div(v_))
  - inner(psi_expr, rot(v_))
  - inner(q_, div(u_))
) * dx

# BCs
bcs_ = [
    DirichletBC(UUP.sub(0), 0.0, 1),
    DirichletBC(UUP.sub(0), 0.0, 2),
    DirichletBC(UUP.sub(1), 0.0, 3),
    DirichletBC(UUP.sub(1), 0.0, 4),
]

# Solve
sp_ = {
    # "pc_type": "lu",
    "ksp_monitor_true_residual": None,
}
solve(F_ == 0, uup_, bcs=bcs_, solver_parameters=sp_)

# Norm
energy_ = 1/2 * assemble(inner(u_,u_)*dx)



# ─── Functions (for actual solve) ─────────────────────────────────────────────────────────────
# Trial function
uupaabw = Function(UUPUUPW)
uupaabw.interpolate(as_vector([
    u_x_ / sqrt(energy_),
    u_y_ / sqrt(energy_),
    0,
    0,
    0,
    0,
    0,
]))
(u_x, u_y, p, alpha_x, alpha_y, beta, omega) = split(uupaabw)
u = as_vector([u_x, u_y]);  alpha = as_vector([alpha_x, alpha_y])

# Test function
vvqggdc = TestFunction(UUPUUPW)
(v_x, v_y, q, gamma_x, gamma_y, delta, chi) = split(vvqggdc)
v = as_vector([v_x, v_y]);  gamma = as_vector([gamma_x, gamma_y])




# ─── Form ───────────────────────────────────────
# Forcing
f = as_vector([0,0])

# Form
curl_2D = lambda vec: as_vector([vec.dx(1), - vec.dx(0)])
cross_2D = lambda vec_1, vec_2 : vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0]
F = (
    (
        inner(Dt(u), v)
        - inner(omega, cross_2D(u, v))
        + 1/Re * inner(alpha, v)
        - inner(p, div(v))
    )
    - inner(div(u), q)
    + (
        inner(alpha, gamma)
        - inner(rot(u), rot(gamma))
        - inner(beta, div(gamma))
    )
    - inner(div(alpha), delta)
    + (
        inner(curl_2D(omega), curl_2D(chi))
        - inner(alpha, curl_2D(chi))
    )
) * dx



# ─── Solve ───────────────────────────────────────
# BCs
bcs = [
    DirichletBC(UUPUUPW.sub(0), 0, 1),
    DirichletBC(UUPUUPW.sub(0), 0, 2),
    DirichletBC(UUPUUPW.sub(1), 0, 3),
    DirichletBC(UUPUUPW.sub(1), 0, 4),
    DirichletBC(UUPUUPW.sub(3), 0, 1),
    DirichletBC(UUPUUPW.sub(3), 0, 2),
    DirichletBC(UUPUUPW.sub(4), 0, 3),
    DirichletBC(UUPUUPW.sub(4), 0, 4),
    DirichletBC(UUPUUPW.sub(6), 0, "on_boundary"),
]

# Timestepper
sp = {
    # "pc_type": "lu",
    "ksp_monitor_true_residual": None,
}
t = Constant(0)
stepper = TimeStepper(F, GaussLegendre(S), t, dt, uupaabw, bcs=bcs, solver_parameters=sp)

# Paraview setup
pvd_cts = VTKFile("output/vortex_2d_cts.pvd")
u_x_out = uupaabw.subfunctions[0];  u_x_out.rename("Velocity (x)")
u_y_out = uupaabw.subfunctions[1];  u_y_out.rename("Velocity (y)")
pvd_discts = VTKFile("output/vortex_2d_discts.pvd")
p_out = uupaabw.subfunctions[2];  p_out.rename("Pressure")
alpha_x_out = uupaabw.subfunctions[3];  alpha_x_out.rename("Alpha (x)")
alpha_y_out = uupaabw.subfunctions[4];  alpha_y_out.rename("Alpha (y)")
beta_out = uupaabw.subfunctions[5];  beta_out.rename("Beta")
omega_out = uupaabw.subfunctions[6];  omega_out.rename("Vorticity")

# Solve loop
pvd_cts.write(u_x_out, u_y_out)
while float(t) < final_t:
    if float(t) + float(dt) > final_t:
        dt.assign(final_t - float(t))
    stepper.advance()
    print(
        float(t),
        assemble(inner(div(u), div(u)) * dx),
        assemble(inner(div(alpha), div(alpha)) * dx),
        assemble(inner(curl_2D(omega)-alpha, curl_2D(omega)-alpha) * dx),
        assemble(1/2 * inner(u, u) * dx),
        assemble(1/2 * inner(rot(u), rot(u)) * dx)
    )
    t.assign(float(t) + float(dt))
    pvd_cts.write(u_x_out, u_y_out)
    pvd_discts.write(p_out, alpha_x_out, alpha_y_out, beta_out, omega_out)
