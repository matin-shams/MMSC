from firedrake import *
from pathlib import Path

output_base = Path(__file__).parent / "output_midpoint"
(output_base / "qois").mkdir(parents=True, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
final_t = 2**3  # Final time
N = 2**5  # Mesh resolution
dt = Constant(2**-10)  # Timestep
Re = Constant(2**16)  # Reynolds number
centres_amplitudes = [  # Vortex centres and relative amplitudes
    ((0.381966, 0.763932), 1),
   #((0.618034, 0.236068), -0.7)
]

GREEN = "\033[92m%s\033[0m"

# ─── Mesh & spaces ─────────────────────────────────────────────────────────────
# Mesh
msh = UnitSquareMesh(N, N, quadrilateral=False)
x, y = SpatialCoordinate(msh)

# # Parallelised print (must come AFTER msh exists)
print_ = print
def print(*args, sep=" ", end="\n"):
    if msh.comm.rank == 0:
        print_(*args, sep=sep, end=end, flush=True)

# Spaces
W = FunctionSpace(msh, "CG", 3, variant="alfeld")  # Scalar CG₃ 
#maybe also possible to set W = DG1(alfeld)
U = FunctionSpace(msh, "CG", 2, variant="alfeld")  # Scalar CG₂
P = FunctionSpace(msh, "DG", 1, variant="alfeld")  # Scalar DG₁
UPW = U*U*P*W  # u_x, u_y, p, ω

# ─── Setting up analytical IC in ψ ──────────────────────────────────────
layers = 2**3

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
print(GREEN % f"Setting up ICs:")
solve(F_ == 0, uup_, bcs=bcs_, solver_parameters=sp_)

# Norm
energy_ = assemble(1/2 * inner(u_, u_) * dx)
uup_.interpolate(as_vector([
    u_x_ / sqrt(energy_),
    u_y_ / sqrt(energy_),
    0,
]))


omega_ = Function(W, name="omega_prev")
omega_.project(rot(u_))


# ─── Functions (for actual solve) ─────────────────────────────────────────────────────────────
# Trial function
upw = Function(UPW)
(u_x, u_y, p, omega) = split(upw)
u = as_vector([u_x, u_y])

# CLASSICAL STOKES-COMPLEX tests
vtest = TestFunction(UPW)
(v_x, v_y, q, chi) = split(vtest)
v = as_vector([v_x, v_y])

# ─── Form ───────────────────────────────────────
# Forcing
f = as_vector([0,0])

# Form
curl_2D = lambda vec: as_vector([vec.dx(1), - vec.dx(0)])
cross_2D = lambda vec_1, vec_2 : vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0]
u_mid = 1/2 * (u + u_)
omega_mid = 1/2 * (omega + omega_)

F = (
    (
    inner((u - u_) / dt, v)
  + inner(omega_mid, cross_2D(u_mid, v))
#   + (1/Re) * inner(grad(u_mid), grad(v))
    )
  - inner(p, div(v))

  - inner(q, div(u))

  + inner(omega - rot(u_mid), chi)
) * dx

# ─── Solve ───────────────────────────────────────
# BCs
bcs = [
    DirichletBC(UPW.sub(0), 0, 1),
    DirichletBC(UPW.sub(0), 0, 2),
    DirichletBC(UPW.sub(1), 0, 3),
    DirichletBC(UPW.sub(1), 0, 4),
]

# ---- QoI operators for velocity ----
def qoi_energy():
    # 1/2 ∫ |u|^2
    return assemble(0.5 * inner(u, u) * dx)

def qoi_enstrophy():
    # 1/2 ∫ |rot(u)|^2
    return assemble(0.5 * inner(rot(u), rot(u)) * dx)

def qoi_div2():
    # ∫ |div u|^2
    return assemble(inner(div(u), div(u)) * dx)

def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = float(qoi_operator())
    print(GREEN % f"{qoi_name}: {qoi}")
    if msh.comm.rank == 0:
        path = output_base / "qois" / f"{qoi_file}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(write_type) as f:
            f.write(str(qoi) + "\n")

qois = [
    {"Name": "Energy",    "File": "energy",    "Operator": qoi_energy},
    {"Name": "Enstrophy", "File": "enstrophy", "Operator": qoi_enstrophy},
    {"Name": "DivL2",     "File": "divu_l2",   "Operator": qoi_div2},
]

def print_write(write_type):
    for q in qois:
        print_write_qoi(q["Name"], q["File"], q["Operator"], write_type)



# Timestepper
sp = {
    # "pc_type": "lu",
    "ksp_monitor_true_residual": None,
}

# Paraview setup (ICs)
u_x_out_ = uup_.subfunctions[0]
u_y_out_ = uup_.subfunctions[1]

# Paraview setup (transient)
Path("output_midpoint").mkdir(parents=True, exist_ok=True)
pvd_cts = VTKFile("output_midpoint/classical_vortex_2d_cts.pvd")

u_x_out = upw.subfunctions[0];  u_x_out.rename("Velocity (x)")
u_y_out = upw.subfunctions[1];  u_y_out.rename("Velocity (y)")

pvd_discts = VTKFile("output_midpoint/classical_vortex_2d_discts.pvd")

p_out = upw.subfunctions[2];  p_out.rename("Pressure")
omega_out = upw.subfunctions[3];  omega_out.rename("Vorticity")

# Solve loop
u_x_out.assign(u_x_out_);  u_y_out.assign(u_y_out_)
omega_out.assign(omega_)
# write initial fields
pvd_cts.write(u_x_out, u_y_out)
pvd_discts.write(p_out, omega_out)

print_write("w") 

t = 0
while t <= final_t - float(dt)/2:
    t += float(dt)
    print(GREEN % f"Solving for time t={t}:")
    solve(F==0, upw, bcs=bcs, solver_parameters=sp)
    print(
        assemble(inner(div(u), div(u)) * dx),
        assemble(0.5 * inner(u, u) * dx),
        assemble(0.5 * inner(rot(u), rot(u)) * dx),
    )

    pvd_cts.write(u_x_out, u_y_out)
    pvd_discts.write(p_out, omega_out)
    u_x_out_.assign(u_x_out);  u_y_out_.assign(u_y_out)
    omega_.assign(omega_out)

    print_write("a")
