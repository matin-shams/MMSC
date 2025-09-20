# Every addition is marked with:   # [ADDED] reason

from firedrake import *
from petsc4py.PETSc import DMPlexTransform, DMPlexTransformType
import numpy as np
from pathlib import Path
from scipy import special
from scipy.fft import fftn, fftfreq  # [ADDED] FFTs for spectrum

GREEN = BLUE = RED = YELLOW = "%s"

# ─── Output directory ─────────────────────────────────────────────────────────
output_base = Path(__file__).parent / "output_boris"
(output_base / "qois").mkdir(parents=True, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
Re = Constant(2**16)   # Reynolds number
N  = 2**2              # Mesh number
k  = 3                 # FE order (>=3)
final_t = 2**0         # Final time
dt = Constant(2**-10)  # Timestep

# ─── Mesh ─────────────────────────────────────────────────────────────────────
def alfeld_split(mesh):
    dm = mesh.topology_dm
    tr = DMPlexTransform().create(comm=mesh.comm)
    tr.setType(DMPlexTransformType.REFINEALFELD)
    tr.setDM(dm)
    tr.setUp()
    return Mesh(tr.apply(dm))

msh = alfeld_split(UnitCubeMesh(N, N, N))
x, y, z = SpatialCoordinate(msh)

# ─── Function spaces ──────────────────────────────────────────────────────────
R = FunctionSpace(msh, "CG", k+1)
W = FunctionSpace(msh, "N1curl", k+1)
V = FunctionSpace(msh, "CG", k)
Q = FunctionSpace(msh, "DG", k-1)

VQVQWR = MixedFunctionSpace([V, V, V, Q, V, V, V, Q, W, R])
VQ_ic  = MixedFunctionSpace([V, V, V, Q])
V_prev = MixedFunctionSpace([V, V, V])

print(RED % f"Degrees of freedom: {VQVQWR.dim()} {[VQVQWR_.dim() for VQVQWR_ in VQVQWR]}")

# ─── Functions ────────────────────────────────────────────────────────────────
upabor = Function(VQVQWR)
u_x, u_y, u_z, p, alpha_x, alpha_y, alpha_z, beta, omega, r = split(upabor)
u     = as_vector([u_x, u_y, u_z])
alpha = as_vector([alpha_x, alpha_y, alpha_z])

u_x_out, u_y_out, u_z_out, p_out, alpha_x_out, alpha_y_out, alpha_z_out, beta_out, omega_out, r_out = upabor.subfunctions

vqgdcs = TestFunction(VQVQWR)
v_x, v_y, v_z, q, gamma_x, gamma_y, gamma_z, delta, chi, s = split(vqgdcs)
v     = as_vector([v_x, v_y, v_z])
gamma = as_vector([gamma_x, gamma_y, gamma_z])

up_ic       = Function(VQ_ic)
u_x_ic, u_y_ic, u_z_ic, p_ic = split(up_ic)
u_ic        = as_vector([u_x_ic, u_y_ic, u_z_ic])

vq_ic       = TestFunction(VQ_ic)
v_x_ic, v_y_ic, v_z_ic, q_ic = split(vq_ic)
v_ic        = as_vector([v_x_ic, v_y_ic, v_z_ic])

u_prev = Function(V_prev)
u_x_prev, u_y_prev, u_z_prev = split(u_prev)

# ─── Hill vortex functions  (unchanged physics) ───────────────────────────────
bessel_J_root = 5.7634591968945506
bessel_J_root_threehalves = bessel_J(3/2, bessel_J_root)

def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (bessel_J(3/2, bessel_J_root*rho) / rho**(3/2) - bessel_J_root_threehalves) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        bessel_J_root * bessel_J(5/2, bessel_J_root*rho) / rho**(1/2)
      + 2 * bessel_J_root_threehalves
      - 2 * bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return bessel_J_root * (bessel_J(3/2, bessel_J_root*rho) / rho**(3/2) - bessel_J_root_threehalves) * rho * sin(theta)

def hill(vec, radius):
    (x_, y_, z_) = vec
    r_cyl = sqrt(x_**2 + y_**2)
    r_sph = sqrt(x_**2 + y_**2 + z_**2)
    theta = conditional(le(r_cyl, 1e-13), 0, pi/2 - atan(z_/r_cyl))
    return conditional(
        ge(r_sph, radius), as_vector([0, 0, 0]),
        conditional(
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((bessel_J_root/2)**(3/2)/special.gamma(5/2) - bessel_J_root_threehalves)]),
            conditional(
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                as_vector(
                    hill_r(r_sph, theta, radius) * as_vector([x_, y_, z_]) / r_sph
                  + hill_theta(r_sph, theta, radius) * as_vector([x_*z_, y_*z_, -r_cyl**2]) / r_sph / r_cyl
                  + hill_phi(r_sph, theta, radius) * as_vector([-y_, x_, 0]) / r_cyl
                )
            )
        )
    )

# ─── ParaView setup (unchanged) ───────────────────────────────────────────────
pvd_cts    = VTKFile(str(output_base / "continuous_data.pvd"))
pvd_discts = VTKFile(str(output_base / "discontinuous_data.pvd"))

u_x_out.rename("Velocity (x)"); u_y_out.rename("Velocity (y)"); u_z_out.rename("Velocity (z)")
p_out.rename("Pressure")
alpha_x_out.rename("Alpha (x)"); alpha_y_out.rename("Alpha (y)"); alpha_z_out.rename("Alpha (z)")
beta_out.rename("Beta")
omega_out.rename("Vorticity")
r_out.rename("Lagrange multiplier")

# ─── QoIs  (unchanged) ────────────────────────────────────────────────────────
qois_cts = [
    {"Name": "Energy",                                "File": "energy",           "Operator": 1/2 * inner(u, u) * dx},
    {"Name": "Enstrophy",                             "File": "enstrophy",        "Operator": 1/2 * inner(curl(u), curl(u)) * dx},
    {"Name": "Helicity",                              "File": "helicity",         "Operator": 1/2 * inner(u, curl(u)) * dx},
    {"Name": "Divergence of u (L2 norm)",             "File": "divergence_u",     "Operator": inner(div(u), div(u)) * dx},
]
qois_discts = [
    {"Name": "Energy dissipation",                    "File": "energy_diss",      "Operator": dt/4/Re * inner(curl(u + u_prev), curl(u + u_prev)) * dx},
    {"Name": "Enstrophy dissipation",                 "File": "enstrophy_diss",   "Operator": dt/Re * inner(curl(omega), curl(omega)) * dx},
    {"Name": "Enstrophy convective generation",       "File": "enstrophy_gen",    "Operator": dt/2 * inner(cross(u + u_prev, omega), curl(omega)) * dx},
    {"Name": "Divergence of alpha (L2 norm)",         "File": "divergence_alpha", "Operator": inner(div(alpha), div(alpha)) * dx},
    {"Name": "Error in curl omega = alpha (L2 norm)", "File": "omega_error",      "Operator": inner(curl(omega) - alpha, curl(omega) - alpha) * dx},
    {"Name": "Lagrange multiplier (L2 norm)",         "File": "lagrange_mult",    "Operator": inner(r, r) * dx}
]

def print_write_qoi(qoi_name, qoi_file, qoi_operator, write_type):
    qoi = assemble(qoi_operator)
    print(BLUE % f"{qoi_name}: {qoi}")
    path = output_base / "qois" / f"{qoi_file}.txt"
    with path.open(write_type) as f:
        f.write(str(qoi) + "\n")

def print_write(write_type):
    for qoi in qois_cts:
        print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)
    for qoi in qois_discts:
        if write_type == "w":
            path = output_base / "qois" / f"{qoi['File']}.txt"
            with path.open("w") as f:
                f.write("No data for discontinuous QoI at initial condition")
        else:
            print_write_qoi(qoi["Name"], qoi["File"], qoi["Operator"], write_type)

# ──────────────────────────────────────────────────────────────────────────────
# [ADDED] 3D SPECTRUM UTILITIES (read-only QoIs; solver unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def _tukey3d(N, alpha=0.15):
    """[ADDED] 3-D Tukey window (RMS-normalized) to reduce leakage; alpha=0.15 keeps more mid-band energy."""
    n = np.arange(N)
    w = np.ones(N)
    L = alpha*(N-1)
    if alpha > 0:
        m = n < L
        w[m] = 0.5*(1 + np.cos(np.pi*(2*n[m]/(alpha*(N-1)) - 1)))
        m = n > (N-1-L)
        w[m] = 0.5*(1 + np.cos(np.pi*(2*(n[m]-(N-1))/(alpha*(N-1)) + 1)))
    W = w[:,None,None]*w[None,:,None]*w[None,None,:]
    return W / np.sqrt(np.mean(W**2))

def sample_velocity_to_grid(u_x_fun, u_y_fun, u_z_fun, Ngrid=96, a=0.0, b=1.0):
    """
    [ADDED] Evaluate FE velocity on a uniform [a,b]^3 grid for FFTs (read-only).
    """
    xg = np.linspace(a, b, Ngrid, endpoint=False)
    yg = np.linspace(a, b, Ngrid, endpoint=False)
    zg = np.linspace(a, b, Ngrid, endpoint=False)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Firedrake .at may return a list; cast to ndarray then reshape
    ux = np.asarray(u_x_fun.at(pts), dtype=float).reshape(Ngrid, Ngrid, Ngrid)
    uy = np.asarray(u_y_fun.at(pts), dtype=float).reshape(Ngrid, Ngrid, Ngrid)
    uz = np.asarray(u_z_fun.at(pts), dtype=float).reshape(Ngrid, Ngrid, Ngrid)

    dxg = (b - a) / Ngrid
    return ux, uy, uz, dxg

def energy_spectrum_3d(ux, uy, uz, dxg, pad_factor=2, nlog=48):
    """
    [ADDED] Isotropic E(k):
      - Tukey window + de-mean (reduce leakage in nonperiodic box)
      - optional zero-padding (smoother spectrum)
      - 3D FFT
      - log-spaced spherical shells: sum modal energy per shell / Δk (density)
      - Parseval-consistent scaling (accounts for dxg and FFT normalization)
    """
    N = ux.shape[0]
    W = _tukey3d(N, alpha=0.15)  # slightly milder than 0.25 to preserve mid-band
    ux = (ux - np.mean(ux)) * W
    uy = (uy - np.mean(uy)) * W
    uz = (uz - np.mean(uz)) * W

    if pad_factor > 1:
        Npad = pad_factor * N
        pad = Npad - N
        pad3 = ((0, pad), (0, pad), (0, pad))
        ux = np.pad(ux, pad3, mode='constant')
        uy = np.pad(uy, pad3, mode='constant')
        uz = np.pad(uz, pad3, mode='constant')
    else:
        Npad = N

    uxh = fftn(ux); uyh = fftn(uy); uzh = fftn(uz)

    k1 = 2*np.pi*fftfreq(Npad, d=dxg)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Parseval-consistent modal energy (numpy FFT has no 1/N scaling)
    mode_energy = 0.5 * (np.abs(uxh)**2 + np.abs(uyh)**2 + np.abs(uzh)**2) * (dxg**3) / (Npad**3)

    Kpos = Kmag[Kmag > 0]
    if Kpos.size == 0:
        return np.array([]), np.array([])
    kmin, kmax = Kpos.min(), Kpos.max()
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nlog+1)
    E_log = np.zeros(nlog, dtype=float)
    for i in range(nlog):
        mask = (Kmag >= edges[i]) & (Kmag < edges[i+1])
        if np.any(mask):
            dk_i = edges[i+1] - edges[i]
            E_log[i] = np.sum(mode_energy[mask]) / dk_i
    k_bins = np.sqrt(edges[:-1] * edges[1:])  # geometric centers
    return k_bins, E_log

def compute_epsilon_from_enstrophy_current_state():
    """
    [ADDED] Dissipation rate from FE state: ε = ν ∫|ω|^2 dV, with ν=1/Re.
    """
    u_vec_now = as_vector([upabor.subfunctions[0], upabor.subfunctions[1], upabor.subfunctions[2]])
    enstrophy_half = assemble(0.5 * inner(curl(u_vec_now), curl(u_vec_now)) * dx)  # 0.5 ∫|ω|^2
    eps = (2.0 * float(enstrophy_half)) / float(Re)  # volume of [0,1]^3 is 1
    return eps

def qoi_kolmogorov_spectrum(Ngrid=96, pad_factor=2, nlog=48, tag="", check_parseval=True):
    """
    [ADDED] Driver QoI:
      1) Sample FE u on uniform grid
      2) Build E(k) with energy_spectrum_3d
      3) Compute ε from FE
      4) Save NPZ/TXT (+ compensated file)
      5) Fit inertial-range slope using an ADAPTIVE compensated-plateau search
      6) (optional) Print Parseval check vs FE kinetic energy
    """
    # 1) sample
    ux, uy, uz, dxg = sample_velocity_to_grid(
        upabor.subfunctions[0],
        upabor.subfunctions[1],
        upabor.subfunctions[2],
        Ngrid=Ngrid
    )
    # 2) spectrum
    k_bins, Ek = energy_spectrum_3d(ux, uy, uz, dxg, pad_factor=pad_factor, nlog=nlog)
    # 3) epsilon
    eps = compute_epsilon_from_enstrophy_current_state()

    # 4) save spectra
    npz_path = output_base / "qois" / f"spectrum3d{tag}.npz"
    np.savez(npz_path, k=k_bins, Ek=Ek, epsilon=eps, Ngrid=Ngrid, pad_factor=pad_factor, nlog=nlog)
    txt_path = output_base / "qois" / f"spectrum3d{tag}.txt"
    with txt_path.open("w") as f:
        f.write("# k  E(k)\n")
        for ki, Ei in zip(k_bins, Ek):
            f.write(f"{ki} {Ei}\n")

    # [ADDED] also save compensated curve for quick plotting
    mpos = (k_bins > 0) & (Ek > 0)
    comp_path = output_base / "qois" / f"spectrum3d_comp{tag}.txt"
    with comp_path.open("w") as f:
        f.write("# k   k^{5/3}E(k)\n")
        for ki, Ei in zip(k_bins[mpos], Ek[mpos]):
            f.write(f"{ki} {(ki**(5/3))*Ei}\n")

    # 5) ADAPTIVE inertial-range slope via compensated-plateau search (avoids dissipation tail)
    if np.count_nonzero(mpos) >= 20:
        kk = k_bins[mpos]; EE = Ek[mpos]

        # Ignore the first/last ~15% of bins (energy bump & dissipation tail)
        i0 = int(0.15 * kk.size)
        i1 = int(0.85 * kk.size)
        if i1 - i0 >= 10:
            kk_c = kk[i0:i1]; EE_c = EE[i0:i1]
            comp = (kk_c**(5.0/3.0)) * EE_c

            # Sliding window; pick flattest (lowest coefficient of variation)
            W = max(8, kk_c.size // 8)
            best_j, best_cv = None, np.inf
            for j in range(0, kk_c.size - W):
                seg = comp[j:j+W]
                mu = np.mean(seg)
                if mu <= 0:
                    continue
                cv = np.std(seg) / mu
                if cv < best_cv:
                    best_cv, best_j = cv, j

            if best_j is not None:
                # Map selection back to kk,EE
                start = i0 + best_j
                stop  = start + W
                k_sel = k_bins[mpos][start:stop]
                E_sel = Ek[mpos][start:stop]
                slope, _ = np.polyfit(np.log10(k_sel), np.log10(E_sel), 1)
                print(GREEN % (
                    f"[E(k)] slope ≈ {slope:.3f} (target −5/3 ≈ {-5/3:.3f}); "
                    f"bins={k_sel.size}; k∈[{k_sel[0]:.3e},{k_sel[-1]:.3e}]"
                ))
            else:
                print(YELLOW % "[E(k)] could not find a stable compensated plateau for fitting.")
        else:
            print(YELLOW % "[E(k)] spectrum not wide enough to form a robust middle band.")
    else:
        print(YELLOW % "[E(k)] spectrum too short to fit.")

    # 6) optional Parseval consistency check (FFT integral vs FE KE)
    if check_parseval:
        KE_fft = np.trapz(Ek[mpos], k_bins[mpos])
        KE_fe  = float(assemble(0.5 * inner(u, u) * dx))
        print(BLUE % f"[Parseval] KE_fft≈{KE_fft:.6e}, KE_FE≈{KE_fe:.6e} (expect close up to windowing loss)")

# ─── IC setup  (unchanged) ────────────────────────────────────────────────────
u_vortex = hill([x-0.5, y-0.5, z-0.5], 0.25)
F_ic = (inner(u_ic - u_vortex, v_ic) - inner(p_ic, div(v_ic)) - inner(div(u_ic), q_ic)) * dx
bcs_ic = [DirichletBC(VQ_ic.sub(index), 0, surface) for (index, surface) in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]]
sp_ic = {"snes_converged_reason": None, "snes_linesearch_monitor": None, "snes_monitor": None,
         "ksp_monitor": None, "ksp_converged_reason": None, "ksp_monitor_true_residual": None}

print(GREEN % "Setting up ICs:")
solve(F_ic==0, up_ic, bcs=bcs_ic, solver_parameters=sp_ic)

# Normalize previous velocity to unit sqrt-energy  (unchanged)
sqrt_energy_ = sqrt(assemble(1/2 * inner(u_ic, u_ic) * dx))
for i in range(3):
    u_prev.sub(i).assign(up_ic.sub(i) / sqrt_energy_)

# ─── Collect initial data (unchanged) ─────────────────────────────────────────
for i in range(3):
    upabor.sub(i).assign(u_prev.sub(i))
pvd_cts.write(u_x_out, u_y_out, u_z_out)
print_write("w")

# [ADDED] Initial spectrum at t=0 (may not show a full inertial range yet)
qoi_kolmogorov_spectrum(Ngrid=96, pad_factor=2, nlog=48, tag="_t0")

# ─── Full solve loop (unchanged solver) ───────────────────────────────────────
u_mid = 1/2 * (u + u_prev)
F = (
    (  1/dt * inner(u - u_prev, v)
     - inner(cross(u_mid, omega), v)
     + 1/Re * inner(alpha, v)
     - inner(p, div(v)) )
  + ( - inner(div(u), q) )
  + (  inner(alpha, gamma)
     - inner(curl(u_mid), curl(gamma))
     - inner(beta, div(gamma)) )
  + ( - inner(div(alpha), delta) )
  + (  inner(curl(omega), curl(chi))
     - inner(alpha, curl(chi))
     + inner(grad(r), chi) )
  + (  inner(omega, grad(s)) )
) * dx

bcs = [DirichletBC(VQVQWR.sub(index), 0, surface) for (index, surface) in
       [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),
        (4,1),(4,2),(5,3),(5,4),(6,5),(6,6),
        (8,"on_boundary"),(9,"on_boundary")]]

sp = {"snes_converged_reason": None, "snes_linesearch_monitor": None, "snes_monitor": None,
      "ksp_monitor": None, "ksp_converged_reason": None, "ksp_monitor_true_residual": None}

t = 0.0
step_count = 0  # [ADDED] throttle spectrum calls (expensive)
while t <= final_t - float(dt)/2:
    t += float(dt)
    step_count += 1
    print(GREEN % f"Solving for time t = {t}:")
    for i in range(3):
        u_prev.sub(i).assign(upabor.sub(i))
    solve(F==0, upabor, bcs=bcs, solver_parameters=sp)

    # outputs/QoIs (unchanged)
    pvd_cts.write(u_x_out, u_y_out, u_z_out)
    pvd_discts.write(p_out, alpha_x_out, alpha_y_out, alpha_z_out, beta_out, omega_out, r_out)
    print_write("a")

    # [ADDED] Spectrum every few steps (adjust cadence/Ngrid as needed)
    if step_count % 5 == 0:
        qoi_kolmogorov_spectrum(Ngrid=96, pad_factor=2, nlog=48, tag=f"_t{t:.6f}")
