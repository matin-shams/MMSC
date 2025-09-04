import glob, numpy as np, pyvista as pv, matplotlib.pyplot as plt, os

# --- load mid-step times & dt ---
times_path = "u_mid/times.txt"
mid_times = np.loadtxt(times_path, ndmin=1)
if len(mid_times) == 0:
    raise RuntimeError("No times in u_mid/times.txt")
if len(mid_times) == 1:
    dt = 2.0 * mid_times[0]
else:
    dt = float(mid_times[1] - mid_times[0])
T = float(mid_times[-1] + 0.5*dt)

# --- read VTU series with PyVista ---
files = sorted(glob.glob("u_mid/u_mid_*.vtu"))
if not files:
    raise RuntimeError("No u_mid_*.vtu files found in u_mid/")
grids = [pv.read(f) for f in files]

# find the vector array name to sample
def pick_vec_name(grid):
    prefer = "u_mid"
    if prefer in grid.point_data: return prefer
    for k in grid.point_data:
        arr = grid.point_data[k]
        if arr.ndim == 2 and arr.shape[1] in (2,3):
            return k
    raise RuntimeError("No vector array found in VTU.")
vecname = pick_vec_name(grids[0])

# --- velocity sampler at step k ---
def vel(k, X):  # X: (N,2); returns (N,2)
    P = pv.PolyData(np.c_[X, np.zeros(len(X))])   # pad z=0 for 2D meshes
    S = grids[k].sample(P)
    V = S.point_data[vecname]
    return V[:, :2]

# --- tracer integrators on a single step (field frozen at mid k) ---
def step_FE(k, X, h):                # Forward Euler
    return X + h*vel(k, X)

def step_IE(k, X, h, iters=6):       # Backward Euler: X+ = X + h u(X+)
    Y = X.copy()
    for _ in range(iters):
        Y = X + h*vel(k, Y)
    return Y

def step_RK2(k, X, h):               # Explicit midpoint (RK2)
    k1 = vel(k, X)
    Xm = X + 0.5*h*k1
    k2 = vel(k, Xm)
    return X + h*k2

def step_IM(k, X, h, iters=6):       # Implicit midpoint: X+ = X + h u((X+X+)/2)
    Y = X.copy()
    for _ in range(iters):
        Ym = 0.5*(X + Y)
        Y = X + h*vel(k, Ym)
    return Y

schemes = {
    "Forward Euler": step_FE,
    "Backward Euler": step_IE,
    "RK2 (explicit midpoint)": step_RK2,
    "Implicit midpoint (symmetric)": step_IM,
}

# --- seed tracers: ring ---
Npts = 50
ctr = np.array([0.5, 0.5])
R = 0.12
theta = np.linspace(0, 2*np.pi, Npts, endpoint=False)
X0 = ctr + R*np.c_[np.cos(theta), np.sin(theta)]

# --- run forward and backward for each scheme ---
snap_tags = [0.0, T/3, 2*T/3, T]
tol = 1e-12

def run_scheme(name, stepper):
    # forward
    X = X0.copy()
    t = 0.0
    fwd = {0.0: X.copy()}
    for k in range(len(grids)):           # one tracer step per u_mid file
        X = stepper(k, X, +dt)
        t += dt
        for tag in snap_tags[1:-1]:
            if tag not in fwd and t >= tag - tol:
                fwd[tag] = X.copy()
    fwd[T] = X.copy()
    # backward
    Xb = X.copy()
    tb = T
    bwd = {T: Xb.copy()}
    for k in reversed(range(len(grids))):
        Xb = stepper(k, Xb, -dt)
        tb -= dt
        for tag in reversed(snap_tags[1:-1]):
            if tag not in bwd and tb <= tag + tol:
                bwd[tag] = Xb.copy()
    bwd[0.0] = Xb.copy()
    return fwd, bwd

results = {name: run_scheme(name, fn) for name, fn in schemes.items()}

# --- plot 2x2 grid: forward dots vs backward crosses ---
fig, axes = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)
for ax, (name, (fwd, bwd)) in zip(axes.ravel(), results.items()):
    for tag in snap_tags:
        F = fwd[tag]; B = bwd[tag]
        ax.plot(F[:,0], F[:,1], "o", ms=4, alpha=0.9, label=f"fwd t={tag:.2f}")
        ax.plot(B[:,0], B[:,1], "x", ms=4, alpha=0.9, label=f"bwd t={tag:.2f}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(name)
    ax.grid(True, lw=0.3, alpha=0.4)
# one legend for all (optional):
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, frameon=False)
fig.suptitle("Tracer forward/backward overlay on u_mid fields", y=0.03)
fig.savefig("tracers_compare.png", dpi=200)
print("Saved tracers_compare.png")
