"""
Assemble-based Fourier modal energy evaluation for Navier-slip-like bases.

Provided families (on [0,1]^3):
  Ex: sin(aπx) cos(bπy) cos(cπz),    a>=1, b>=0, c>=0
  Ey: cos(aπx) sin(bπy) cos(cπz),    a>=0, b>=1, c>=0
  Ez: cos(aπx) cos(bπy) sin(cπz),    a>=0, b>=0, c>=1

Energy per mode φ: E = 0.5 * (⟨u_comp, φ⟩)^2 / ⟨φ, φ⟩, with ⟨·,·⟩ = ∫ ·· dV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import h5py

from firedrake import *



def _mode_norm_factor_sq_x(a: int, b: int, c: int) -> float:
    if a <= 0:
        return 0.0
    ix = 0.5  # ∫_0^1 sin^2(aπx) dx, a>=1
    iy = 1.0 if b == 0 else 0.5  # ∫_0^1 cos^2(bπy) dy
    iz = 1.0 if c == 0 else 0.5  # ∫_0^1 cos^2(cπz) dz
    return ix * iy * iz

def _mode_norm_factor_sq_y(a: int, b: int, c: int) -> float:
    if b <= 0:
        return 0.0
    ix = 1.0 if a == 0 else 0.5  # ∫_0^1 cos^2(aπx) dx
    iy = 0.5  # ∫_0^1 sin^2(bπy) dy, b>=1
    iz = 1.0 if c == 0 else 0.5  # ∫_0^1 cos^2(cπz) dz
    return ix * iy * iz

def _mode_norm_factor_sq_z(a: int, b: int, c: int) -> float:
    if c <= 0:
        return 0.0
    ix = 1.0 if a == 0 else 0.5  # ∫_0^1 cos^2(aπx) dx
    iy = 1.0 if b == 0 else 0.5  # ∫_0^1 cos^2(bπy) dy
    iz = 0.5  # ∫_0^1 sin^2(cπz) dz, c>=1
    return ix * iy * iz



def _phi_x(a: int, b: int, c: int, x, y, z):
    return sin(pi*a*x) * cos(pi*b*y) * cos(pi*c*z)

def _phi_y(a: int, b: int, c: int, x, y, z):
    return cos(pi*a*x) * sin(pi*b*y) * cos(pi*c*z)

def _phi_z(a: int, b: int, c: int, x, y, z):
    return cos(pi*a*x) * cos(pi*b*y) * sin(pi*c*z)



def compute_fourier_modal_energies(mesh, u_x, u_y, u_z, a_max: int, b_max: int, c_max: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute modal energies for Ex, Ey, Ez families on the provided mesh.

    Parameters
    - mesh: Firedrake Mesh
    - u_x, u_y, u_z: UFL scalar expressions (e.g., components from split(up))
    - a_max, b_max, c_max: maximum indices (inclusive)

    Returns
    - Ex, Ey, Ez: numpy arrays of shape (a_max+1, b_max+1, c_max+1)
    """
    x, y, z = SpatialCoordinate(mesh)

    Ex = np.zeros((a_max+1, b_max+1, c_max+1), dtype=float)
    Ey = np.zeros((a_max+1, b_max+1, c_max+1), dtype=float)
    Ez = np.zeros((a_max+1, b_max+1, c_max+1), dtype=float)

    # X-directed modes (a>=1)
    for a in range(1, a_max+1):
        for b in range(0, b_max+1):
            for c in range(0, c_max+1):
                phi = _phi_x(a, b, c, x, y, z)
                ip = assemble(u_x * phi * dx)
                n2 = _mode_norm_factor_sq_x(a, b, c)
                if n2 > 0.0:
                    Ex[a, b, c] = 0.5 * (ip*ip) / n2

    # Y-directed modes (b>=1)
    for a in range(0, a_max+1):
        for b in range(1, b_max+1):
            for c in range(0, c_max+1):
                phi = _phi_y(a, b, c, x, y, z)
                ip = assemble(u_y * phi * dx)
                n2 = _mode_norm_factor_sq_y(a, b, c)
                if n2 > 0.0:
                    Ey[a, b, c] = 0.5 * (ip*ip) / n2

    # Z-directed modes (c>=1)
    for a in range(0, a_max+1):
        for b in range(0, b_max+1):
            for c in range(1, c_max+1):
                phi = _phi_z(a, b, c, x, y, z)
                ip = assemble(u_z * phi * dx)
                n2 = _mode_norm_factor_sq_z(a, b, c)
                if n2 > 0.0:
                    Ez[a, b, c] = 0.5 * (ip*ip) / n2

    return Ex, Ey, Ez



def write_modal_energies_h5(Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray, t: float,
                             a_max: int, b_max: int, c_max: int, path: Path) -> None:
    """Append a snapshot to an HDF5 file with resizable datasets.
    Datasets shapes: (T, a_max+1, b_max+1, c_max+1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode="a") as h5:
        ds_args = dict(maxshape=(None, a_max+1, b_max+1, c_max+1), compression="gzip", compression_opts=4)
        if "Ex" not in h5:
            h5.create_dataset("Ex", shape=(0, a_max+1, b_max+1, c_max+1), dtype="f8", **ds_args)
            h5.create_dataset("Ey", shape=(0, a_max+1, b_max+1, c_max+1), dtype="f8", **ds_args)
            h5.create_dataset("Ez", shape=(0, a_max+1, b_max+1, c_max+1), dtype="f8", **ds_args)
            h5.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8", compression="gzip", compression_opts=4)
            h5.attrs["a_max"] = a_max
            h5.attrs["b_max"] = b_max
            h5.attrs["c_max"] = c_max
            h5.attrs["note"] = (
                "Fourier modal energies for Navier-slip-like basis: "
                "Ex: sin(aπx)cos(bπy)cos(cπz) a>=1; "
                "Ey: cos(aπx)sin(bπy)cos(cπz) b>=1; "
                "Ez: cos(aπx)cos(bπy)sin(cπz) c>=1"
            )
        for name, arr in ("Ex", Ex), ("Ey", Ey), ("Ez", Ez):
            ds = h5[name]
            T = ds.shape[0]
            ds.resize((T+1, a_max+1, b_max+1, c_max+1))
            ds[T, :, :, :] = arr
        ts = h5["time"]
        ts.resize((ts.shape[0]+1,))
        ts[-1] = float(t)



def compute_and_write_modal_energies(mesh, u_x, u_y, u_z, a_max: int, b_max: int, c_max: int,
                                     t: float, path: Path) -> Tuple[float, float]:
    """Compute modal energies and append to HDF5. Returns (total_modal_ke, fe_ke)."""
    Ex, Ey, Ez = compute_fourier_modal_energies(mesh, u_x, u_y, u_z, a_max, b_max, c_max)
    write_modal_energies_h5(Ex, Ey, Ez, t, a_max, b_max, c_max, path)
    total_modal_ke = float(Ex.sum() + Ey.sum() + Ez.sum())
    return total_modal_ke
