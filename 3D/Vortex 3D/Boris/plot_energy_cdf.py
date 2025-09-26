#!/usr/bin/env python3
"""
Plot the cumulative distribution function (CDF) of modal energy vs. length scale.

Length scale is taken as L = sqrt(a^2 + b^2 + c^2), per the user's definition,
where (a, b, c) are the modal indices used in the Fourier modal energy output.

Input: HDF5 file produced by fourier_modal_energy.write_modal_energies_h5 with datasets:
  - Ex[T, A, B, C], Ey[T, A, B, C], Ez[T, A, B, C]
  - time[T]

Outputs:
  - PNG (and optionally CSV) of CDF curve(s)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py


def compute_cdf_for_snapshot(Eabc: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute CDF of energy vs. length scale L = sqrt(a^2+b^2+c^2) for a single snapshot.

    Returns (L_sorted, CDF, total_energy).
    """
    if Eabc.ndim != 3:
        raise ValueError(f"Expecting 3D array for energies, got shape {Eabc.shape}")
    A, B, C = Eabc.shape
    a = np.arange(A)[:, None, None]
    b = np.arange(B)[None, :, None]
    c = np.arange(C)[None, None, :]
    L = np.sqrt(a*a + b*b + c*c)

    E_flat = Eabc.ravel().astype(float)
    L_flat = L.ravel().astype(float)

    total = float(E_flat.sum())
    if total <= 0:
        return L_flat, np.zeros_like(L_flat), 0.0

    order = np.argsort(L_flat)
    L_sorted = L_flat[order]
    E_sorted = E_flat[order]
    CDF = np.cumsum(E_sorted) / total
    return L_sorted, CDF, total


def main():
    parser = argparse.ArgumentParser(description="Plot CDF of energy vs. modal length scale from HDF5 modal energies")
    parser.add_argument("--h5", type=Path, default=None, help="Path to fourier_modes.h5 (default: output/im_spectrum/fourier_modes.h5 relative to this script)")
    parser.add_argument("--time-index", type=int, default=None, help="Time index to plot (default: last)")
    parser.add_argument("--all-times", action="store_true", help="Plot CDF for all time snapshots")
    parser.add_argument("--save-csv", action="store_true", help="Also save CSV with L and CDF values")
    parser.add_argument("--out", type=Path, default=None, help="Output image path (default: alongside H5)")
    args = parser.parse_args()

    # Default H5 path relative to this script to match im_spectrum.py output
    if args.h5 is None:
        default_h5 = Path(__file__).parent / "output" / "im_spectrum" / "fourier_modes.h5"
        h5_path = default_h5
    else:
        h5_path = args.h5

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        Ex = h5["Ex"]  # shape (T, A, B, C)
        Ey = h5["Ey"]
        Ez = h5["Ez"]
        time = h5["time"][:] if "time" in h5 else np.arange(Ex.shape[0])

        T = Ex.shape[0]
        if T == 0:
            raise RuntimeError("No snapshots found in HDF5 file.")

        # Prepare output path
        out_base = (args.out if args.out is not None else (h5_path.parent / "energy_cdf.png")).with_suffix("")

        # Import matplotlib lazily to avoid dependency if only CSV is needed
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            plt = None
            print(f"matplotlib not available ({e}), will skip PNG plotting and only write CSV if requested.")

        if args.all_times:
            # Plot all snapshots
            if plt is not None:
                fig, ax = plt.subplots(figsize=(7, 5))
            for t_idx in range(T):
                Eabc = Ex[t_idx] + Ey[t_idx] + Ez[t_idx]
                L_sorted, CDF, total = compute_cdf_for_snapshot(Eabc)
                t_val = float(time[t_idx]) if t_idx < len(time) else t_idx

                if plt is not None:
                    ax.plot(L_sorted, CDF, label=f"t={t_val:.4g} (E={total:.3e})", alpha=0.7)

                if args.save_csv:
                    csv_path = Path(str(out_base) + f"_t{t_val:.6f}.csv")
                    np.savetxt(csv_path, np.column_stack([L_sorted, CDF]), delimiter=",", header="L,CDF", comments="")

            if plt is not None:
                ax.set_xlabel("Length scale L = sqrt(a^2 + b^2 + c^2)")
                ax.set_ylabel("Cumulative energy fraction")
                ax.set_title("Energy CDF vs. length scale")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower right", fontsize=8)
                png_path = Path(str(out_base) + "_all.png")
                fig.tight_layout()
                fig.savefig(png_path, dpi=200)
                print(f"Saved {png_path}")
        else:
            # Single snapshot (default: last)
            t_idx = T - 1 if args.time_index is None else int(args.time_index)
            if t_idx < 0:
                t_idx = T + t_idx
            if not (0 <= t_idx < T):
                raise IndexError(f"time-index {t_idx} out of range [0,{T-1}]")

            Eabc = Ex[t_idx] + Ey[t_idx] + Ez[t_idx]
            L_sorted, CDF, total = compute_cdf_for_snapshot(Eabc)
            t_val = float(time[t_idx]) if t_idx < len(time) else t_idx

            if args.save_csv:
                csv_path = Path(str(out_base) + f"_t{t_val:.6f}.csv")
                np.savetxt(csv_path, np.column_stack([L_sorted, CDF]), delimiter=",", header="L,CDF", comments="")
                print(f"Saved {csv_path}")

            if plt is not None:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(L_sorted, CDF, color="tab:blue")
                ax.set_xlabel("Length scale L = sqrt(a^2 + b^2 + c^2)")
                ax.set_ylabel("Cumulative energy fraction")
                ax.set_title(f"Energy CDF at t={t_val:.4g} (E_total={total:.3e})")
                ax.grid(True, alpha=0.3)
                png_path = Path(str(out_base) + f"_t{t_val:.6f}.png")
                fig.tight_layout()
                fig.savefig(png_path, dpi=200)
                print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
