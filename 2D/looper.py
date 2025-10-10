from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from convergence import run_case


def _polyfit_loglog(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    logx = np.log(np.array(list(xs), dtype=float))
    logy = np.log(np.array(list(ys), dtype=float))
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def _write_table(path: Path, header: str, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(header + "\n")
        for a, b in rows:
            fh.write(f"{a:.8e} {b:.8e}\n")


def _parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",")] if s else []


# ---------------------- TIME: multi-curve (different stages) ------------------
def _time_convergence(args):
    stages_list = _parse_list(args.stages_list, int) or [1]
    dts = [args.initial_dt / 2 ** level for level in range(args.num_runs)]

    output_dir = args.output_dir / "time"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    last_slope = None

    for S in stages_list:
        errors = []
        for dt in dts:
            r = run_case(
                N=args.N,
                dt_value=dt,
                final_time=args.final_time,
                Re_value=args.Re,
                time_mode="transient",
                stages=S,
            )
            errors.append(r.error)

        slope, intercept = _polyfit_loglog(dts, errors)
        last_slope = slope

        _write_table(output_dir / f"data_S{S}.txt", "# dt error", zip(dts, errors))
        ax.loglog(dts, errors, "o-", label=f"S={S}  (slope≈{slope:.2f})")

        # Fitted reference line (for visual confirmation)
        ref_line = np.exp(intercept) * (np.array(dts) ** slope)
        ax.loglog(dts, ref_line, "k--", alpha=0.35)

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\lVert u_h(T) - u_{ex}(T) \rVert_{L^2}$")
    ax.set_title(f"Temporal convergence (Gauss–Legendre, N={args.N})")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150)
    return last_slope if last_slope is not None else float("nan")


# ---------------------- SPACE: single curve (steady MMS) ----------------------
def _space_convergence(args):
    hs = []
    errors = []
    for level in range(args.num_runs):
        N = args.initial_N * 2 ** level
        h = 1.0 / N
        r = run_case(
            N=N,
            dt_value=args.dt,
            final_time=args.final_time,
            Re_value=args.Re,
            time_mode="steady",
            stages=1,  # stage irrelevant here since dt is tiny; keep 1
        )
        hs.append(h)
        errors.append(r.error)

    slope, intercept = _polyfit_loglog(hs, errors)

    output_dir = args.output_dir / "space"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_table(output_dir / "data.txt", "# h error", zip(hs, errors))

    fig, ax = plt.subplots()
    ax.loglog(hs, errors, "o-", label=f"data (slope≈{slope:.2f})")
    ref_line = np.exp(intercept) * (np.array(hs) ** slope)
    ax.loglog(hs, ref_line, "k--", label="fit", alpha=0.35)

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\lVert u_h(T) - u_{ex}(T) \rVert_{L^2}$")
    ax.set_title("Spatial convergence (manufactured solution)")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150)
    return slope


# ---------------------- CLI ---------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Manufactured-solution convergence for the Boris scheme")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    time_parser = subparsers.add_parser("time", help="Temporal convergence study (multi-stage)")
    time_parser.add_argument("--N", type=int, default=64, help="Number of elements in each direction")
    time_parser.add_argument("--initial-dt", type=float, default=2 ** -7, help="Initial time-step size")
    time_parser.add_argument("--num-runs", type=int, default=5, help="Number of refinements (dt halved each run)")
    time_parser.add_argument("--stages-list", type=str, default="1", help="Comma-separated Gauss stages, e.g. '1,2,3,4'")

    space_parser = subparsers.add_parser("space", help="Spatial convergence study (steady MMS)")
    space_parser.add_argument("--initial-N", type=int, default=16, help="Coarsest resolution (N doubles each run)")
    space_parser.add_argument("--dt", type=float, default=2 ** -10, help="Time-step size used for spatial study")
    space_parser.add_argument("--num-runs", type=int, default=5, help="Number of refinements")

    for sp in (time_parser, space_parser):
        sp.add_argument("--final-time", type=float, default=1.0, help="Final time T")
        sp.add_argument("--Re", type=float, default=4.0, help="Reynolds number")
        sp.add_argument(
            "--output-dir",
            type=Path,
            default=Path("output") / "mms_convergence",
            help="Directory where tables and figures are written",
        )

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.mode == "time":
        slope = _time_convergence(args)
        print(f"Temporal convergence: last curve slope ≈ {slope:.2f}")
    elif args.mode == "space":
        slope = _space_convergence(args)
        print(f"Spatial convergence: slope ≈ {slope:.2f}")
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
