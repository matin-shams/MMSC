import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DT = 2 ** (-8)


def load_series(directory, name):
    path = directory / f"{name}.txt"
    if not path.exists():
        print(f"Missing {path}, skipping.")
        return None
    return np.loadtxt(path)


def main():
    base = Path(__file__).parent
    datasets = {
        "Boris": base / "boris_output" / "qois",
        "MEEVC": base / "meevc_output" / "qois",
        "Classical": base / "classical_output" / "qois",
    }
    series = {}
    for name, directory in datasets.items():
        broken = load_series(directory, "broken_enstrophy")
        internal = load_series(directory, "internal_enstrophy")
        aux = load_series(directory, "auxiliary_enstrophy")
        if broken is None or internal is None or aux is None:
            print(f"Skipping plots for {name}; missing data.")
            continue
        times = np.arange(len(broken)) * DT
        series[name] = {
            "times": times,
            "broken": broken,
            "internal": internal,
            "aux": aux,
        }

    if not series:
        print("No data to plot. Run the Boris, MEEVC, and Classical solvers first.")
        return

    # Plot broken and internal enstrophy
    plt.figure()
    for name, data in series.items():
        plt.plot(data["times"], data["broken"], label=f"{name} broken")
        plt.plot(data["times"], data["internal"], linestyle="--", label=f"{name} internal")
    plt.xlabel("Time")
    plt.ylabel("Enstrophy")
    plt.title("Broken vs Internal Enstrophy")
    plt.legend()
    plt.savefig(base / "enstrophy_comparison.png")

    # Plot auxiliary enstrophy
    plt.figure()
    for name, data in series.items():
        plt.plot(data["times"], data["aux"], label=name)
    plt.xlabel("Time")
    plt.ylabel("Auxiliary Enstrophy")
    plt.title("Auxiliary Enstrophy")
    plt.legend()
    plt.savefig(base / "auxiliary_enstrophy.png")


if __name__ == "__main__":
    main()