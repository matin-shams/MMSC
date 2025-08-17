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
    directory = base / "meevc_output" / "qois"
    aux = load_series(directory, "auxiliary_enstrophy")
    if aux is None:
        print("No MEEVC auxiliary enstrophy data found.")
        return
    times = np.arange(len(aux)) * DT

    plt.figure()
    plt.plot(times, aux, label="MEEVC")
    plt.xlabel("Time")
    plt.ylabel("Auxiliary Enstrophy")
    plt.title("MEEVC Auxiliary Enstrophy")
    plt.legend()
    plt.savefig(base / "meevc_auxiliary_enstrophy.png")

if __name__ == "__main__":
    main()