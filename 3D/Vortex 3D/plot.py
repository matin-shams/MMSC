import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot_compare.py <data_name> [timestep]")
    sys.exit(1)

data_name = sys.argv[1]
dt = float(eval(sys.argv[2])) if len(sys.argv) > 2 else 2**-10

# Define consistent colors for each method
color_map = {
    "Classical": "tab:blue",
    "Boris": "tab:orange",
}

paths = [
    ("Classical", f"output/classical/{data_name}.txt", "-", color_map["Classical"]),
    ("Boris", f"output/boris/{data_name}.txt", "-", color_map["Boris"]),
    ("Classical (IE)", f"output/classical_ie/{data_name}.txt", "--", color_map["Classical"]),
    ("Boris (IE)", f"output/boris_ie/{data_name}.txt", "--", color_map["Boris"]),
]

plt.figure()
for label, path, linestyle, color in paths:
    try:
        with open(path) as f:
            data = [float(line.strip()) for line in f if line.strip()]
        timesteps = np.arange(len(data)) * dt
        plt.plot(timesteps, data, label=label, linestyle=linestyle, color=color)
    except FileNotFoundError:
        print(f"File not found: {path}")
        continue

plt.title(f"Comparison of {data_name.capitalize()}")
plt.xlabel("Time")
plt.ylabel(data_name.capitalize())
plt.legend()
plt.tight_layout()
plt.show()
