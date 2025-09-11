# plot_compare.py
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_compare.py <data_name>")
    sys.exit(1)

data_name = sys.argv[1]
paths = [
    ("Classical", f"output/classical/{data_name}.txt"),
    ("Boris", f"output/boris/{data_name}.txt"),
]

plt.figure()
for label, path in paths:
    try:
        with open(path) as f:
            data = [float(line.strip()) for line in f if line.strip()]
        plt.plot(data, label=label)
    except FileNotFoundError:
        print(f"File not found: {path}")
        continue

plt.title(f"Comparison of {data_name.capitalize()}")
plt.xlabel("Timestep")
plt.ylabel(data_name.capitalize())
plt.legend()
plt.tight_layout()
plt.show()

