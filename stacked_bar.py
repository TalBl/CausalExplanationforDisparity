import matplotlib.pyplot as plt
import numpy as np

# data from https://allisonhorst.github.io/palmerpenguins/

species = (
    "SO",
    "MEPS",
)
weight_counts = {
    "Step1": np.array([0.4, 0.33]),
    "Step2": np.array([208.8, 271.5]),
    "Step3": np.array([33, 58]),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(2)

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.legend(loc="upper right")

plt.show()