import matplotlib.pyplot as plt
import numpy as np

scores = {"SO": {"FCI": 0.3682372, "LINGAM": 0.368237, "GES": 0.36823726},
          "MEPS": {"FCI": 0.3682372, "LINGAM": 0.352007, "GES": 0.365332},
          "ACS": {"FCI": 0.7173318, "LINGAM": 0.717332, "GES": 0.454663}}


datasets = list(scores.keys())
methods = ["FCI", "LINGAM", "GES"]
n_datasets = len(datasets)
x = np.arange(n_datasets)  # label locations
width = 0.25  # width of the bars

# Build bar heights for each method
values = {method: [scores[ds][method] if scores[ds][method] is not None else 0 for ds in datasets] for method in methods}
missing = {method: [scores[ds][method] is None for ds in datasets] for method in methods}

# Plotting
fig, ax = plt.subplots()
colors = {'FCI': 'blue', 'LINGAM': 'green', 'GES': 'red'}

for i, method in enumerate(methods):
    bar = ax.bar(x + i*width, values[method], width, label=method, color=colors[method])
    # Optionally mark missing values differently
    for j, missing_val in enumerate(missing[method]):
        if missing_val:
            bar[j].set_color('gray')
            bar[j].set_hatch('//')  # visually indicate missing

# Labels and formatting
ax.set_ylabel('Score')
ax.set_title('Causal Discovery Comparison Across Datasets')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()

plt.tight_layout()
plt.savefig("outputs/dags/causalDagExp.png")
