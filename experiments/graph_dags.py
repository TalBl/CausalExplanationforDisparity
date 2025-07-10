import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List of methods and their CSV filenames
methods = ["default", "integration", "fci", "ges"]
method_names = {"default": "DEFAULT", "integration": "INTEGRATION", "fci": "FCI", "ges": "GES"}

# Step 1: Load all results into dictionary of Series
results = {}
for method in methods:
    df = pd.read_csv(f"outputs/dags/{method}_results.csv")
    df = df.set_index("dataset")["res"]
    results[method_names[method]] = df

# Step 2: Normalize each method by the integration method
normalized_scores = pd.DataFrame()

for method in method_names.values():
    normalized_scores[method] = results[method] / results["INTEGRATION"]

# Keep datasets as index
normalized_scores["Dataset"] = results["INTEGRATION"].index
df = normalized_scores.melt(id_vars="Dataset", var_name="Method", value_name="Score")

# Step 3: Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

barplot = sns.barplot(
    data=df,
    x='Dataset',
    y='Score',
    hue='Method',
    hatch='/',
    edgecolor='black'
)

# Apply custom hatches
hatches = ['/', '\\', 'x', '+', '|']
bars = barplot.patches
num_methods = df['Method'].nunique()
for i, bar in enumerate(bars):
    hatch = hatches[i % num_methods]
    bar.set_hatch(hatch)

# Set labels and formatting
plt.ylabel("Normalized Score (vs. Integration)", fontsize=16)
plt.xlabel("Dataset", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Move the legend above the plot
plt.legend(
    title='',
    fontsize=14,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.savefig("outputs/dags/causalDagExp.pdf")
