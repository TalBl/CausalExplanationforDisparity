import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Original scores dictionary
scores = {
    "SO": {"FCI": 0.3113, "LINGAM": 0.3113, "GES": 0.3113, "INTEGRATION": 0.3113, "DEFAULT": 0.7},
    "MEPS": {"FCI": 0.3901, "LINGAM": 0.386, "GES": 0.3665, "INTEGRATION": 0.430, "DEFAULT": 0.57},
    "ACS": {"FCI": 0.386, "LINGAM": 0.399, "GES": 0.338, "INTEGRATION": 0.399, "DEFAULT": 0.5}
}

# Convert to DataFrame
data = []
for dataset, method_scores in scores.items():
    for method, value in method_scores.items():
        data.append({'Dataset': dataset, 'Method': method, 'Score': value})
df = pd.DataFrame(data)

# Set Seaborn style
sns.set(style="whitegrid")

# Create the barplot
plt.figure(figsize=(8, 6))
barplot = sns.barplot(
    data=df,
    x='Dataset',
    y='Score',
    hue='Method',
    hatch='/',
    edgecolor='black'
)

# Apply hatch patterns manually
hatches = ['/', '\\', 'x', '+', '|']
bars = barplot.patches
num_methods = df['Method'].nunique()
for i, bar in enumerate(bars):
    hatch = hatches[i % num_methods]
    bar.set_hatch(hatch)

# Set labels and formatting
plt.ylabel("Score", fontsize=16)
plt.xlabel("Dataset", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Move the legend above the plot
plt.legend(
    title='',
    fontsize=14,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False
)

plt.tight_layout()  # leave space at the top for legend
plt.savefig("outputs/dags/causalDagExp.pdf")
