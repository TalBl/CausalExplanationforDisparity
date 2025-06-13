import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for name in ['so', 'acs', 'meps']:
    # ExDis
    jaccard_matrix = pd.read_csv(f"outputs/{name}/jaccard_matrix.csv", index_col=0)
    plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = sns.heatmap(
        jaccard_matrix,
        annot=False,
        cmap="Greys",
        linewidths=1.5,        # thickness between cells
        linecolor='black',     # border color
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1
    )

    # Optional: draw a border around the entire heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    plt.savefig(f"outputs/heatmaps/{name}_exdis.jpg")

    # Brute Force
    jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/naive_jaccard_matrix.csv", index_col=0)
    plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = sns.heatmap(
        jaccard_matrix,
        annot=False,
        cmap="Greys",
        linewidths=1.5,        # thickness between cells
        linecolor='black',     # border color
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1
    )

    # Optional: draw a border around the entire heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    plt.savefig(f"outputs/heatmaps/{name}_bruteforce.jpg")

    # Top-K
    jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/top_k_jaccard_matrix.csv", index_col=0)
    plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = sns.heatmap(
        jaccard_matrix,
        annot=False,
        cmap="Greys",
        linewidths=1.5,        # thickness between cells
        linecolor='black',     # border color
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        vmin=0,
        vmax=1
    )

    # Optional: draw a border around the entire heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    plt.savefig(f"outputs/heatmaps/{name}_topk.jpg")

    # DE
    import os
    path = f"outputs/{name}/baselines/de_jaccard_matrix.csv"
    if os.path.exists(path):
        jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/de_jaccard_matrix.csv", index_col=0)
        if jaccard_matrix.shape[0] > 1:
            plt.figure(figsize=(10, 8), constrained_layout=True)
            ax = sns.heatmap(
                jaccard_matrix,
                annot=False,
                cmap="Greys",
                linewidths=1.5,        # thickness between cells
                linecolor='black',     # border color
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                vmin=0,
                vmax=1
            )

            # Optional: draw a border around the entire heatmap
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            plt.savefig(f"outputs/heatmaps/{name}_de.jpg")
        else:
            # Show placeholder plot
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
            plt.axis('off')
            plt.savefig(f"outputs/heatmaps/{name}_de.jpg")

    # RF
    path = f"outputs/{name}/baselines/rf_jaccard_matrix.csv"
    if os.path.exists(path):
        jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/rf_jaccard_matrix.csv", index_col=0)
        if jaccard_matrix.shape[0] > 1:
            plt.figure(figsize=(10, 8), constrained_layout=True)
            ax = sns.heatmap(
                jaccard_matrix,
                annot=False,
                cmap="Greys",
                linewidths=1.5,        # thickness between cells
                linecolor='black',     # border color
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                vmin=0,
                vmax=1
            )

            # Optional: draw a border around the entire heatmap
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            plt.savefig(f"outputs/heatmaps/{name}_rf.jpg")
        else:
            # Show placeholder plot
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
            plt.axis('off')
            plt.savefig(f"outputs/heatmaps/{name}_rf.jpg")

