import csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Utils import get_indices
from algorithms.final_algorithm.new_greedy import print_matrix


def calc_jaacard_matrix(name_dataset, df_results_path, output_path):
    if name_dataset != 'acs':
        df_clean = pd.read_csv(f"outputs/{name_dataset}/clean_data.csv")
    else:
        df_clean = pd.read_csv(f"outputs/{name_dataset}/sample_data.csv")
    df_results = pd.read_csv(df_results_path)
    df_results['indices'] = df_results['subpop'].apply(get_indices, data=df_clean)
    df_results = df_results.sort_values(by='score', ascending=False)
    jaccard_matrix = print_matrix({}, {}, [[x['subpop'], x['indices']] for _, x in df_results.iterrows()])
    jaccard_matrix.to_csv(f"tmp_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    jaccard_matrix = pd.read_csv(f"tmp_jaccard_matrix.csv", index_col=0)
    plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = sns.heatmap(
        jaccard_matrix,
        annot=False,
        cmap="Greys",
        linecolor='black',  # border color
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
    plt.savefig(output_path)


for name in ['acs', 'meps']:
    calc_jaacard_matrix(name, f"outputs/{name}/clustering_results.csv", f"outputs/heatmaps/{name}_exdis.jpg")
    calc_jaacard_matrix(name, f"outputs/{name}/baselines/facts_naive.csv", f"outputs/heatmaps/{name}_bruteforce.jpg")
    calc_jaacard_matrix(name, f"outputs/{name}/baselines/facts_top_k.csv", f"outputs/heatmaps/{name}_topk.jpg")
    df = pd.read_csv(f"outputs/{name}/baselines/facts_de.csv")
    if df.shape[0] > 0:
        calc_jaacard_matrix(name, f"outputs/{name}/baselines/facts_de.csv", f"outputs/heatmaps/{name}_de.jpg")
    else:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
        plt.axis('off')
        plt.savefig(f"outputs/heatmaps/{name}_de.jpg")
    df = pd.read_csv(f"outputs/{name}/baselines/facts_rf.csv")
    if df.shape[0] > 0:
        calc_jaacard_matrix(name, f"outputs/{name}/baselines/facts_final_rf.csv", f"outputs/heatmaps/{name}_rf.jpg")
    else:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
        plt.axis('off')
        plt.savefig(f"outputs/heatmaps/{name}_rf.jpg")
    # ExDis
    # jaccard_matrix = pd.read_csv(f"outputs/{name}/jaccard_matrix.csv", index_col=0)
    # jaccard_matrix = jaccard_matrix.sort_index().sort_index(axis=1)
    #
    # plt.figure(figsize=(10, 8), constrained_layout=True)
    # ax = sns.heatmap(
    #     jaccard_matrix,
    #     annot=False,
    #     cmap="Greys",
    #     linecolor='black',     # border color
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=False,
    #     vmin=0,
    #     vmax=1
    # )
    #
    # # Optional: draw a border around the entire heatmap
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    #     spine.set_color('black')
    #     spine.set_linewidth(2)
    # plt.savefig(f"outputs/heatmaps/{name}_exdis.jpg")

    # Brute Force
    # jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/naive_jaccard_matrix.csv", index_col=0)
    # jaccard_matrix = jaccard_matrix.sort_index().sort_index(axis=1)
    #
    # plt.figure(figsize=(10, 8), constrained_layout=True)
    # ax = sns.heatmap(
    #     jaccard_matrix,
    #     annot=False,
    #     cmap="Greys",
    #     linecolor='black',     # border color
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=False,
    #     vmin=0,
    #     vmax=1
    # )
    #
    # # Optional: draw a border around the entire heatmap
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    #     spine.set_color('black')
    #     spine.set_linewidth(2)
    # plt.savefig(f"outputs/heatmaps/{name}_bruteforce.jpg")

    # Top-K
    # jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/top_k_jaccard_matrix.csv", index_col=0)
    # jaccard_matrix = jaccard_matrix.sort_index().sort_index(axis=1)
    #
    # plt.figure(figsize=(10, 8), constrained_layout=True)
    # ax = sns.heatmap(
    #     jaccard_matrix,
    #     annot=False,
    #     cmap="Greys",
    #     linecolor='black',     # border color
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=False,
    #     vmin=0,
    #     vmax=1
    # )
    #
    # # Optional: draw a border around the entire heatmap
    # for _, spine in ax.spines.items():
    #     spine.set_visible(True)
    #     spine.set_color('black')
    #     spine.set_linewidth(2)
    # plt.savefig(f"outputs/heatmaps/{name}_topk.jpg")

    # DE
    # import os
    # path = f"outputs/{name}/baselines/de_jaccard_matrix.csv"
    # if os.path.exists(path):
    #     jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/de_jaccard_matrix.csv", index_col=0)
    #     jaccard_matrix = jaccard_matrix.sort_index().sort_index(axis=1)
    #
    #     if jaccard_matrix.shape[0] > 1:
    #         plt.figure(figsize=(10, 8), constrained_layout=True)
    #         ax = sns.heatmap(
    #             jaccard_matrix,
    #             annot=False,
    #             cmap="Greys",
    #             linecolor='black',     # border color
    #             xticklabels=False,
    #             yticklabels=False,
    #             cbar=False,
    #             vmin=0,
    #             vmax=1
    #         )
    #
    #         # Optional: draw a border around the entire heatmap
    #         for _, spine in ax.spines.items():
    #             spine.set_visible(True)
    #             spine.set_color('black')
    #             spine.set_linewidth(2)
    #         plt.savefig(f"outputs/heatmaps/{name}_de.jpg")
    #     else:
    #         # Show placeholder plot
    #         plt.figure(figsize=(10, 8))
    #         plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
    #         plt.axis('off')
    #         plt.savefig(f"outputs/heatmaps/{name}_de.jpg")

    # RF
    # path = f"outputs/{name}/baselines/rf_jaccard_matrix.csv"
    # if os.path.exists(path):
    #     jaccard_matrix = pd.read_csv(f"outputs/{name}/baselines/rf_jaccard_matrix.csv", index_col=0)
    #     jaccard_matrix = jaccard_matrix.sort_index().sort_index(axis=1)
    #
    #     if jaccard_matrix.shape[0] > 1:
    #         plt.figure(figsize=(10, 8), constrained_layout=True)
    #         ax = sns.heatmap(
    #             jaccard_matrix,
    #             annot=False,
    #             cmap="Greys",
    #             linecolor='black',     # border color
    #             xticklabels=False,
    #             yticklabels=False,
    #             cbar=False,
    #             vmin=0,
    #             vmax=1
    #         )
    #
    #         # Optional: draw a border around the entire heatmap
    #         for _, spine in ax.spines.items():
    #             spine.set_visible(True)
    #             spine.set_color('black')
    #             spine.set_linewidth(2)
    #         plt.savefig(f"outputs/heatmaps/{name}_rf.jpg")
    #     else:
    #         # Show placeholder plot
    #         plt.figure(figsize=(10, 8))
    #         plt.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=90)
    #         plt.axis('off')
    #         plt.savefig(f"outputs/heatmaps/{name}_rf.jpg")

