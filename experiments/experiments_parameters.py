import time
from Utils import Dataset
from algorithms.final_algorithm.clustering2 import clustering
from algorithms.final_algorithm.full import acs, so, meps
import random
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import math
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO


warnings.filterwarnings("ignore")

DEFAULT_K = 5
DEFAULT_SUPPORT = 0.01
DEFAULT_THRESHOLD = 0.55
DEFAULT_NUM_CLUSTERS = 2*DEFAULT_K
df_results = []
NAIVE_RES = {"so": 0.238902, "meps": 0.1788718, "acs": 0.482032}


def run_test2(d: Dataset, k, num_clusters, threshold):
    selected_df = clustering(d, k=k, jaccard_threshold=threshold, num_clusters=num_clusters)
    return {"dataset": d.name, "k": k, "jaccard_threshold": threshold, "num_clusters": num_clusters, "score": sum(selected_df['score'])}


def save_graph2(df_results, param_name):
    plt.figure(figsize=(8, 6))
    df_results['naive_score'] = df_results["dataset"].map(NAIVE_RES)
    df_results["norm_score"] = df_results["score"] / df_results['naive_score'] * 100
    df_results.to_csv("outputs/parameters/n_clusters_results.csv", index=False)

    # Set Seaborn style
    sns.set(style="whitegrid", font_scale=1.2)
    sns.scatterplot(
        data=df_results,
        x=param_name,
        y='norm_score',
        hue='dataset',
        style='dataset',
        s=100 # Explicit marker
    )

    # Improve axis labels and font sizes
    plt.xlabel(param_name, fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Dataset', fontsize=14, title_fontsize=16)

    # Reduce number of x-ticks for readability
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))

    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_comparison.pdf')
    plt.close()


def plot_graph_for_k():
    df = pd.read_csv("outputs/parameters/k_results.csv")
    # Set Seaborn style
    sns.set(style="whitegrid", font_scale=1.2)

    # Define marker styles for threshold values (expand if you have more thresholds)
    marker_styles = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']
    thresholds = df['jaccard_threshold'].unique()
    marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(sorted(thresholds))}

    # Plot each dataset
    for dataset in df['dataset'].unique():
        plt.figure(figsize=(8, 6))
        subset = df[df['dataset'] == dataset]
        subset['norm_score'] = subset['norm_score'] * 100

        # Plot manually to control markers
        for threshold_value, group in subset.groupby('jaccard_threshold'):
            sns.scatterplot(
                data=group,
                x='k',
                y='norm_score',
                marker=marker_map[threshold_value],
                s=100,
                label=f"jaccard_threshold={threshold_value}"
            )

        plt.title(f"Score vs K for {dataset.upper()}")
        plt.xlabel("K")
        plt.ylabel("Score")
        plt.legend(title="Jaccard threshold")
        plt.tight_layout()
        plt.savefig(f"outputs/parameters/k_comparison_{dataset}.pdf")

l = []
# for t in [0,0.2,0.4,0.6,0.8,1]:
#     datasets = [[acs, 0], [so, 0], [meps, 0]]
#     for k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
#         tmp_datasets = datasets
#         datasets = []
#         for data, last_score in tmp_datasets:
#             # d: Dataset, k, num_clusters, threshold
#             r = run_test2(d=data, k=k, threshold=t, num_clusters=DEFAULT_NUM_CLUSTERS)
#             if r['score'] > last_score:
#                 datasets.append([data, r['score']])
#                 l.append(r)
# pd.DataFrame(l).to_csv("../outputs/parameters/k_results.csv", index=False)

# l = []
# for n_c in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
#     for data in [acs, so, meps]:
#         r = run_test2(d=data, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, num_clusters=n_c)
#         l.append(r)
# pd.DataFrame(l).to_csv("../outputs/parameters/n_clusters_results.csv", index=False)

# plot_graph_for_k()

l = pd.read_csv("outputs/parameters/n_clusters_results.csv")
save_graph2(l, "num_clusters")


