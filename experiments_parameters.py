import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm, acs, so, meps
from algorithms.final_algorithm.new_greedy import calc_facts_metrics, find_group
import random
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import math
from matplotlib import pyplot as plt
import warnings
import seaborn as sns


warnings.filterwarnings("ignore")

DEFAULT_K = 5
DEFAULT_SUPPORT = 0.01
DEFAULT_THRESHOLD = 0.25
df_results = []


# def run_test(d: Dataset, k, lamda, alpha, threshold_support):
#     r = algorithm(d, k, lamda, alpha, threshold_support)
#     return {"dataset": d.name, "k": k, "lamda": lamda, "alpha": alpha,
#             "threshold_support": threshold_support, "score": r}

def run_test2(d: Dataset, k, threshold_support, threshold):
    calc_facts_metrics(d).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    df_clean = pd.read_csv(d.clean_path)
    max_outcome = max(df_clean[d.outcome_col])
    r = find_group(d, k, max_outcome, threshold)
    return {"dataset": d.name, "k": k, "threshold": threshold, "threshold_support": threshold_support, "score": r}


def save_graph(df_results, param_name):
    plt.figure(figsize=(10, 6))
    # Plot a line for each unique basename
    subset = df_results[df_results['dataset'] == "so"]
    if 'log_score' in subset.keys():
        subset['uti'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle='None', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    if 'log_score' in subset.keys():
        subset['score'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle='None', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    if 'log_score' in subset.keys():
        subset['score'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle='None', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.ylabel('Log Utility', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', alpha=1)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_comparison.png')


def save_graph2(df_results, param_name):
    plt.figure(figsize=(10, 6))

    # Clearer markers using `marker='o'`
    sns.lineplot(
        data=df_results,
        x=param_name,
        y='score',
        hue='dataset',
        style='dataset',
        markers=True,
        dashes=True,
        linewidth=2.5,
        marker='o'  # Explicit marker
    )

    # Improve axis labels and font sizes
    plt.xlabel(param_name, fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Dataset', fontsize=14, title_fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Reduce number of x-ticks for readability
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))

    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_comparison.pdf')
    plt.close()

def save_graph3(df_results, param_name):
    plt.figure(figsize=(10, 6))

    # Use seaborn lineplot
    sns.scatterplot(
        data=df_results,
        x=param_name,
        y='utility',
        hue='dataset',
        style='dataset',
        s=200  # marker size
    )

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.xscale("log")
    plt.ylabel('Log Utility', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(title='Dataset', fontsize=20, title_fontsize=22)
    plt.grid(True, linestyle='--', alpha=1)

    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_log_utility_comparison.pdf')


# l = []
# for t in [0,0.2,0.4,0.6,0.8,1]:
#     for k in [1, 3, 5, 7, 9, 11, 13]:
#         l.append(run_test2(d=meps, k=k, threshold=t, threshold_support=DEFAULT_SUPPORT))
#         l.append(run_test2(d=so, k=k, threshold=t, threshold_support=DEFAULT_SUPPORT))
#         l.append(run_test2(d=acs, k=k, threshold=t, threshold_support=DEFAULT_SUPPORT))
# pd.DataFrame(l).to_csv("outputs/parameters/k_results.csv", index=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO
df = pd.read_csv("outputs/parameters/k_results.csv")

# Set Seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Define marker styles for threshold values (expand if you have more thresholds)
marker_styles = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']
thresholds = df['threshold'].unique()
marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(sorted(thresholds))}

# Plot each dataset
for dataset in df['dataset'].unique():
    plt.figure(figsize=(8, 6))
    subset = df[df['dataset'] == dataset]

    # Plot manually to control markers
    for threshold_value, group in subset.groupby('threshold'):
        sns.scatterplot(
            data=group,
            x='k',
            y='score',
            marker=marker_map[threshold_value],
            s=100,
            label=f"threshold={threshold_value}"
        )

    plt.title(f"Score vs K for {dataset.upper()}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.legend(title="Threshold")
    plt.tight_layout()
    plt.savefig(f"k_comparison_{dataset}.pdf")



# l = pd.read_csv("outputs/parameters/k_results.csv")
# save_graph2(l, "k")
# l = []
# for threshold in [0.1, 0.25, 0.35, 0.45, 0.5, 0.65, 0.75, 0.85, 1]:
#     l.append(run_test2(d=meps, k=DEFAULT_K, threshold=threshold, threshold_support=DEFAULT_SUPPORT))
#     l.append(run_test2(d=so, k=DEFAULT_K, threshold=threshold, threshold_support=DEFAULT_SUPPORT))
#     l.append(run_test2(d=acs, k=DEFAULT_K, threshold=threshold, threshold_support=DEFAULT_SUPPORT))
# pd.DataFrame(l).to_csv("outputs/parameters/threshold_results.csv")
# l = pd.read_csv("outputs/parameters/threshold_results.csv")
# save_graph2(l, "threshold")


# l = []
# for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
#     l.append(run_test2(d=meps, k=5, lamda=None, alpha=alpha, threshold_support=0.05))
#     l.append(run_test2(d=so, k=5, lamda=None, alpha=alpha, threshold_support=0.05))
#     l.append(run_test2(d=acs, k=5, lamda=None, alpha=alpha, threshold_support=0.05))
#
# pd.DataFrame(l).to_csv("outputs/parameters/alpha_results.csv", index=False)
# save_graph2(pd.read_csv("outputs/parameters/alpha_results.csv"), "alpha")

l = []
"""for threshold_support in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    l.append(run_test(d=meps, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
    l.append(run_test(d=so, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
    l.append(run_test(d=acs, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
save_graph(pd.DataFrame(l), "threshold_support")"""
