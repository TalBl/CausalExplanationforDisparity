import time
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import warnings
import logging
from time import time
from algorithms.final_algorithm.find_best_treatment import find_best_treatment as find_with_all
from algorithms.experiments.find_treatment_without_parallel import find_best_treatment as find_best_treatment_only_cache
from algorithms.experiments.find_treatment_without_cache import find_best_treatment as find_best_treatment_only_parallel
from algorithms.experiments.find_treatment_without_none import find_best_treatment as find_best_treatment_none
from algorithms.final_algorithm.full import acs, so, meps, algorithm
from Utils import Dataset

df_results = []


def run_test(d: Dataset, treatments_func, method_name):
    time_start = time()
    algorithm(d, treatments_func=treatments_func)
    time_end = time()
    return {"dataset": d.name, "runtime": round(time_end - time_start, 2), "method": method_name}


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_graph1(df):
    """
    Plot runtime comparisons using Seaborn from a DataFrame, with per-method hatches and color-hatch legend.
    Parameters:
        df (pd.DataFrame): Must contain columns: 'dataset', 'runtime', 'method'
    """
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Create barplot
    ax = sns.barplot(data=df, x="dataset", y="runtime", hue="method", edgecolor="black")

    # Map methods to hatches
    methods = df["method"].unique()
    hatch_patterns = ['/', '\\', 'x', '-', '|', '+', 'o', '.']
    method_to_hatch = {method: hatch_patterns[i % len(hatch_patterns)] for i, method in enumerate(methods)}

    # Extract bar colors
    # Seaborn plots in order: dataset1_method1, dataset2_method1, ..., dataset1_method2, ...
    num_datasets = df["dataset"].nunique()
    method_colors = {}
    for i, method in enumerate(methods):
        patch_index = i * num_datasets  # first bar of each method group
        patch = ax.patches[patch_index]
        method_colors[method] = patch.get_facecolor()

    # Apply hatches to bars
    for patch, method in zip(ax.patches, df["method"]):
        patch.set_hatch(method_to_hatch[method])

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10)

    # Create color+hatch legend
    legend_handles = []
    for method in methods:
        patch = mpatches.Patch(
            facecolor=method_colors[method],
            edgecolor='black',
            hatch=method_to_hatch[method],
            label=method
        )
        legend_handles.append(patch)

    ax.set_ylabel("Runtime (seconds)", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, df["runtime"].max() * 1.1)

    plt.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=12, title="")
    plt.tight_layout()
    plt.savefig("outputs/optimizations/optimizations_comparison.pdf")


if __name__ == '__main__':
    # l = []
    # for method_name, method_func in [['No Optimizations', find_best_treatment_none], ['OnlyCache', find_best_treatment_only_cache], ['OnlyParallel', find_best_treatment_only_parallel], ['All', find_with_all]]:
    #     print(f"RUN {method_name}")
    #     for d in [so, meps, acs]:
    #         l.append(run_test(d, method_func, method_name))
    #         print(f"FINISH DATASET {d.name}")
    # pd.DataFrame(l).to_csv("outputs/optimizations/results.csv", index=False)
    save_graph1(pd.read_csv("../outputs/optimizations/results.csv"))
