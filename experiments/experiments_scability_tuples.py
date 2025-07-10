from random import sample
import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm
import random
import math
from matplotlib import pyplot as plt
import random
from algorithms.final_algorithm.full import so, acs, meps


DEFAULT_K = 5
DEFAULT_SUPPORT = 0.01
DEFAULT_THRESHOLD = 0.55
DEFAULT_NUM_CLUSTERS = 2*DEFAULT_K

df_results = []

def run_test(d: Dataset, count_atts):
    dd = d.copy()
    if len(dd.subpopulations_atts) > count_atts:
        subs = random.sample(dd.subpopulations_atts, count_atts)
        dd.subpopulations_atts = subs
        dd.columns_to_ignore = []
        for c in d.columns_to_ignore:
            col_name = c.split("=0")[0]
            if col_name in subs:
                dd.columns_to_ignore.append(f"{col_name}=0")
    if len(dd.treatments_atts) > count_atts:
        treats = random.sample(dd.treatments_atts, count_atts)
        dd.treatments_atts = treats
    start_time = time.time()
    algorithm(D=dd)
    end_time = time.time()
    runtime = end_time-start_time
    return {"dataset": d.name, "NUMBER OF ATTS": count_atts, "runtime": round(runtime, 2)}

def save_graph(df_results, param_name):
    plt.figure(figsize=(10, 6))
    subset = df_results[df_results['dataset'] == "so"]
    plt.plot(subset[param_name], subset['runtime'], linestyle='--', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    plt.plot(subset[param_name], subset['runtime'], linestyle=':', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    plt.plot(subset[param_name], subset['runtime'], linestyle='-.', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.ylabel('Run Time (Seconds)', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', alpha=1)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'outputs/scalability/{param_name}_comparison.png')


if __name__ == "__main__":
    l = []
    for count_atts in [2, 4, 6, 8, 10]:
        l.append(run_test(d=meps, count_atts=count_atts))
        l.append(run_test(d=so, count_atts=count_atts))
        l.append(run_test(d=acs, count_atts=count_atts))
    pd.DataFrame(l).to_csv("../outputs/scalability/tuples_run.csv")
    save_graph(pd.DataFrame(l), "NUMBER OF ATTS")
