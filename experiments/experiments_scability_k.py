import time
import pandas as pd
from Utils import Dataset
from matplotlib import pyplot as plt



df_results = []
from algorithms.final_algorithm.full import acs, so, meps, algorithm


def run_test(d: Dataset, k):
    start_time = time.time()
    algorithm(D=d, k=k)
    end_time = time.time()
    runtime = end_time-start_time
    return {"dataset": d.name, "K": k, "runtime": round(runtime, 2)}


def save_graph(df_results, param_name):
    plt.figure(figsize=(10, 6))
    subset = df_results[df_results['dataset'] == "so"]
    plt.plot(subset[param_name], subset['runtime'], linestyle='None', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    plt.plot(subset[param_name], subset['runtime'], linestyle='None', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    plt.plot(subset[param_name], subset['runtime'], linestyle='None', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.ylabel('Run Time (Seconds)', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', linewidth=1, alpha=1)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'outputs/scalability/{param_name}_comparison.pdf')

if __name__ == "__main__":
    l = []
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        l.append(run_test(d=meps, k=k))
        l.append(run_test(d=so, k=k))
        l.append(run_test(d=acs, k=k))
    pd.DataFrame(l).to_csv("../outputs/scalability/k_run.csv", index=False)
    l = pd.read_csv("../outputs/scalability/k_run.csv")
    save_graph(l, "K")

