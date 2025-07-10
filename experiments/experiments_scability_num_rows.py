from random import sample
import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm
from cleaning_datasets.clean_meps import build_mini_df as clean_meps_func
from cleaning_datasets.clean_so import build_mini_df as clean_so_func
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
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


def run_test(d: Dataset, count_rows):
    frac, str_count = count_rows
    dd = d.copy()
    df_clean = pd.read_csv(dd.clean_path)
    df_sample = df_clean.sample(frac=frac, random_state=42)
    df_sample.to_csv(f"outputs/{dd.name}/clean_data_{str_count}.csv", index=False)
    dd.clean_path = f"outputs/{dd.name}/clean_data_{str_count}.csv"
    runtime_list = []
    for i in range(3):
        print(f"Running test {i}")
        start_time = time.time()
        algorithm(D=dd)
        end_time = time.time()
        runtime = end_time-start_time
        print("Finished running test {i}")
        runtime_list.append(runtime)
    runtime_avg = sum(runtime_list)/len(runtime_list)
    return {"dataset": d.name,"COUNT ROWS": frac, "runtime": round(runtime_avg, 2)}

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
    # l = []
    # for count_rows in [(0.1, "10"), (0.2, "20"),(0.3, "30"), (0.4, "40"),(0.5, "50"),(0.6, "60"), (0.7, "70"),(0.8, "80"),(0.9, "90"), (1, "100")]:
    #     l.append(run_test(d=meps, count_rows=count_rows))
    #     l.append(run_test(d=so, count_rows=count_rows))
    #     l.append(run_test(d=acs, count_rows=count_rows))
    #     print(f"finishing {count_rows}")
    # pd.DataFrame(l).to_csv("outputs/scalability/count_rows_comparison.csv", index=False)
    save_graph(pd.read_csv("../outputs/scalability/count_rows_comparison.csv"), "COUNT ROWS")
