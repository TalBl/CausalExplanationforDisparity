from random import sample
import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm
from cleaning_datasets.clean_meps import build_mini_df as clean_meps_func
from cleaning_datasets.clean_so import build_mini_df as clean_so_func
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
import random
import math
from matplotlib import pyplot as plt
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


DEFAULT_K =5


df_results = []
from algorithms.final_algorithm.full import acs, so, meps, algorithm
# from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
# from cleaning_datasets.clean_so import filter_facts as so_filter_facts
# from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
#
# meps = Dataset(name="meps", outcome_col="FeltNervous",
#                treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking'],
#                subpopulations=['MaritalStatus', 'Region', 'Race', 'Age',
#                                'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
#                columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
#                func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
#                dag_file="data/meps/causal_dag.txt")
# so = Dataset(name="so", outcome_col="ConvertedSalary",
#              treatments=['YearsCodingProf', 'Hobby', 'FormalEducation', 'WakeTime', 'HopeFiveYears'],
#              subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
#                              'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
#                              'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
#                              'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'],
#              columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
#                                 'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
#                                 'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
#                                 'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
#              clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
#              dag_file="data/so/causal_dag.txt")
# acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
#               treatments=['Temporary absence from work', 'Worked last week',
#                           'Widowed in the past 12 months', "Total person earnings",
#                           'Educational attainment', 'Georgraphic division'],
#               subpopulations=['Sex', 'Age', 'With a disability', "Race/Ethnicity",
#                               'Region', 'Language other than English spoken at home', 'state code',
#                               'Marital status', 'Nativity', 'Related child'],
#               columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
#               func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_dag.txt")

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


l = []
for k in [3, 5, 7, 9, 11, 13, 15]:
    l.append(run_test(d=meps, k=k))
    l.append(run_test(d=so, k=k))
    l.append(run_test(d=acs, k=k))
pd.DataFrame(l).to_csv("outputs/scalability/k_run.csv", index=False)
l = pd.read_csv("outputs/scalability/k_run.csv")
save_graph(l, "K")

