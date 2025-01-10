import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm
from algorithms.final_algorithm.new_greedy import calc_facts_metrics, find_group
from Cleaning_Datasets.clean_meps import build_mini_df as clean_meps_func
from Cleaning_Datasets.clean_so import build_mini_df as clean_so_func
import random
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import math
from matplotlib import pyplot as plt
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
import warnings

warnings.filterwarnings("ignore")

DEFAULT_K =5
DEFAULT_LAMDA = 0.001
DEFAULT_ALPHA = 0.65

df_results = []

meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['DoesDoctorRecommendExercise', 'TakesAspirinFrequently', 'BMI', 'Exercise',
                           'CurrentlySmoke'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Education',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],
               columns_to_ignore=['Education=UnAcceptable', 'IsWorking=UnAcceptable'], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
so = Dataset(name="so", outcome_col="ConvertedCompYearly",
             treatments=['YearsCodingProf', 'Hobby', 'LastNewJob', 'Student', 'WakeTime', 'DevType'],
             subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                             'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                             'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                             'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'],
             columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
             clean_path="outputs/so/clean_data.csv", func_filter_treats=so_filter_facts, func_filter_subs=so_filter_facts, need_filter_subpopulations=True,
             need_filter_treatments=True)

acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

def run_test(d: Dataset, k, lamda, alpha, threshold_support):
    r = algorithm(d, k, lamda, alpha, threshold_support)
    return {"dataset": d.name, "k": k, "lamda": lamda, "alpha": alpha,
            "threshold_support": threshold_support, "score": r}

def run_test2(d: Dataset, k, lamda, alpha, threshold_support):
    calc_facts_metrics(d, lamda).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    r = find_group(d, k, alpha)["score"]
    return {"dataset": d.name, "k": k, "lamda": lamda, "alpha": alpha,
            "threshold_support": threshold_support, "score": r}


def save_graph(df_results, param_name):
    plt.figure(figsize=(10, 6))
    # Plot a line for each unique basename
    subset = df_results[df_results['dataset'] == "so"]
    if 'log_score' in subset.keys():
        subset['score'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle='--', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    if 'log_score' in subset.keys():
        subset['score'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle=':', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    if 'log_score' in subset.keys():
        subset['score'] = subset['log_score']
    plt.plot(subset[param_name], subset['score'], linestyle='-.', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.ylabel('Score', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', alpha=1)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_comparison.png')


def save_graph2(df_results, param_name):
    # Create a Figure and Axes
    fg = Figure(figsize=(10, 6))
    ax = fg.gca()

    # Plot a line for each unique basename
    subset = df_results[df_results['dataset'] == "so"]
    ax.plot(subset[param_name], subset['score'], linestyle='--', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    ax.plot(subset[param_name], subset['score'], linestyle=':', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    ax.plot(subset[param_name], subset['score'], linestyle='-.', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Customize the Axes
    ax.set_xlabel(param_name, fontsize=24, fontweight='bold')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=24)
    ax.grid(True, linestyle='--', alpha=1)

    # Ensure x-axis ticks are integers if applicable
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the figure
    fg.savefig(f'outputs/parameters/{param_name}_comparison.png')

def save_graph3(df_results, param_name):
    plt.figure(figsize=(10, 6))
    # Plot a line for each unique basename
    subset = df_results[df_results['dataset'] == "so"]
    plt.plot(subset[param_name], subset['score'], linestyle='--', marker='o', color='b', label="SO", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "meps"]
    plt.plot(subset[param_name], subset['score'], linestyle=':', marker='s', color='r', label="MEPS", linewidth=6, markersize=14)

    subset = df_results[df_results['dataset'] == "acs"]
    plt.plot(subset[param_name], subset['score'], linestyle='-.', marker='^', color='g', label="ACS", linewidth=6, markersize=14)

    # Add title, labels, and legend
    plt.xlabel(param_name, fontsize=24, fontweight='bold')
    plt.xscale("log")
    plt.ylabel('Score', fontsize=24, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle='--', alpha=1)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'outputs/parameters/{param_name}_comparison.png')


l = []
for k in [1, 3, 5, 8, 10, 12, 14]:
    l.append(run_test2(d=meps, k=k, lamda=None, alpha=0.65, threshold_support=0.05))
    l.append(run_test2(d=so, k=k, lamda=None, alpha=0.65, threshold_support=0.05))
    l.append(run_test2(d=acs, k=k, lamda=None, alpha=0.65, threshold_support=0.05))
pd.DataFrame(l).to_csv("outputs/parameters/k_results.csv")
save_graph2(pd.DataFrame(l), "k")
l = []
for lamda in [0.00000005, 0.0000005, 0.000005, 0.00005, 0.0005, 0.005, 0.05, 0.5, 1]:
    l.append(run_test2(d=meps, k=5, lamda=lamda, alpha=0.65, threshold_support=0.05))
    l.append(run_test2(d=so, k=5, lamda=lamda, alpha=0.65, threshold_support=0.05))
    l.append(run_test2(d=acs, k=5, lamda=lamda, alpha=0.65, threshold_support=0.05))
pd.DataFrame(l).to_csv("outputs/parameters/lamda_results.csv")
save_graph3(pd.DataFrame(l), "lamda")


"""l = []
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.85, 1]:
    l.append(run_test2(d=meps, k=5, lamda=None, alpha=alpha, threshold_support=0.05))
    l.append(run_test2(d=so, k=5, lamda=None, alpha=alpha, threshold_support=0.05))
    l.append(run_test2(d=acs, k=5, lamda=None, alpha=alpha, threshold_support=0.05))

save_graph(pd.DataFrame(l), "alpha")"""

l = []
"""for threshold_support in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    l.append(run_test(d=meps, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
    l.append(run_test(d=so, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
    l.append(run_test(d=acs, k=5, lamda=None, alpha=0.65, threshold_support=threshold_support))
save_graph(pd.DataFrame(l), "threshold_support")"""
