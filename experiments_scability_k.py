from random import sample
import time
import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.full import algorithm
from Cleaning_Datasets.clean_meps import build_mini_df as clean_meps_func
from Cleaning_Datasets.clean_so import build_mini_df as clean_so_func
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
import random
import math
from matplotlib import pyplot as plt
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


DEFAULT_K =5


df_results = []
meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['IsHadStroke', 'DoesDoctorRecommendExercise', 'TakesAspirinFrequently',
                           'WearsSeatBelt', 'Exercise', 'LongSinceLastFluVaccination', 'CurrentlySmoke'],
               subpopulations=['Married', 'Region', 'Race',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],
               columns_to_ignore=[], func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
               func_filter_treats=acs_filter_treats, clean_path="outputs/meps/clean_data.csv")

so = Dataset(name="so", outcome_col="ConvertedCompYearly",
             treatments=['YearsCodingProf', 'JobSatisfaction', 'Hobby', 'LastNewJob', 'Exercise', 'Student',
                         'WakeTime', 'DevType'],
             subpopulations=['Gender', 'FormalEducation','Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                             'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                             'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                             'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'],
             columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'], func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
             func_filter_treats=acs_filter_treats, clean_path="outputs/so/clean_data.csv")

acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

def run_test(d: Dataset, k):
    start_time = time.time()
    calc_facts_metrics(d)
    find_group(d, k, 0.65)
    end_time = time.time()
    runtime = end_time-start_time
    return {"dataset": d.name,"K": k, "runtime": round(runtime, 2)}


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


l = []
for k in [3, 5, 8, 10, 12, 14]:
    l.append(run_test(d=meps, k=k))
    l.append(run_test(d=so, k=k))
    l.append(run_test(d=acs, k=k))
save_graph(pd.DataFrame(l), "K")

