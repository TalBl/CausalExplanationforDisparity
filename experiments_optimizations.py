import time
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
import warnings
import logging
from time import time
from Utils import Dataset
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_treatment_new import find_best_treatment as find_with_all
from algorithms.experiments.find_treatment_without_parallel import find_best_treatment as find_without_parallel
from algorithms.experiments.find_treatment_without_cache import find_best_treatment as find_without_cache
from algorithms.experiments.find_treatment_without_none import find_best_treatment as find_none
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.ERROR,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

DEFAULT_K =5
DEFAULT_ALPHA = 0.65
THRESHOLD_SUPPORT = 0.05

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
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

def run_test(d: Dataset):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    s = time()
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(d.clean_path)
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver \
        .get_pattern_divergence(min_support=THRESHOLD_SUPPORT, quantitative_outcomes=[d.outcome_col],
                                group_1_column="group1", group_2_column="group2", attributes=d.subpopulations_atts,
                                COLUMNS_TO_IGNORE=d.columns_to_ignore) \
        .sort_values(by="support", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    subgroups.head(20).to_csv(f"outputs/{d.name}/interesting_subpopulations.csv", index=False)
    e1 = time()
    # step 2 - find the best treatment for each subpopulation
    logger.critical('Started')
    df_treatments = find_without_parallel(d)
    if len(df_treatments) == 0:
        e2 = time()
        return {"dataset": d.name, "optimization": "WITHOUT PARALLEL", "step1": e1-s, "step2": e2-e1,
                "step3": 0, "all": e2-s}
    df_treatments.to_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv", index=False)
    e2 = time()
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(d).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    find_group(d, DEFAULT_K, DEFAULT_ALPHA)
    e3 = time()
    return {"dataset": d.name, "optimization": "WITHOUT PARALLEL", "step1": e1-s, "step2": e2-e1,
            "step3": e3-e2, "all": e3-s}

def run_test2(d: Dataset):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    s = time()
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(d.clean_path)
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver \
        .get_pattern_divergence(min_support=THRESHOLD_SUPPORT, quantitative_outcomes=[d.outcome_col],
                                group_1_column="group1", group_2_column="group2", attributes=d.subpopulations_atts,
                                COLUMNS_TO_IGNORE=d.columns_to_ignore) \
        .sort_values(by="support", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    subgroups.head(20).to_csv(f"outputs/{d.name}/interesting_subpopulations.csv", index=False)
    e1 = time()
    # step 2 - find the best treatment for each subpopulation
    logger.critical('Started')
    df_treatments = find_without_cache(d)
    if len(df_treatments) == 0:
        e2 = time()
        return {"dataset": d.name, "optimization": "WITHOUT CACHE", "step1": e1-s, "step2": e2-e1,
                "step3": 0, "all": e2-s}
    df_treatments.to_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv", index=False)
    e2 = time()
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(d).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    find_group(d, DEFAULT_K, DEFAULT_ALPHA)
    e3 = time()
    return {"dataset": d.name, "optimization": "WITHOUT CACHE", "step1": e1-s, "step2": e2-e1,
            "step3": e3-e2, "all": e3-s}

def run_test3(d: Dataset):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    s = time()
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(d.clean_path)
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver \
        .get_pattern_divergence(min_support=THRESHOLD_SUPPORT, quantitative_outcomes=[d.outcome_col],
                                group_1_column="group1", group_2_column="group2", attributes=d.subpopulations_atts,
                                COLUMNS_TO_IGNORE=d.columns_to_ignore) \
        .sort_values(by="support", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    subgroups.head(20).to_csv(f"outputs/{d.name}/interesting_subpopulations.csv", index=False)
    e1 = time()
    # step 2 - find the best treatment for each subpopulation
    logger.critical('Started')
    df_treatments = find_with_all(d)
    if len(df_treatments) == 0:
        e2 = time()
        return {"dataset": d.name, "optimization": "ALL", "step1": e1-s, "step2": e2-e1,
                "step3": 0, "all": e2-s}
    df_treatments.to_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv", index=False)
    e2 = time()
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(d).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    find_group(d, DEFAULT_K, DEFAULT_ALPHA)
    e3 = time()
    return {"dataset": d.name, "optimization": "ALL", "step1": e1-s, "step2": e2-e1,
            "step3": e3-e2, "all": e3-s}


def run_test4(d: Dataset):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    s = time()
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(d.clean_path)
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver \
        .get_pattern_divergence(min_support=THRESHOLD_SUPPORT, quantitative_outcomes=[d.outcome_col],
                                group_1_column="group1", group_2_column="group2", attributes=d.subpopulations_atts,
                                COLUMNS_TO_IGNORE=d.columns_to_ignore) \
        .sort_values(by="support", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    subgroups.head(20).to_csv(f"outputs/{d.name}/interesting_subpopulations.csv", index=False)
    e1 = time()
    # step 2 - find the best treatment for each subpopulation
    logger.critical('Started')
    df_treatments = find_none(d)
    if len(df_treatments) == 0:
        e2 = time()
        return {"dataset": d.name, "optimization": "NONE", "step1": e1-s, "step2": e2-e1,
                "step3": 0, "all": e2-s}
    df_treatments.to_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv", index=False)
    e2 = time()
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(d).to_csv(f"outputs/{d.name}/all_facts.csv", index=False)
    find_group(d, DEFAULT_K, DEFAULT_ALPHA)
    e3 = time()
    return {"dataset": d.name, "optimization": "NONE", "step1": e1-s, "step2": e2-e1,
            "step3": e3-e2, "all": e3-s}

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

"""res_1 = []
res_1.append(run_test(so))
res_1.append(run_test(meps))
res_1.append(run_test(acs))
pd.DataFrame(res_1).to_csv("outputs/optimization/without_parallel.csv", index=False)

res_2 = []
res_2.append(run_test2(so))
res_2.append(run_test2(meps))
res_2.append(run_test2(acs))
pd.DataFrame(res_2).to_csv("outputs/optimization/without_cache.csv", index=False)

res_3 = []
res_3.append(run_test3(so))
res_3.append(run_test3(meps))
res_3.append(run_test3(acs))
pd.DataFrame(res_3).to_csv("outputs/optimization/all.csv", index=False)


res_4 = []
res_4.append(run_test4(so))
res_4.append(run_test4(meps))
res_4.append(run_test4(acs))
pd.DataFrame(res_4).to_csv("outputs/optimization/none.csv", index=False)"""

dataset = ("SO", "MEPS", "ACS")
optimizations = {
    'None': (165, 57, 931),
    'WithoutCache': (185, 65, 910),
    'WithoutParallel': (159, 45, 782),
    'All': (156, 44, 765),
}

x = np.arange(len(dataset))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in optimizations.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Runtime (seconds)')
ax.set_xticks(x + width, dataset)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1000)

plt.show()
