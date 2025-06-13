from algorithms.final_algorithm.new_greedy import parse_subpopulation, print_matrix, get_union, get_intersection
from algorithms.final_algorithm.find_treatment_new import find_best_treatment
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
import pandas as pd
import numpy as np
import warnings
from Utils import Dataset, choose_lamda
import itertools
import ast
from tqdm import tqdm
import csv
import random
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


warnings.filterwarnings("ignore")

K = 5
alpha = 0.5

"""
1. for each subpopulation -> find his arg-max treatment
2. run all over subpopulations -> find arg-max group
"""

def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))

def get_score(group, alpha, d, n, calc_intersection, calc_union, L, max_outcome):
    g = []
    iscore = 0
    for _, row in group:
        g.append(row)
        iscore += row['iscore'] / max_outcome
    for row1, row2 in itertools.combinations(g, 2):
        intersection = get_intersection(row1, row2, d, calc_intersection)
        union = get_union(row1, row2, d, calc_union)
        jaccard = intersection / union
        if jaccard > 0.25:
            return {"score": 0}
    return {"score": iscore}

def parse_subpoplation(sub_str, d: Dataset):
    sub_str = ast.literal_eval(sub_str)
    df_original = pd.read_csv(d.clean_path)
    df = df_original.copy()
    for s in sub_str:
        k, v = s.split("=")
        try:
            v = float(v)
        except:
            v = v
        df = df.loc[df[k]==v]
    belong_groups = df.loc[(df['group1'] == 1) | (df['group2'] == 1)]
    support = belong_groups.shape[0] / df_original.shape[0]
    return support


def run_search(d, k, df_treatments, calc_intersection, calc_union, L, max_outcome, n):
    max_score = 0
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), k)):
        # L, group, attribute, alpha, d, CALCED_INTERSECTION, N
        scores = get_score(group=group, alpha=alpha, d=d, n=n, calc_intersection=calc_intersection, calc_union=calc_union, L=L, max_outcome=max_outcome)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    if max_score > 0:
        return max_score, res_group, scores_dict
    else:
        return 0, [], {}

def baseline(d: Dataset):
    calc_intersection = {}
    calc_union = {}
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    df_treatments = find_best_treatment(d)
    df_treatments.to_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv", index=False)
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("sample", "clean")
    df_treatments = pd.read_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv")
    L = df_treatments.shape[0]
    max_score = 0
    res_group = pd.DataFrame()
    scores_dict = {}
    df_treatments['support'] = df_treatments['subpopulation'].apply(lambda x: parse_subpoplation(x, d))
    df = pd.read_csv(d.clean_path)
    n = df.shape[0]
    max_outcome = max(df[d.outcome_col])
    res_group = []
    k = K
    while res_group == []:
        max_score, res_group, scores_dict = run_search(d, k, df_treatments, calc_intersection, calc_union, L, max_outcome, n)
        k -= 1
    g = []
    for x in res_group:
        _, row = x
        g.append(row)
    jaccard_matrix = print_matrix(d, calc_intersection, calc_union, [x['subpopulation'] for x in g])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/naive_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    pd.DataFrame(g).to_csv(f'outputs/{d.name}/baselines/facts_naive.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/naive_scores.csv')

from algorithms.final_algorithm.full import acs, so, meps
# from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
# from cleaning_datasets.clean_so import filter_facts as so_filter_facts
# from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
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
#                           'Educational attainment'],
#               subpopulations=['Sex', 'Age', 'With a disability', "Race/Ethnicity",
#                               'Region', 'Language other than English spoken at home', 'state code',
#                               'Marital status', 'Nativity', 'Related child'],
#               columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
#               func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_dag.txt")
import time
start = time.time()
# baseline(so)
e1 = time.time()
# print(f"so took {e1-start} seconds")
baseline(meps)
e2 = time.time()
print(f"meps took {e2-e1} seconds")
# baseline(acs)
# e3 = time.time()
# print(f"acs took {e3-e2} seconds")

"""
acs took 10303.978988409042 seconds
meps took 155.4656527042389 seconds
so took 840.0273792743683 seconds

"""
