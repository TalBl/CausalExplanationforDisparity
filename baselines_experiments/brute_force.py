from algorithms.final_algorithm.new_greedy import parse_subpopulation, get_intersection
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

def stretch_power(x, alpha):
    return 1 - (1 - x)**alpha

def get_score(group, alpha, d, n, calc_intersection, L):
    intersection = 0
    g = []
    iscore = 0
    for _, row in group:
        g.append(row)
        iscore += row['support'] * row['nd_score']
    utility = stretch_power(iscore / L, alpha=6.69998)
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d, calc_intersection)
    f_intersection = ((n*L*L) - intersection) / (n*L*L)
    no_overlap = stretch_power(f_intersection, alpha=0.11484)
    score = (alpha * utility) + ((1 - alpha) * no_overlap)
    return {"ni_score_sum": iscore, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "no_overlap": no_overlap, "score": score}

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


def baseline(d: Dataset):
    calc_intersection = {}
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
    lamda = choose_lamda(df_treatments['iscore'])
    df_treatments['nd_score'] = df_treatments['iscore'].apply(lambda x: ni_score(x, lamda))
    df_treatments['support'] = df_treatments['subpopulation'].apply(lambda x: parse_subpoplation(x, d))
    df = pd.read_csv(d.clean_path)
    n = df.shape[0]
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), K)):
        # L, group, attribute, alpha, d, CALCED_INTERSECTION, N
        scores = get_score(group=group, alpha=alpha, d=d, n=n, calc_intersection=calc_intersection, L=L)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    g = []
    for x in res_group:
        _, row = x
        g.append(row)
    pd.DataFrame(g).to_csv(f'outputs/{d.name}/baselines/facts_naive.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/naive_scores.csv')


from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
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
from algorithms.final_algorithm.full import acs, meps, so

baseline(so)
baseline(meps)
baseline(acs)
