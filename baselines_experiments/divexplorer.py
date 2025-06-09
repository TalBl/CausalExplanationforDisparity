from algorithms.final_algorithm.new_greedy import get_intersection, ni_score
from algorithms.final_algorithm.find_treatment_new import findBestTreatment, get_subpopulation, getTreatmentATE, changeDAG, calc_dag
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
import pandas as pd
import numpy as np
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from Utils import Dataset, choose_lamda
import itertools
import ast
import pickle

THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5


def CalcIScore(treats: set, d: Dataset, dag, size, df_group1, df_group2, p_value):
    try:
        if size - 1 >= 1:
            lines = changeDAG(dag, [x[0] for x in treats])
            dag = calc_dag(lines)
        else:
            treatment_att_file_name = list(treats)[0][0]
            if treatment_att_file_name.startswith('DevType'):
                treatment_att_file_name = 'DevType'
            with open(f"data/{d.name}/causal_dags_files/{treatment_att_file_name}.pkl", 'rb') as file:
                dag = pickle.load(file)
        ate_group1 = getTreatmentATE(df_group1, dag, treats, d.outcome_col, p_value)
        ate_group2 = getTreatmentATE(df_group2, dag, treats, d.outcome_col, p_value)
        if ate_group1 and ate_group2: # pass p_value checks
            cate = abs(ate_group1 - ate_group2)
            return cate, ate_group1, ate_group2
        else:
            return None
    except Exception as e:
        print(e)


def stretch_power(x, alpha):
    return 1 - (1 - x)**alpha


def get_score(group, alpha, d, N, L):
    intersection = 0
    g = []
    iscore = 0
    for _, row in group.iterrows():
        g.append(row)
        if pd.notna(row['iscore']):
            iscore += row['ni_score'] * row['support']
    utility = stretch_power(iscore / L, alpha=6.69998) if iscore > 0 else 0
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    no_overlap = stretch_power(f_intersection, alpha=0.11484)
    score = (alpha * utility) + ((1 - alpha) * no_overlap)
    return {"ni_score_sum": iscore, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "no_overlap": no_overlap, "score": score}


def baseline(d: Dataset, K):
    df_clean = pd.read_csv(d.clean_path)
    s = pd.read_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv")
    L = s.shape[0]
    N = df_clean.shape[0]
    df = pd.read_csv(f'{d.name}_de_base.csv')
    df = df.sort_values(by=['ni_score', 'support'], ascending=(False, False)).head(K)
    scores = get_score(group=df, alpha=ALPHA, d=d, N=N, L=L)
    return scores


meps = Dataset(name="meps", outcome_col="FeltNervous",
               treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Age',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
               columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
               dag_file="data/meps/causal_dag.txt")
so = Dataset(name="so", outcome_col="ConvertedSalary",
             treatments=['YearsCodingProf', 'Hobby', 'FormalEducation', 'WakeTime', 'HopeFiveYears'],
             subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                             'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                             'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                             'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'],
             columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
             dag_file="data/so/causal_dag.txt")
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Temporary absence from work', 'Worked last week',
                          'Widowed in the past 12 months', "Total person earnings",
                          'Educational attainment', 'Georgraphic division'],
              subpopulations=['Sex', 'Age', 'With a disability', "Race/Ethnicity",
                              'Region', 'Language other than English spoken at home', 'state code',
                              'Marital status', 'Nativity', 'Related child'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_dag.txt")

l = []
for k in [1, 3, 5, 7, 9, 11, 13]:
    d = baseline(so, k)
    l.append({"baseline": "div-explorer", "dataset": "so", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(meps, k)
    l.append({"baseline": "div-explorer", "dataset": "meps", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(acs, k)
    l.append({"baseline": "div-explorer", "dataset": "acs", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
pd.DataFrame(l).to_csv("outputs/baselines_comparison/de_results.csv", index=False)

