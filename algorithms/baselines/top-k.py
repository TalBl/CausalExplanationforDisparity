import pandas as pd
from algorithms.final_algorithm.new_greedy import get_intersection, get_union, print_matrix
from Utils import Dataset, get_indices
import itertools
import numpy as np
import csv
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5
K = 5


def get_score(group, d):
    ni_score_sum = 0
    for _, row in group.iterrows():
        ni_score_sum += row['score']
    jaccard_matrix = print_matrix(d, {}, {}, [[x['subpop'], x['indices']] for _, x in pd.DataFrame(group).iterrows()])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/top_k_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    return {"score": ni_score_sum}


def baseline(d: Dataset):
    subs = pd.read_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv")
    df_facts_top_k = subs.sort_values(by='score', ascending=False).head(K)
    df_facts_top_k['indices'] = df_facts_top_k['subpop'].apply(get_indices, data=pd.read_csv(d.clean_path))
    scores = get_score(group=df_facts_top_k, d=d)
    df_facts_top_k.to_csv(f'outputs/{d.name}/baselines/facts_top_k.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/top_k_scores.csv')


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

from algorithms.final_algorithm.full import acs, so, meps
import time
start = time.time()
baseline(so)
e1 = time.time()
print(f"so took {e1-start}")
baseline(meps)
e2 = time.time()
print(f"meps took {e2-e1}")
baseline(acs)
e3 = time.time()
print(f"acs took {e3-e2}")

"""
so took 5.979233026504517
meps took 0.7690680027008057
acs took 95.76555919647217
"""