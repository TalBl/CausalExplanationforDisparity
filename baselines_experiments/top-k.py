import pandas as pd
from algorithms.final_algorithm.new_greedy import get_intersection
from Utils import Dataset, ni_score, choose_lamda
import itertools
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5

def stretch_power(x, alpha):
    return 1 - (1 - x)**alpha

def get_score(group, alpha, d, N, L):
    intersection = 0
    ni_score_sum = 0
    for _, row in group.iterrows():
        ni_score_sum += row['nd_score'] * row['support']
    utility = stretch_power(ni_score_sum / L, alpha=6.69998)
    for pair in itertools.combinations(group.iterrows(), 2):
        _, row1 = pair[0]
        _, row2 = pair[1]
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    no_overlap = stretch_power(f_intersection, alpha=0.11484)
    score = (alpha * utility) + ((1 - alpha) * no_overlap)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "no_overlap": no_overlap, "score": score}


def baseline(d: Dataset, K):
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    subs = pd.read_csv(f"outputs/{d.name}/subpopulations_and_treatments.csv")
    L = subs.shape[0]
    df_clean = pd.read_csv(d.clean_path)
    N = df_clean.shape[0]
    df_facts['utility'] = df_facts['nd_score'] * df_facts['support']
    df_facts_top_k = df_facts.sort_values(by='utility', ascending=False).head(K)
    scores = get_score(group=df_facts_top_k, alpha=ALPHA, d=d, L=L, N=N)
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
    l.append({"baseline": "top-k", "dataset": "so", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(meps, k)
    l.append({"baseline": "top-k", "dataset": "meps", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(acs, k)
    l.append({"baseline": "top-k", "dataset": "acs", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
pd.DataFrame(l).to_csv("outputs/baselines_comparison/top_k_results.csv", index=False)
