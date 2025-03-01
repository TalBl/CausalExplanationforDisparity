import pandas as pd
from algorithms.final_algorithm.new_greedy import get_intersection
from Utils import Dataset
import itertools
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


THRESHOLD_SUPPORT = 0.01
ALPHA = 0.65
K = 5

def get_score(group, alpha, d, N, L):
    intersection = 0
    ni_score_sum = 0
    for _, row in group.iterrows():
        ni_score_sum += row['ni_score'] * row['support']
    utility = ni_score_sum
    for pair in itertools.combinations(group.iterrows(), 2):
        _, row1 = pair[0]
        _, row2 = pair[1]
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def baseline(d: Dataset):
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    subs = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    L = subs.shape[0]
    df_clean = pd.read_csv(d.clean_path)
    N = df_clean.shape[0]
    df_facts_top_k = df_facts.sort_values(by='ni_score', ascending=False).head(K)
    scores = get_score(group=df_facts_top_k, alpha=ALPHA, d=d, L=L, N=N)
    df_facts_top_k.to_csv(f'outputs/{d.name}/baselines/facts_top_k.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/top_k_scores.csv')


meps = Dataset(name="meps", outcome_col="FeltNervous",
               treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Age',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
               columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
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
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Temporary absence from work', 'Worked last week', "person weight",
                          'Widowed in the past 12 months', "Total person's earnings",
                          'Educational attainment', 'Georgraphic division'],
              subpopulations=['Sex', 'Age', 'With a disability', "Race/Ethnicity",
                              'Region', 'Language other than English spoken at home', 'state code',
                              'Marital status', 'Nativity', 'Related child'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)


baseline(so)
baseline(meps)
baseline(acs)
