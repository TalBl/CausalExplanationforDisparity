import pandas as pd
from algorithms.final_algorithm.new_greedy import get_intersection
from Utils import Dataset
import itertools
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


K = 5
alpha = 0.65

def get_score(group, alpha, d, max_subpopulation):
    intersection = 0
    ni_score_sum = sum(group['ni_score'])
    utility = ni_score_sum / len(group)
    for pair in itertools.combinations(group.iterrows(), 2):
        _, row1 = pair[0]
        _, row2 = pair[1]
        intersection += get_intersection(row1, row2, d)
    f_intersection = ((max_subpopulation*len(group)*len(group)) - intersection) / (max_subpopulation*len(group)*len(group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def baseline(d: Dataset):
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    df_facts_top_k = df_facts.sort_values(by='ni_score', ascending=False).head(K)
    max_subpopulation = max(df_facts['size_subpopulation'])
    scores = get_score(group=df_facts_top_k, alpha=alpha, d=d, max_subpopulation=max_subpopulation)
    df_facts_top_k.to_csv(f'outputs/{d.name}/baselines/facts_top_k.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/top_k_scores.csv')


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
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True)

meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['DoesDoctorRecommendExercise', 'TakesAspirinFrequently', 'BMI', 'Exercise',
                           'CurrentlySmoke'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Education',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],
               columns_to_ignore=['Education=UnAcceptable', 'IsWorking=UnAcceptable'], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True)

acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=False, need_filter_treatments=False,
              func_filter_treats=acs_filter_treats, can_ignore_treatments_filter=True)

#baseline(so)
# baseline(meps)
baseline(acs)
