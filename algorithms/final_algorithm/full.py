import pandas as pd
import numpy as np
from time import time
from Utils import Dataset
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_treatment_new import find_best_treatment
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import warnings
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.ERROR,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')


CALCED_INTERSECTIONS = {}
CALCED_GRAPHS = {}

THRESHOLD_SUPPORT = 0.05
# LAMDA = 0.0001
ALPHA = 0.65
K = 5


def algorithm(D: Dataset, k=K, lamda=None, alpha=ALPHA, threshold_support=THRESHOLD_SUPPORT):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    s = time()
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(D.clean_path)
    need_sample = len(df_clean) > 80000
    if need_sample:
        df_group1 = df_clean.loc[df_clean['group1']==1].sample(n=50000, random_state=42)
        df_group2 = df_clean.loc[df_clean['group2']==1].sample(n=50000, random_state=42)
        df_sample = pd.concat([df_group1, df_group2])
        df_sample.to_csv(f"outputs/{D.name}/sample_data.csv", index=False)
        original_clean_path = D.clean_path
        D.clean_path = f"outputs/{D.name}/sample_data.csv"
        df_clean = df_sample
    logger.critical('Started')
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver\
        .get_pattern_divergence(min_support=threshold_support, quantitative_outcomes=[D.outcome_col],
                                group_1_column="group1", group_2_column="group2", attributes=D.subpopulations_atts,
                                COLUMNS_TO_IGNORE=D.columns_to_ignore)\
        .sort_values(by="support", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    if D.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: D.func_filter_subs(row[f'{D.outcome_col}_group1'], row[f'{D.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    subgroups.head(300).to_csv(f"outputs/{D.name}/interesting_subpopulations.csv", index=False)
    e1 = time()
    # step 2 - find the best treatment for each subpopulation
    df_treatments = find_best_treatment(D)
    if len(df_treatments) == 0:
        return None
    df_treatments.to_csv(f"outputs/{D.name}/subpopulations_and_treatments.csv", index=False)
    if need_sample:
        D.clean_path = original_clean_path
    e2 = time()
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(D, lamda).to_csv(f"outputs/{D.name}/all_facts.csv", index=False)
    r = find_group(D, k, alpha)
    e3 = time()
    return r['score'], (e1-s), (e2-e1), (e3-e2), (e3-s)


from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['BMI', 'Exercise', 'CurrentlySmoke'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Education',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking', 'DoesDoctorRecommendExercise'],
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
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
acs = Dataset(name="acs", outcome_col="Total person's earnings",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', 'Occupation recode', 'Worked last week',
                          'School enrollment', 'Insurance through a current or former employer or union', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)
import json

# r=algorithm(D=so)
# with open("res_so.json", "w") as f:
#     json.dump(r, f)
# r=algorithm(D=meps)
# with open("res_meps.json", "w") as f:
#     json.dump(r, f)
r = algorithm(D=acs)
# with open("res_acs.json", "w") as f:
#     json.dump(r, f)
# [0.6555158154401797, 62.27809143066406, 1689.460212469101, 1912.2186589241028, 3663.956962823868]