import pandas as pd
import numpy as np
from time import time

# import demo
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

THRESHOLD_SUPPORT = 0.01
K = 5
THRESHOLD = 0.25


def algorithm(D: Dataset, k=K, threshold_support=THRESHOLD_SUPPORT, threshold=THRESHOLD):
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(D.clean_path)
    max_outcome = max(df_clean[D.outcome_col])
    need_sample = len(df_clean) > 80000
    if need_sample:
        length = df_clean.loc[df_clean['group1']==1].shape[0]
        df_group1 = df_clean.loc[df_clean['group1']==1].sample(n=min(40000, length), random_state=42)
        df_group2 = df_clean.loc[df_clean['group2']==1].sample(n=40000, random_state=42)
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
    subgroups.sort_values(by="support", ascending=False, ignore_index=True).head(200).to_csv(f"outputs/{D.name}/interesting_subpopulations.csv", index=False)
    # step 2 - find the best treatment for each subpopulation
    df_treatments = find_best_treatment(D)
    if len(df_treatments) == 0:
        return None
    df_treatments.to_csv(f"outputs/{D.name}/subpopulations_and_treatments.csv", index=False)
    if need_sample:
        D.clean_path = original_clean_path
    logger.critical('Finished')
    # step 3 - find the best group with greedy algorithm
    calc_facts_metrics(D).to_csv(f"outputs/{D.name}/all_facts.csv", index=False)
    r = find_group(D, k, max_outcome, threshold)
    return r #, (e1-s), (e2-e1), (e3-e2), (e3-s)


from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
meps = Dataset(name="meps", outcome_col="FeltNervous",
               treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking',
                           'LongSinceLastFluVaccination', 'WearsSeatBelt', 'TakesAspirinFrequently'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Age', 'IsDiagnosedDiabetes',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
               columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
               dag_file="data/meps/causal_dag.txt")
so = Dataset(name="so", outcome_col="ConvertedSalary",
               treatments=['YearsCodingProf', 'Hobby', 'FormalEducation', 'WakeTime', 'HopeFiveYears', 'Dependents',
                           'HoursComputer', 'UndergradMajor', 'CompanySize'],
               subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                               'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                               'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                               'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country',
                               'SexualOrientation', 'EducationParents'],
               columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                  'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                  'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                  'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
             dag_file="data/so/causal_dag.txt")
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Temporary absence from work', 'Worked last week',
                          'Widowed in the past 12 months', "Total person earnings",
                          'Educational attainment', 'Gave birth within past year', "Field of degree - Science and Engineering flag"],
              subpopulations=['Sex', 'Age', 'With a disability', "Race/Ethnicity",
                              'Region', 'Language other than English spoken at home', 'state code',
                              'Marital status', 'Nativity', 'Related child'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_dag.txt")
import time
if __name__ == "__main__":
    start2 = time.time()
    # r=algorithm(D=meps)
    # e12 = time.time()
    # print(f"meps took {e12 - start2}")
    # r=algorithm(D=so)
    # e22 = time.time()
    # print(f"so took {e22 - e12}")
    # r = algorithm(D=acs)
    # e32 = time.time()
    # print(f"acs took {e32 - e22}")
    """
acs took 1725.8723595142365
meps took 121.82038521766663
so took 450.9267861843109
    """

