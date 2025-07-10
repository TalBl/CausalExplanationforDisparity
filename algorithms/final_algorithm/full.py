import csv

import pandas as pd
import numpy as np
from time import time

# import demo
from Utils import Dataset
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.clustering2 import clustering
from algorithms.final_algorithm.find_best_treatment import find_best_treatment
import warnings
import logging

from algorithms.final_algorithm.new_greedy import print_matrix

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.ERROR,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


CALCED_INTERSECTIONS = {}
CALCED_GRAPHS = {}

THRESHOLD_SUPPORT = 0.05
K = 5
THRESHOLD = 0.55


def algorithm(D: Dataset, k=K, threshold_support=THRESHOLD_SUPPORT, jaccard_threshold=THRESHOLD, num_clusters=2*K, treatments_func=None):
    s1 = time()
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(D.clean_path)
    max_outcome = max(df_clean[D.outcome_col])
    need_sample = len(df_clean) > 50000
    if need_sample:
        length = df_clean.loc[df_clean['group1']==1].shape[0]
        df_group1 = df_clean.loc[df_clean['group1']==1].sample(n=min(25000, length), random_state=42)
        df_group2 = df_clean.loc[df_clean['group2']==1].sample(n=25000, random_state=42)
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
    subgroups[f'{D.outcome_col}_div'] = abs(subgroups[f'{D.outcome_col}_div'])
    subgroups_filtered = subgroups[subgroups['support'] > threshold_support]
    subgroups_filtered.sort_values(by=f'support', ascending=False, ignore_index=True).to_csv(f"outputs/{D.name}/interesting_subpopulations.csv", index=False)
    s2 = time()
    print(f"step1 took {s2-s1} seconds")
    # step 2 - find the best treatment for each subpopulation
    if not treatments_func:
        df_treatments = find_best_treatment(D, max_outcome)
    else:
        df_treatments = treatments_func(D, max_outcome)
    if len(df_treatments) == 0:
        return 0
    pd.DataFrame(df_treatments).to_csv(f"outputs/{D.name}/subpopulations_and_treatments.csv", index=False)
    logger.info(f"Saved results to outputs/{D.name}/subpopulations_and_treatments.csv")
    if need_sample:
        D.clean_path = original_clean_path
    logger.critical('Finished')
    s3 = time()
    print(f"step2 took {s3-s2} seconds")
    #step 3 - clustering
    try:
        selected_df = clustering(D, k=k, jaccard_threshold=jaccard_threshold, num_clusters=num_clusters)
        selected_df[['subpop', 'treatment_combo', 'ate1', 'ate2', 'score', 'cluster']].to_csv(f"outputs/{D.name}/clustering_results.csv", index=False)
        jaccard_matrix = print_matrix(D, {}, {}, [[x['subpop'], x['indices']] for _, x in selected_df.iterrows()])
        jaccard_matrix.to_csv(f"outputs/{D.name}/jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
        s4 = time()
        print(f"step3 took {s4-s3} seconds")
        return sum(selected_df['score'])
    except Exception as e:
        return 0


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
               dag_file="data/meps/causal_graph.dot")
so = Dataset(name="so", outcome_col="ConvertedSalary",
               treatments=['YearsCodingProf', 'Hobby', 'FormalEducation', 'WakeTime', 'HopeFiveYears', 'Dependents',
                           'HoursComputer', 'UndergradMajor', 'CompanySize', 'Student', 'Exercise'],
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
             dag_file="data/so/causal_graph.dot")
acs = Dataset(name="acs", outcome_col="HealthInsuranceCoverageRecode",
              treatments=['TemporaryAbsenceFromWork', 'WorkedLastWeek',
                          "TotalPersonEarnings", "UsualHoursWorkedPerWeekPast12Months", 'PersonWeight',
                          'EducationalAttainment', 'GaveBirthWithinPastYear', "FieldOfDegreeScienceAndEngineeringFlag",
                          'SchoolEnrollment'],
              subpopulations=['Sex', 'Age', 'WithADisability', "RaceEthnicity",
                              'Region', 'LanguageOtherThanEnglishSpokenAtHome', 'StateCode',
                              'MaritalStatus', 'Nativity', 'RelatedChild', 'CitizenshipStatus'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_graph.dot")

if __name__ == "__main__":
    r = 0
    start2 = time()
    try:
        r = 0
        r=algorithm(D=meps)
    except Exception:
        print("fail run meps")
    e12 = time()
    print(f"result for meps: {r}")
    print(f"meps took {e12 - start2}")
    try:
        r = 0
        r=algorithm(D=so)
    except Exception:
        print("fail run so")
    e22 = time()
    print(f"result for so: {r}")
    print(f"so took {e22 - e12}")
    try:
        r = algorithm(D=acs)
    except Exception:
        print("fail run acs")
    e32 = time()
    print(f"result for acs: {r}")
    print(f"acs took {e32 - e22}")



