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

THRESHOLD_SUPPORT = 0.05
ALPHA = 0.65
K = 5


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


def get_score(group, alpha, d, N, L):
    intersection = 0
    g = []
    ni_score_sum = 0
    for _, row in group.iterrows():
        g.append(row)
        if row["ni_score"]:
            ni_score_sum += row['ni_score'] * row['support']
    utility = ni_score_sum
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def baseline(d: Dataset):
    with open(f"data/{d.name}/causal_dag.txt", "r") as f:
        dag = f.readlines()
    p_value_threshold = 0.05 if d.name != "meps" else 0.1
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    treatments, _ = findBestTreatment("", d, dag, p_value_threshold)
    treats_size = len(treatments)
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("sample", "clean")
    df_clean = pd.read_csv(d.clean_path)
    subgroups = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    res = []
    L = subgroups.shape[0]
    N = df_clean.shape[0]
    for s in subgroups.iterrows():
        _, row = s
        population = get_subpopulation(df_clean, row['itemset'])
        df_group1 = population.loc[population['group1']==1]
        df_group2 = population.loc[population['group2']==1]
        result = CalcIScore(treatments, d, dag, treats_size, df_group1, df_group2, p_value_threshold)
        population = population.loc[(population['group1']==1) | (population['group2']==1)]
        size = population.shape[0]
        support = size / df_clean.shape[0]
        size_group1 = df_group1.shape[0]
        size_group2 = df_group2.shape[0]
        diff_means = np.mean(df_group1[d.outcome_col]) - np.mean(df_group2[d.outcome_col])
        if result:
            res.append({'subpopulation': str(ast.literal_eval(f"{{{str(row['itemset'])[11:-2]}}}")), 'treatment': treatments, 'cate1': result[1], 'cate2': result[2],
                            'iscore': result[0], 'size_subpopulation': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                            "diff_means": diff_means, "avg_group1": np.mean(df_group1[d.outcome_col]),
                            "avg_group2": np.mean(df_group2[d.outcome_col])})
        else:
            res.append({'subpopulation': str(ast.literal_eval(f"{{{str(row['itemset'])[11:-2]}}}")), 'treatment': treatments, 'cate1': None, 'cate2': None,
                        'iscore': None, 'size_subpopulation': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                        "diff_means": diff_means, "avg_group1": np.mean(df_group1[d.outcome_col]),
                        "avg_group2": np.mean(df_group2[d.outcome_col])})
    df = pd.DataFrame(res)
    df_clean = pd.read_csv(d.clean_path)
    lamda = choose_lamda(df_clean[d.outcome_col])
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda) if x else None)
    df = df.sort_values(by=['ni_score', 'support'], ascending=(False, False)).head(K)
    df.to_csv(f'outputs/{d.name}/baselines/facts_de.csv', index=False)
    scores = get_score(group=df, alpha=ALPHA, d=d, N=N, L=L)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/de_scores.csv')


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
