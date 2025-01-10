from algorithms.final_algorithm.new_greedy import get_intersection, ni_score
from algorithms.final_algorithm.find_treatment_new import findBestTreatment, get_subpopulation, getTreatmentATE, changeDAG, calc_dag
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
import pandas as pd
import numpy as np
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from Utils import Dataset, choose_lamda
import itertools
import ast
import pickle

THRESHOLD_SUPPORT = 0.05
# LAMDA = 0.0001
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


def get_score(group, alpha, d, max_subpopulation):
    intersection = 0
    g = []
    for x in group.iterrows():
        _, row = x
        g.append(row)
    ni_score_sum = sum(x['ni_score'] for x in g)
    utility = ni_score_sum / len(group)
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d)
    f_intersection = ((max_subpopulation*len(group)*len(group)) - intersection) / (max_subpopulation*len(group)*len(group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def baseline(d: Dataset):
    with open(f"data/{d.name}/causal_dag.txt", "r") as f:
        dag = f.readlines()
    treatments, _ = findBestTreatment("", d, dag, 0.5)
    print("fount t")
    treats_size = len(treatments)
    df_clean = pd.read_csv(d.clean_path)
    subgroups = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    res = []
    for s in subgroups.iterrows():
        _, row = s
        population = get_subpopulation(df_clean, row['itemset'])
        df_group1 = population.loc[population['group1']==1]
        df_group2 = population.loc[population['group2']==1]
        result = CalcIScore(treatments, d, dag, treats_size, df_group1, df_group2, 1)
        if result:
            size = population.shape[0]
            support = size / df_clean.shape[0]
            size_group1 = df_group1.shape[0]
            size_group2 = df_group2.shape[0]
            diff_means = np.mean(df_group1[d.outcome_col]) - np.mean(df_group2[d.outcome_col])
            res.append({'subpopulation': str(ast.literal_eval(f"{{{str(row['itemset'])[11:-2]}}}")), 'treatment': treatments, 'cate1': result[1], 'cate2': result[2],
                            'iscore': result[0], 'size_subpopulation': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                            "diff_means": diff_means, "avg_group1": np.mean(df_group1[d.outcome_col]),
                            "avg_group2": np.mean(df_group2[d.outcome_col])})
    lamda = choose_lamda([x["iscore"] for x in res])
    df = pd.DataFrame(res)
    max_subpopulation = max(df['size_subpopulation'])
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda))
    df = df.sort_values(by='ni_score', ascending=False).head(K)
    scores = get_score(group=df, alpha=ALPHA, d=d, max_subpopulation=max_subpopulation)
    df.to_csv(f'outputs/{d.name}/baselines/facts_de.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/de_scores.csv')


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
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts,
             need_filter_subpopulations=True, can_ignore_treatments_filter=True)


meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['DoesDoctorRecommendExercise', 'TakesAspirinFrequently', 'BMI', 'Exercise',
                           'CurrentlySmoke'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Education',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],
               columns_to_ignore=['Education=UnAcceptable', 'IsWorking=UnAcceptable'], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True,
               can_ignore_treatments_filter=True)

acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, can_ignore_treatments_filter=True)

#baseline(so) 0.4
#baseline(meps) 0.4
#baseline(acs) 0.5
