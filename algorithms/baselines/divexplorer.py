from algorithms.final_algorithm.new_greedy import get_intersection, get_union, ni_score, print_matrix
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
import csv
from tqdm import tqdm


THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5
K = 5
THRESHOLD = 0.25


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



def get_score(group, d, calc_intersection, calc_union, max_outcome):
    g = []
    iscore = 0
    for _, row in group:
        g.append(row)
        iscore += row['iscore'] / max_outcome
    for row1, row2 in itertools.combinations(g, 2):
        intersection = get_intersection(row1, row2, d, calc_intersection)
        union = get_union(row1, row2, d, calc_union)
        jaccard = intersection / union
        if jaccard > THRESHOLD:
            return {"score": 0}
    return {"score": iscore}


def run_search(d, k, df_treatments, calc_intersection, calc_union, max_outcome):
    max_score = 0
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), k)):
        scores = get_score(group=group, d=d, calc_intersection=calc_intersection, calc_union=calc_union, max_outcome=max_outcome)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    if max_score > 0:
        return max_score, res_group, scores_dict
    else:
        return 0, [], {}


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
    max_outcome = max(df_clean[d.outcome_col])
    subgroups = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    if d.need_filter_subpopulations:
        subgroups['condition'] = subgroups.apply(lambda row: d.func_filter_subs(row[f'{d.outcome_col}_group1'], row[f'{d.outcome_col}_group2']), axis=1)
        subgroups = subgroups.loc[subgroups['condition']==True]
    res = []
    for s in tqdm(subgroups.iterrows(), total=subgroups.shape[0]):
        _, row = s
        if d.name == "acs":
            d.clean_path = d.clean_path.replace("clean", "sample")
            df_clean = pd.read_csv(d.clean_path)
        population = get_subpopulation(df_clean, row['itemset'])
        df_group1 = population.loc[population['group1']==1]
        df_group2 = population.loc[population['group2']==1]
        result = CalcIScore(treatments, d, dag, treats_size, df_group1, df_group2, p_value_threshold)
        if d.name == "acs":
            d.clean_path = d.clean_path.replace("sample", "clean")
            df_clean = pd.read_csv(d.clean_path)
        population = get_subpopulation(df_clean, row['itemset'])
        df_group1 = population.loc[population['group1']==1]
        df_group2 = population.loc[population['group2']==1]
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
    df = df.loc[df['iscore'].notnull()]
    g = []
    calc_intersection, calc_union, scores_dict = {}, {}, {}
    if df.shape[0] > 0:
        res_group = []
        k = K
        while res_group == [] and k > 0:
            max_score, res_group, scores_dict = run_search(d, k, df, calc_intersection, calc_union, max_outcome)
            k -= 1
        for x in res_group:
            _, row = x
            g.append(row)
    jaccard_matrix = print_matrix(d, calc_intersection, calc_union, [x['subpopulation'] for x in g])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/de_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    pd.DataFrame(g).to_csv(f'outputs/{d.name}/baselines/facts_de.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/de_scores.csv', index=False)


from algorithms.final_algorithm.full import acs, so, meps
import time
# start = time.time()
# baseline(so)
e1 = time.time()
# print(f"so took {e1-start}")
baseline(meps)
e2 = time.time()
print(f"meps took {e2-e1}")
# baseline(acs)
# e3 = time.time()
# print(f"acs took {e3-e2}")
"""
acs took 35294.20775818825
meps took 35.963284730911255
so took 24.49563431739807


"""

"""
0. table results: one column utility + one column the matrix + one column the runtime - all precentage comparison Brute force results
1. threshold vs score (10% - 75%) multiple minimal support threshold - one graph per dataset
2. threshold vs runtime
3. rerun k vs utility
"""


