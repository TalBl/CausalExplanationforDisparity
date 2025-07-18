import pandas as pd
import numpy as np
import itertools
import ast
import csv
from tqdm import tqdm
from Utils import Dataset, ni_score
from concurrent.futures import ThreadPoolExecutor, as_completed


# CALCED_INTERSECTIONS = {}
# THRESHOLD = 0.25


# Convert values to appropriate types
def convert_value(val):
    try:
        return float(val)
    except ValueError:
        return val


def parse_subpopulation(input_str):
    items = ast.literal_eval(input_str)
    res_dict = {}
    for k, v in items.items():
        val = convert_value(v)
        res_dict[k] = val
    return res_dict


def calc_facts_metrics(dataset: Dataset):
    data = pd.read_csv(dataset.clean_path)
    df_metadata = pd.read_csv(f"outputs/{dataset.name}/subpopulations_and_treatments.csv")
    n = data.shape[0]
    results = []
    for idx, row in df_metadata.iterrows():
        subpop = row["subpop"]
        treatment_combo = row["treatment_combo"]
        score = row["score"]
        ate1 = row["ate1"]
        ate2 = row["ate2"]
        subpopulation = parse_subpopulation(subpop)
        population = data.copy()
        for key, value in subpopulation.items():
            population = population[population[key] == value]
        belong_groups = population.loc[(population['group1'] == 1) | (population['group2'] == 1)]
        support = belong_groups.shape[0] / n
        df_group1 = belong_groups.copy()
        df_group1 = df_group1.loc[df_group1['group1'] == 1]
        df_group2 = belong_groups.copy()
        df_group2 = df_group2.loc[df_group2['group2'] == 1]
        size_group1 = df_group1.shape[0]
        size_group2 = df_group2.shape[0]
        results.append({'subpop': subpop, 'treatment': treatment_combo, 'cate1': ate1, 'cate2': ate2,
                        'score': score, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                        "avg_group1": np.mean(df_group1[dataset.outcome_col]), "avg_group2": np.mean(df_group2[dataset.outcome_col])})
    df = pd.DataFrame(results)
    return df


def get_intersection(att1, att2, d, CALCED_INTERSECTIONS):
    if "_".join([att1['subpop'], att2['subpop']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att1['subpop'], att2['subpop']])]
    if "_".join([att2['subpop'], att1['subpop']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att2['subpop'], att1['subpop']])]
    item_set1 = ast.literal_eval(att1['subpop'])
    item_set2 = ast.literal_eval(att2['subpop'])
    if type(item_set1) == set:
        item_set1 = parse_subpopulation(att1['subpop'])
        item_set2 = parse_subpopulation(att2['subpop'])
    population = pd.read_csv(d.clean_path)
    for key, value in item_set1.items():
        population = population[population[key] == value]
        if population.shape[0] == 0:
            CALCED_INTERSECTIONS["_".join([att1['subpop'], att2['subpop']])] = 0
            return 0
    for key, value in item_set2.items():
        population = population[population[key] == value]
    if population.shape[0] == 0:
        CALCED_INTERSECTIONS["_".join([att1['subpop'], att2['subpop']])] = 0
        return 0
    r = population.shape[0]
    CALCED_INTERSECTIONS["_".join([att1['subpop'], att2['subpop']])] = r
    return r


def get_union(att1, att2, d, CALCED_UNIONS):
    if "_".join([att1['subpop'], att2['subpop']]) in CALCED_UNIONS:
        return CALCED_UNIONS["_".join([att1['subpop'], att2['subpop']])]
    if "_".join([att2['subpop'], att1['subpop']]) in CALCED_UNIONS:
        return CALCED_UNIONS["_".join([att2['subpop'], att1['subpop']])]
    item_set1 = ast.literal_eval(att1['subpop'])
    item_set2 = ast.literal_eval(att2['subpop'])
    if type(item_set1) == set:
        item_set1 = parse_subpopulation(att1['subpop'])
        item_set2 = parse_subpopulation(att2['subpop'])
    population1 = pd.read_csv(d.clean_path)
    for key, value in item_set1.items():
        population1 = population1[population1[key] == value]
    idx1 = list(population1.index.values)
    population2 = pd.read_csv(d.clean_path)
    for key, value in item_set2.items():
        population2 = population2[population2[key] == value]
    idx2 = list(population2.index.values)
    idxs = list(set(idx1).union(idx2))
    CALCED_UNIONS["_".join([att1['subpop'], att2['subpop']])] = len(idxs)
    return len(idxs)

def stretch_power(x, alpha):
    return 1 - (1 - x)**alpha

def alpha_lift_monotonic(x, midpoint=0.5, alpha=0.0001, target_low=0.3):
    w = 1 / (1 + np.exp(-alpha * (midpoint - x)))  # sigmoid weight
    return (1 - w) * x + w * target_low

def get_score(group, attribute, d, CALCED_INTERSECTION, max_outcome, CALCED_UNIONS, curr_score, threshold):
    checked_group = group.copy()
    checked_group.append(attribute)
    iscore = sum([i['score'] for i in checked_group])
    if iscore <= curr_score:
        return None
    score = iscore
    if group:
        for item in group:
            intersection = get_intersection(attribute, item, d, CALCED_INTERSECTION)
            union = get_union(attribute, item, d, CALCED_UNIONS)
            jackard = intersection / union
            if jackard > threshold:
                return None
    return {"score": score}


def print_matrix(intersections, unions, group):
    jaccard_matrix = pd.DataFrame(index=[str(x[0]) for x in group], columns=[str(x[0]) for x in group])
    for a, idx_a in group:
        for b, idx_b in group:
            inter = len(idx_a & idx_b)
            union = len(idx_a | idx_b)
            jaccard_matrix.at[str(a), str(b)] = round(inter / union, 4)
    # Save to CSV
    return jaccard_matrix


def greedy(d, df_facts, K, max_outcome, threshold):
    group = []
    scores = []
    items = []
    CALCED_INTERSECTION = {}
    CALCED_UNION = {}
    K = min(K, df_facts.shape[0])
    curr_score = 0
    for j in range(K):
        results = None
        for _, group_rows in tqdm(df_facts.iterrows(), total=df_facts.shape[0]):
            if group_rows['subpop'] in items:
                continue
            r = get_score(group, group_rows, d, CALCED_INTERSECTION, max_outcome, CALCED_UNION, curr_score, threshold)
            if r and r['score'] > curr_score:
                r["row"] = group_rows
                results = r
                curr_score = r['score']
        if results:
            score_dictionary = results["score"]
            group.append(results['row'])
            items.append(results['row']['subpop'])
            scores.append(results)
    # jaccard_matrix = print_matrix(d, CALCED_INTERSECTION, CALCED_UNION, items)
    # jaccard_matrix.to_csv(f"outputs/{d.name}/jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    return group, scores, score_dictionary


def find_group(d, K, max_outcome, threshold):
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    group, scores, score_dictionary = greedy(d, df_facts, K, max_outcome, threshold)
    df_calc = pd.concat(group, axis=1)
    transposed_df1 = df_calc.T
    # transposed_df1.to_csv(f"outputs/{d.name}/find_k/{K}.csv", index=False)
    # pd.DataFrame(scores).to_csv(f"outputs/{d.name}/scores/{K}.csv", index=False)
    return score_dictionary

