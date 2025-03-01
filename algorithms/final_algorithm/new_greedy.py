import pandas as pd
import numpy as np
import itertools
import ast
from tqdm import tqdm
from Utils import Dataset, choose_lamda
from concurrent.futures import ThreadPoolExecutor, as_completed


# CALCED_INTERSECTIONS = {}


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


# Convert values to appropriate types
def convert_value(val):
    try:
        return float(val)
    except ValueError:
        return val


def parse_subpopulation(input_str):
    items = list(ast.literal_eval(input_str))
    res_dict = {}
    for t in items:
        val = convert_value(t.split("=")[-1])
        key = t.replace("="+t.split("=")[-1], "")
        res_dict[key] = val
    return res_dict


def calc_facts_metrics(dataset: Dataset, lamda):
    data = pd.read_csv(dataset.clean_path)
    df_metadata = pd.read_csv(f"outputs/{dataset.name}/subpopulations_and_treatments.csv")
    n = data.shape[0]
    results = []
    for idx, (itemset, treatment, iscore, cate1, cate2) in df_metadata.iterrows():
        subpopulation = parse_subpopulation(itemset)
        population = data.copy()
        for key, value in subpopulation.items():
            population = population[population[key] == value]
        size = population.shape[0]
        belong_groups = population.loc[(population['group1'] == 1) | (population['group2'] == 1)]
        support = belong_groups.shape[0] / n
        df_group1 = population.copy()
        df_group1 = df_group1.loc[df_group1['group1'] == 1]
        df_group2 = population.copy()
        df_group2 = df_group2.loc[df_group2['group2'] == 1]
        size_group1 = df_group1.shape[0]
        size_group2 = df_group2.shape[0]
        diff_means = np.mean(df_group1[dataset.outcome_col]) - np.mean(df_group2[dataset.outcome_col])
        results.append({'subpopulation': itemset, 'treatment': treatment, 'cate1': cate1, 'cate2': cate2,
                        'iscore': iscore, 'size_subpopulation': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                        "diff_means": diff_means, "avg_group1": np.mean(df_group1[dataset.outcome_col]),
                        "avg_group2": np.mean(df_group2[dataset.outcome_col])})
    if not lamda:
        lamda = choose_lamda([data[dataset.outcome_col]])
        print(lamda)
    df = pd.DataFrame(results)
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda))
    return df


def get_intersection(att1, att2, d, CALCED_INTERSECTIONS):
    if "_".join([att1['subpopulation'], att2['subpopulation']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att2['subpopulation']])]
    if "_".join([att2['subpopulation'], att1['subpopulation']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att2['subpopulation'], att1['subpopulation']])]
    item_set1 = ast.literal_eval(att1['subpopulation'])
    item_set2 = ast.literal_eval(att2['subpopulation'])
    if type(item_set1) == set:
        item_set1 = parse_subpopulation(att1['subpopulation'])
        item_set2 = parse_subpopulation(att2['subpopulation'])
    population = pd.read_csv(d.clean_path)
    for key, value in item_set1.items():
        population = population[population[key] == value]
        if population.shape[0] == 0:
            CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att2['subpopulation']])] = 0
            return 0
    for key, value in item_set2.items():
        population = population[population[key] == value]
    if population.shape[0] == 0:
        CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att2['subpopulation']])] = 0
        return 0
    r = population.shape[0]
    CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att2['subpopulation']])] = r
    return r


def get_score(L, group, attribute, alpha, d, CALCED_INTERSECTION, N):
    intersection = 0
    checked_group = group.copy()
    checked_group.append(attribute)
    ni_score_sum = sum([i['ni_score']*i['support'] for i in checked_group])
    utility = ni_score_sum
    if group:
        for pair in itertools.combinations(checked_group, 2):
            intersection += get_intersection(pair[0], pair[1], d, CALCED_INTERSECTION)
    f_intersection = ((N * L * L) - intersection) / (N * L * L)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score}


def greedy(d, df_facts, L, alpha, K, N):
    j = 0
    group = []
    scores = []
    items = []
    CALCED_INTERSECTION = {}
    K = min(K, df_facts.shape[0])
    while j < K:
        results = []
        for _, group_rows in tqdm(df_facts.iterrows(), total=df_facts.shape[0]):
            if group_rows['subpopulation'] in items:
                continue
            r = get_score(L, group, group_rows, alpha, d, CALCED_INTERSECTION, N)
            if r:
                r["row"] = group_rows
                results.append(r)
        score_dictionary = max(results, key=lambda k: k["score"])
        group.append(score_dictionary['row'])
        items.append(score_dictionary['row']['subpopulation'])
        scores.append(score_dictionary)
        j += 1
    return group, scores, score_dictionary


def find_group(d, K, alpha, L, N):
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    # max_subpopulation = max(df_facts['size_subpopulation'])
    group, scores, score_dictionary = greedy(d, df_facts, L, alpha, K, N)
    df_calc = pd.concat(group, axis=1)
    transposed_df1 = df_calc.T
    transposed_df1.to_csv(f"outputs/{d.name}/find_k/{K}_{alpha}.csv", index=False)
    pd.DataFrame(scores).to_csv(f"outputs/{d.name}/scores/{K}_{alpha}.csv", index=False)
    return score_dictionary

