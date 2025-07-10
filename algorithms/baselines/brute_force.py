from algorithms.final_algorithm.new_greedy import parse_subpopulation, print_matrix, get_union, get_intersection
# from algorithms.final_algorithm.find_treatment_new import find_best_treatment
from algorithms.baselines.find_best_treatment_for_brute_force import find_best_treatment
# from algorithms.final_algorithm.find_best_treatment import find_best_treatment

import pandas as pd
import numpy as np
import warnings
from Utils import Dataset, calc_sim, get_indices
import itertools
import ast
from tqdm import tqdm
import csv
import random
from algorithms.final_algorithm.full import acs, so, meps
import time


warnings.filterwarnings("ignore")

K = 11
THRESHOLD = 1.0

"""
1. for each subpopulation -> find his arg-max treatment
2. run all over subpopulations -> find arg-max group
"""

def get_score(group, max_outcome, dict_scoress):
    g = []
    score = 0
    for _, row in group:
        g.append(row)
        score += row['score']
    for row1, row2 in itertools.combinations(g, 2):
        if f'{row1['subpop']}_{row2["subpop"]}' in dict_scoress:
            jaccard = dict_scoress[f'{row1["subpop"]}_{row2["subpop"]}']
        elif f'{row2["subpop"]}_{row1["subpop"]}' in dict_scoress:
            jaccard = dict_scoress[f'{row2["subpop"]}_{row1["subpop"]}']
        else:
            jaccard = calc_sim(row1['indices'], row2['indices'])
            dict_scoress[f'{row1['subpop']}_{row2["subpop"]}'] = jaccard
        if jaccard > THRESHOLD:
            return {"score": 0}
    return {"score": score}


def run_search(k, df_treatments, max_outcome):
    max_score = 0
    dict_scoress = {}
    possibles = []
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), k)):
        scores = get_score(group=group, max_outcome=max_outcome, dict_scoress=dict_scoress)
        if scores["score"] > max_score:
            max_score = scores["score"]
            possibles.append([group, scores["score"]])
            res_group = group
            scores_dict = scores
    if max_score > 0:
        return max_score, res_group, scores_dict
    else:
        return 0, [], {}

def baseline(d: Dataset):
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    data = pd.read_csv(d.clean_path)
    max_outcome = max(data[d.outcome_col])
    # df_treatments = find_best_treatment(d, max_outcome)
    # pd.DataFrame(df_treatments).to_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv", index=False)
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("sample", "clean")
    df_treatments = pd.read_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv")
    df_treatments['indices'] = df_treatments['subpop'].apply(get_indices, data=data)
    max_score, res_group, scores_dict = run_search(K, df_treatments, max_outcome)
    g = []
    for x in res_group:
        _, row = x
        g.append(row)
    res_df = pd.DataFrame(g)
    res_df = res_df.drop(columns=['indices'], axis=1)
    print(max_score)
    # res_df.to_csv(f'outputs/{d.name}/baselines/facts_naive.csv', index=False)
    # pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/naive_scores.csv')
    # jaccard_matrix = print_matrix(d, {}, {}, [[x['subpop'], x['indices']] for _, x in pd.DataFrame(g).iterrows()])
    # jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/naive_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    start = time.time()
    baseline(so)
    e1 = time.time()
    print(f"so took {e1-start} seconds")
    # baseline(meps)
    e2 = time.time()
    print(f"meps took {e2-e1} seconds")
    # baseline(acs)
    e3 = time.time()
    print(f"acs took {e3-e2} seconds")


