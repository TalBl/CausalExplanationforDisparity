import pandas as pd

from algorithms.baselines.find_best_treatment_for_brute_force import find_best_treatment
from algorithms.final_algorithm.new_greedy import print_matrix
from Utils import Dataset, get_indices
import itertools
import numpy as np
import csv


K = 5


def get_score(group, d):
    ni_score_sum = 0
    for _, row in group.iterrows():
        ni_score_sum += row['score']
    jaccard_matrix = print_matrix({}, {}, [[x['subpop'], x['indices']] for _, x in pd.DataFrame(group).iterrows()])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/top_k_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    return {"score": ni_score_sum}


def baseline(d: Dataset):
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    df_clean = pd.read_csv(d.clean_path)
    max_outcome = max(df_clean[d.outcome_col])
    df_treatments = find_best_treatment(d, max_outcome)
    pd.DataFrame(df_treatments).to_csv(f"outputs/{d.name}/baselines/top_k_subpopulations_and_treatments.csv", index=False)
    subs = pd.read_csv(f"outputs/{d.name}/baselines/top_k_subpopulations_and_treatments.csv")
    df_facts_top_k = subs.sort_values(by='score', ascending=False).head(K)
    df_facts_top_k['indices'] = df_facts_top_k['subpop'].apply(get_indices, data=pd.read_csv(d.clean_path))
    scores = get_score(group=df_facts_top_k, d=d)
    df_facts_top_k.to_csv(f'outputs/{d.name}/baselines/facts_top_k.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/top_k_scores.csv')


from algorithms.final_algorithm.full import acs, so, meps
import time
if __name__ == '__main__':
    start_time = time.time()
    start = time.time()
    baseline(so)
    e1 = time.time()
    print(f"so took {e1-start}")
    # baseline(meps)
    e2 = time.time()
    print(f"meps took {e2-e1}")
    # baseline(acs)
    e3 = time.time()
    print(f"acs took {e3-e2}")
