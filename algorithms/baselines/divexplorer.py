import networkx as nx

from algorithms.final_algorithm.find_best_treatment import process_subpopulation, modify_graph_on_demand, prune_graph
from algorithms.final_algorithm.new_greedy import get_intersection, get_union, ni_score, print_matrix
from algorithms.final_algorithm.find_treatment_new import findBestTreatment, get_subpopulation, changeDAG, calc_dag
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
import pandas as pd
import numpy as np
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from Utils import Dataset, get_subpopulation_df, parse_subpopulation_str, getTreatmentATE
import itertools
import ast
import pickle
import csv
from tqdm import tqdm


THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5
K = 5
THRESHOLD = 0.55


def get_score(group, d, calc_intersection, calc_union):
    g = []
    iscore = 0
    for _, row in group:
        g.append(row)
        iscore += row['score']
    for row1, row2 in itertools.combinations(g, 2):
        intersection = len(row1['indices'] & row2['indices'])
        union = len(row1['indices'] | row2['indices'])
        jaccard = intersection / union if union != 0 else 0
        if jaccard > THRESHOLD:
            return {"score": 0}
    return {"score": iscore}


def run_search(d, k, df_treatments, calc_intersection, calc_union):
    max_score = 0
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), k)):
        scores = get_score(group=group, d=d, calc_intersection=calc_intersection, calc_union=calc_union)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    if max_score > 0:
        return max_score, res_group, scores_dict
    else:
        return 0, [], {}


def baseline(d: Dataset):
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    base_graph = nx.DiGraph(nx.nx_pydot.read_dot(d.dag_file))
    df_clean = pd.read_csv(d.clean_path)
    max_outcome = max(df_clean[d.outcome_col])
    treatment = process_subpopulation(pd.Series(), d, base_graph, max_outcome)['treatment_combo']
    print(treatment)
    if len(treatment) == 1:
        causal_graph = f"data/{d.name}/causal_dags_files/graph_{treatment[0][0]}.dot"
    else:
        causal_graph = modify_graph_on_demand(base_graph, treatment)
        causal_graph = prune_graph(causal_graph, "TempTreatment", d.outcome_col)
    subgroups = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    res = []
    for s in tqdm(subgroups.iterrows(), total=subgroups.shape[0]):
        _, row = s
        str_subpop = parse_subpopulation_str(row['itemset'])
        population = get_subpopulation_df(d, str_subpop)
        df_group1 = population.loc[population['group1']==1]
        res1 = getTreatmentATE(df_group1, causal_graph, treatment, d.outcome_col)
        if not res1:
            continue
        df_group2 = population.loc[population['group2'] == 1]
        res2 = getTreatmentATE(df_group2, causal_graph, treatment, d.outcome_col)
        if not res2:
            continue
        if d.func_filter_treats(res1[0], res2[0]):
            result = abs(res1[0] - res2[0]) / max_outcome
            res.append({'subpop': str_subpop, 'treatment': treatment, 'ate1': res1[0], 'ate2': res2[0], 'score': result, 'indices': set(population.index)})
    df = pd.DataFrame(res)
    print(f"total {df.shape[0]}")
    g = []
    calc_intersection, calc_union, scores_dict = {}, {}, {}
    if df.shape[0] > 0:
        res_group = []
        k = K
        while res_group == [] and k > 0:
            max_score, res_group, scores_dict = run_search(d, k, df, calc_intersection, calc_union)
            k -= 1
        for x in res_group:
            _, row = x
            g.append(row)
    jaccard_matrix = print_matrix(d, calc_intersection, calc_union, [[x['subpop'], x['indices']] for x in g])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/de_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    pd.DataFrame(g).to_csv(f'outputs/{d.name}/baselines/facts_de.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/de_scores.csv', index=False)


from algorithms.final_algorithm.full import acs, so, meps
import time
start = time.time()
baseline(so)
e1 = time.time()
print(f"so took {e1-start}")
baseline(meps)
e2 = time.time()
print(f"meps took {e2-e1}")
baseline(acs)
e3 = time.time()
print(f"acs took {e3-e2}")
"""
acs took 491.41048097610474
meps took 3.6268370151519775
so took 83.07054781913757
"""
