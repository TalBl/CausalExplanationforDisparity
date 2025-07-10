import random

import pandas as pd
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from dowhy import CausalModel
import pickle
import os
import itertools
import logging

from networkx.drawing.nx_pydot import to_pydot

from Utils import Dataset, get_subpopulation_df, parse_subpopulation_str, getTreatmentATE


logger = logging.getLogger(__name__)

# --- CONFIG ---
NUM_PROCESSES = os.cpu_count() - 2  # Adjust based on cores
ATE_THRESHOLD = 0.1  # Example threshold for sufficient ATE

# --- UTILS ---
def modify_graph_on_demand(base_graph, combo):
    G = base_graph.copy()
    for t in combo:
        if t in G:
            for succ in list(G.successors(t)):
                G.add_edge("TempTreatment", succ)
            G.remove_node(t)
    G.add_node("TempTreatment")
    return G

import os

def delete_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)



def build_single_treatment_pickles(d: Dataset, base_graph: nx.DiGraph):
    os.makedirs(f"data/{d.name}/causal_dags_files", exist_ok=True)
    delete_all_files(f"data/{d.name}/causal_dags_files")
    for treatment in d.treatments_atts:
        G = modify_graph_on_demand(base_graph, [treatment])
        G = prune_graph(G, "TempTreatment", d.outcome_col)
        if nx.has_path(G, source="TempTreatment", target=d.outcome_col):
            file_path = os.path.join(f"data/{d.name}/causal_dags_files", f"graph_{treatment}.dot")
            nx.nx_pydot.write_dot(G, file_path)
            logger.info(f"Saved single treatment graph: {file_path}")
        else:
            logger.info(f"Skipped saving graph for {treatment}: no path to {d.outcome_col}")


def load_valid_treatments(output_dir: str, df: pd.DataFrame):
    valid_files = [f for f in os.listdir(output_dir) if f.startswith("graph_") and f.endswith(".dot")]
    valid_attrs = [f.replace("graph_", "").replace(".dot", "") for f in valid_files]
    valid_treatments = []
    for attr in valid_attrs:
        for val in df[attr].unique():
            if pd.notna(val):
                valid_treatments.append((attr,val))
    return valid_treatments

def prune_graph(G, treatment, outcome):
    ancestors = nx.ancestors(G, treatment).union(nx.ancestors(G, outcome))
    ancestors.update([treatment, outcome])
    G_pruned = G.subgraph(ancestors).copy()
    return G_pruned

# --- MAIN WORK ---
def process_subpopulation(subpop_row: pd.Series, d: Dataset, base_graph: nx.DiGraph, max_outcome):
    if subpop_row.empty:
        df_sub = pd.read_csv(d.clean_path)
        attributes = "ALL"
    else:
        attributes = parse_subpopulation_str(subpop_row['itemset'])
        df_sub = get_subpopulation_df(d, attributes)
    df_group1 = df_sub.loc[df_sub['group1']==1]
    df_group2 = df_sub.loc[df_sub['group2']==1]
    results = []
    initial_valid_treatments = load_valid_treatments(f"data/{d.name}/causal_dags_files", df_sub)
    current_valid_treatments = initial_valid_treatments[:]
    for size in range(1, len(initial_valid_treatments) + 1):
        if not current_valid_treatments:
            break
        combos = list(itertools.combinations(current_valid_treatments, size))
        next_valid_treatments = set()
        for combo in combos:
            if len([x[0] for x in combo]) != len(set([x[0] for x in combo])):
                continue
            try:
                if len(combo) == 1:
                    path_name = f"data/{d.name}/causal_dags_files/graph_{combo[0][0]}.dot"
                else:
                    causal_graph = modify_graph_on_demand(base_graph, combo)
                    causal_graph = prune_graph(causal_graph, "TempTreatment", d.outcome_col)
                    rand_int = random.randint(1, 100)
                    path_name = f"tmp_graph_{rand_int}.dot"
                    nx.nx_pydot.write_dot(causal_graph, f"tmp_graph_{rand_int}.dot")
                res1 = getTreatmentATE(df_group1, path_name, combo, d.outcome_col)
                if res1:
                    res2 = getTreatmentATE(df_group2, path_name, combo, d.outcome_col)
                else:
                    continue
                if res1 and res2 and d.func_filter_treats(res1[0], res2[0]):
                    results.append({
                        'subpop': attributes,
                        'treatment_combo': combo,
                        'ate1': res1[0],
                        'p-value1': res1[1],
                        'ate2': res2[0],
                        'p-value2': res2[1],
                        'score': abs(res1[0] - res2[0]) / max_outcome,
                    })
                    next_valid_treatments.update(combo)
                    logger.critical('Found Treatment')
            except Exception as e:
                logger.error(f"Error for subpop {attributes}, combo {combo}: {e}")
        if len(next_valid_treatments)==0:
            break
        current_valid_treatments = list(next_valid_treatments)
    logger.critical('finish subpopulation process')
    if results:
        return max(results, key=lambda x: x['score'])
    return None


# --- DRIVER ---
def find_best_treatment(d: Dataset, max_outcome):
    if not os.path.exists(d.dag_file):
        logger.error(f"DOT file not found: {d.dag_file}")
        return []
    base_graph = nx.DiGraph(nx.nx_pydot.read_dot(d.dag_file))
    build_single_treatment_pickles(d, base_graph)
    df_subs = pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")
    all_results = []
    for _, row in df_subs.iterrows():
        try:
            res = process_subpopulation(row, d, base_graph, max_outcome)
            if res:
                all_results.append(res)
        except Exception as e:
            logger.error(f"Subpopulation processing failed: {e}")
    return all_results

# if __name__ == "__main__":
#     dot_path = "path_to_dot_files/base.dot"  # Replace with actual path
#     treatment_attributes = ["Treatment1", "Treatment2"]  # Replace with actual treatments
#     output_dir = "single_treatment_graphs"
#     build_single_treatment_pickles(dot_path, treatment_attributes, output_dir)
#     # Example for cProfile: run `python -m cProfile -s cumtime your_script.py`
