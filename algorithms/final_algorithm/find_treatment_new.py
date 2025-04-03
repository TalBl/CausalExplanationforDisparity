import pandas as pd
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import networkx as nx
from dowhy import CausalModel
import pickle
import os
import warnings
from Utils import Dataset
import itertools
import logging
import re
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore")

def changeDAG(dag, treatments):
    DAG = copy.deepcopy(dag)
    toRomove = []
    toAdd = ['TempTreatment;']
    for a in treatments:
        if 'DevType' in a:
            a = "DevType"
        for c in DAG:
            if '->' in c:
                words_only = [re.sub(r'[^\w\s]', '', x).replace('\n', '') for x in c.split((" -> "))]
                if a in words_only:
                    toRomove.append(c)
                    # left hand side
                    if a in c.split(" ->")[0]:
                        string = c.replace(a, "TempTreatment")
                        if not string in toAdd:
                            toAdd.append(string)
    for r in toRomove:
        if r in DAG:
            DAG.remove(r)
    for a in toAdd:
        if not a in DAG:
            DAG.append(a)
    return list(set(DAG))


def calc_causal_graph_per_treatment(df, GRAPH, d, treatment):
    DAG_ = changeDAG(GRAPH, [treatment])
    edges = []
    found = False
    for line in DAG_:
        if '->' in line:
            if not found and "TempTreatment ->" in line:
                found = True
            if line[0] == '"':
                edges.append([line.split(" ->")[0].split('"')[1], line.split("-> ")[1].split(';"')[0]])
            else:
                if line[0] == "'":
                    edges.append([line.split(" ->")[0].split("'")[1], line.split("-> ")[1].split(";'")[0]])
    if not found:
        return
    causal_graph = nx.DiGraph()
    causal_graph.add_edges_from(edges)
    df_group = df.copy()
    if 'DevType' == treatment:
        treatment = "DevType_Systemadministrator"
    for val in list(df_group[treatment].dropna().unique()):
        df_group['TempTreatment'] = df_group[treatment].apply(lambda x: 1 if x and x == val else 0)
        try:
            model = CausalModel(
                data=df_group,
                graph=causal_graph,
                treatment="TempTreatment",
                outcome=d.outcome_col)
            estimands = model.identify_effect()
        except Exception as e:
            raise e
        if not estimands.no_directed_path:
            with open(f"data/{d.name}/causal_dags_files/{treatment}.pkl", 'wb') as file:
                pickle.dump(causal_graph, file)
                return
import os

def remove_all_files_in_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def write_pickle_files(d: Dataset):
    remove_all_files_in_dir(f"data/{d.name}/causal_dags_files")
    df = pd.read_csv(d.clean_path)
    with open(f"{d.dag_file}", "r") as f:
        graph = f.readlines()
    cleaned_treats = []
    for t in d.treatments_atts:
        try:
            calc_causal_graph_per_treatment(df, graph, d, t)
        except Exception as e:
            print("fail")
        if os.path.isfile(f"data/{d.name}/causal_dags_files/{t}.pkl"):
            cleaned_treats.append(t)
    d.treatments_atts = cleaned_treats



def GenChildren(d: Dataset):
    df = pd.read_csv(d.clean_path)
    children = []
    treatments = d.treatments_atts
    if 'DevType' in treatments:
        # Find all columns in df that start with 'DevType' and add them to the list of atts
        dev_type_atts = list(filter(lambda att: att.startswith('DevType_'), df.columns))
        treatments.remove('DevType')
        treatments.extend(dev_type_atts)
    for att in d.treatments_atts:
        for val in df[att].dropna().unique():
            if type(val) in [int, np.int64, np.float64] and val < 0:
                continue
            children.append(((att, val),))
    return children


def getTreatmentATE(df_group, causal_graph, treatments, outcome_col, p_value_param):
    df_group['TempTreatment'] = df_group.apply(lambda row: int(all(row[attr] == val for attr, val in treatments)), axis=1)
    if df_group.loc[df_group['TempTreatment'] == 1].shape[0] < 50:
        return None
    try:
        model = CausalModel(
            data=df_group,
            graph=causal_graph,
            treatment='TempTreatment',
            outcome=outcome_col)
        estimands = model.identify_effect()
        causal_estimate_reg = model.estimate_effect(estimands,
                                                    method_name="backdoor.linear_regression",
                                                    target_units="ate",
                                                    effect_modifiers=[],
                                                    test_significance=True)
        ate, p_value = causal_estimate_reg.value, causal_estimate_reg.test_stat_significance()['p_value']
        if isinstance(p_value, np.ndarray): # When running on SO data - p_value returned as an array
            p_value = p_value[0]
        if p_value > p_value_param:
            return None
    except Exception as e:
        return None
    return ate


def calc_dag(lines):
    edges = []
    for line in lines:
        if '->' in line:
            if line[0] == '"':
                edges.append([line.split(" ->")[0].split('"')[1], line.split("-> ")[1].split(';"')[0]])
            else:
                if line[0] == "'":
                    edges.append([line.split(" ->")[0].split("'")[1], line.split("-> ")[1].split(";'")[0]])
    causal_graph = nx.DiGraph()
    causal_graph.add_edges_from(edges)
    return causal_graph


def ComputeCATEnFilter(treats: set, d: Dataset, dag, depth, dict_res: dict, df_group1, df_group2, p_value):
    # check if all his parents exists -> calc CATE -> if exists + pass p_val
    # -> check if sign of iscore equal the sign diff avg
    try:
        if depth - 1 >= 1:
            for comb in itertools.combinations(treats, depth-1):
                if comb not in dict_res[depth - 1].keys():
                    return None
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
            iscore = abs(ate_group1 - ate_group2)
            if d.can_ignore_treatments_filter:
                return iscore, ate_group1, ate_group2
            if d.need_filter_treatments:
                if d.func_filter_treats(ate_group1, ate_group2):
                    logger.critical("Found Treatment")
                    return iscore, ate_group1, ate_group2, p_value
                else:
                    return None
            return iscore, ate_group1, ate_group2, p_value
    except Exception as e:
        print(e)


def GetTopTreatment(candidates):
    if not candidates:
        return None
    return max({(k, v): v for k, v in candidates.items() if v is not None}.keys(), key=lambda k: k[0])


def GenChildrenNextLevel(res_dict: dict, depth: int):
    children = []
    for treat_0 in res_dict[1].keys():
        for treats in res_dict[depth - 1].keys():
            l = list(treats)
            l.extend(list(treat_0))
            new_t = tuple(set(l))
            if len(new_t) == depth: # if treat_0 already appear in treats then skip
                children.append(new_t)
    return list(set(children))


def get_subpopulation(df, s):
    if type(s) != str:
        s = str(s)
    elements = ast.literal_eval(f"{{{s[11:-2]}}}")
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        try:
            value = float(value)
        except:
            value = value
        item_set[key] = value
    for key, value in item_set.items():
        df = df[df[key] == value]
    return df


def findBestTreatment(subpopulation_str, d: Dataset, dag, p_value):
    max_workers = 100 if d.name == "acs" else None
    df = pd.read_csv(d.clean_path)
    if subpopulation_str != "":
        subpopulation = get_subpopulation(df, subpopulation_str)
    else: # for overall data case in baseline
        subpopulation = df
    df_group1 = subpopulation.loc[subpopulation['group1'] == 1]
    df_group2 = subpopulation.loc[subpopulation['group2'] == 1]
    # Step 1: Get all single-predicate patterns.
    candidates = GenChildren(d)
    depth = 1
    dict_res = {}
    dict_res[depth] = {}
    # Step 2: Compute CATE and filter candidates.
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor2:
            future_to_item = {executor2.submit(ComputeCATEnFilter, i, d.copy(), dag, depth,
                                               dict_res, df_group1.copy(), df_group2.copy(), p_value): i for i in candidates}
            for future in as_completed(future_to_item):
                i = future_to_item[future]
                try:
                    r = future.result()
                    if r:
                        dict_res[depth][i] = future.result()
                except Exception as e:
                    print(f"Error: {e}", flush=True)
    except Exception as e:
        print(e)

    # Step 3: Find the pattern with the maximum CATE.
    max_treatment = GetTopTreatment(dict_res[depth])
    while True:
        depth += 1
        dict_res[depth] = {}
        # Step 4: Generate patterns for the next level.
        candidates = GenChildrenNextLevel(dict_res, depth)

        # Step 5: Compute CATE and filter candidates for the next level.
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor2:
                future_to_item = {executor2.submit(ComputeCATEnFilter, i, d.copy(), dag, depth,
                                                   dict_res, df_group1.copy(), df_group2.copy(), p_value): i for i in candidates}
                for future in as_completed(future_to_item):
                    i = future_to_item[future]
                    try:
                        r = future.result()
                        if r:
                            dict_res[depth][i] = future.result()
                    except Exception as e:
                        print(f"Error: {e}", flush=True)
        except Exception as e:
            print(e)
        # Step 6: Find the top treatment for the current level.
        top_treatment = GetTopTreatment(dict_res[depth])
        if top_treatment and top_treatment[1][0] > max_treatment[1][0]:  # Compare CATE scores.
            max_treatment = top_treatment
        else:
            break
    return max_treatment


def run_subpopulations(d):
    max_workers = 10 if d.name == "acs" else None
    p_value_threshold = 0.05 if d.name != "meps" else 0.1
    subpopulations = list(pd.read_csv(f"outputs/{d.name}/interesting_subpopulations.csv")['itemset'])
    results = {}
    with open(f"{d.dag_file}", "r") as f:
        dag = f.readlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(findBestTreatment, s, d.copy(), dag, p_value_threshold): s for s in subpopulations}
        for future in as_completed(future_to_item):
            i = future_to_item[future]
            try:
                results[i] = future.result()
                logger.critical("finish population")
            except Exception as e:
                print(e)
    return results


def find_best_treatment(d: Dataset):
    write_pickle_files(d)
    results = run_subpopulations(d)
    processed_results = []
    for k, value in results.items():
        k_str = ast.literal_eval(f"{{{str(k)[11:-2]}}}")
        if not value:
            continue
        treatment_tuple, cate_data = value
        processed_results.append({"subpopulation": k_str, "treatment": treatment_tuple, "iscore": cate_data[0], "cate_group1": cate_data[1], "cate_group2": cate_data[2]})
    df_results = pd.DataFrame(processed_results)
    return df_results
