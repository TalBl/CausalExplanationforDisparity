import random

import numpy as np
from dowhy import CausalModel
import copy
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import pickle
import pandas as pd
import pydot
import ast
import logging
logger = logging.getLogger(__name__)
import signal

MAX_K=7
MIN_SIZE_FOR_TREATED_GROUP = 50
P_VALUE_THRESHOLD = 0.05

class Dataset:
    def __init__(self, name, outcome_col, treatments, subpopulations, columns_to_ignore, clean_path, func_filter_subs, func_filter_treats,
                 need_filter_subpopulations=False, need_filter_treatments=False, can_ignore_treatments_filter=False, dag_file=None):
        self.name = name
        self.outcome_col = outcome_col
        self.treatments_atts = treatments
        self.subpopulations_atts = subpopulations
        self.columns_to_ignore = columns_to_ignore
        self.clean_path = clean_path
        self.func_filter_subs = func_filter_subs
        self.func_filter_treats = func_filter_treats
        self.need_filter_subpopulations = need_filter_subpopulations
        self.need_filter_treatments = need_filter_treatments
        df = pd.read_csv(self.clean_path)
        df_group1 = df.loc[df['group1'] == 1]
        df_group2 = df.loc[df['group2'] == 1]
        item_set1_avg = df_group1[self.outcome_col].mean()
        item_set2_avg = df_group2[self.outcome_col].mean()
        self.is_avg_diff_positive = item_set1_avg - item_set2_avg > 0
        self.can_ignore_treatments_filter = can_ignore_treatments_filter
        self.dag_file = dag_file

    def copy(self):
        return copy.deepcopy(self)

def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))

def parse_subpopulation_str(s: str):
    elements = ast.literal_eval(f"{{{s[11:-2]}}}")
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        try:
            value = float(value)
        except:
            value = value
        item_set[key] = value
    return item_set

def get_subpopulation_df(d: Dataset, elements: dict):
    df = pd.read_csv(d.clean_path)
    for key, value in elements.items():
        df = df[df[key] == value]
    return df

def getTreatmentATE(df_group, causal_graph, treatments, outcome_col):
    df_group['TempTreatment'] = df_group.apply(lambda row: int(all(row[attr] == val for attr, val in treatments)), axis=1)
    if df_group.loc[df_group['TempTreatment'] == 1].shape[0] < MIN_SIZE_FOR_TREATED_GROUP:
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
        if p_value > P_VALUE_THRESHOLD:
            return None
    except Exception as e:
        return None
    return ate, p_value


def get_indices(s, data):
    s_con = ast.literal_eval(s)
    data_copy = data.copy()
    for attr, val in s_con.items():
        data_copy = data_copy[data_copy[attr] == val]
    return set(data_copy.index)


def calc_sim(A, B):
    intersection = len(A & B)
    union = len(A | B)
    sim = intersection / union if union > 0 else 0
    return sim

