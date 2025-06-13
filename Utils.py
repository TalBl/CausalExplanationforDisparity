import numpy as np
from dowhy import CausalModel
import copy
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import pickle
import pandas as pd
import ast
import logging
logger = logging.getLogger(__name__)

MAX_K=7

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


def choose_lamda(data, target_range=1.0):
    data_range = np.percentile(data, 90)
    lamda = data_range
    return lamda

# def choose_lamda(data, upper_limit=0.999):
#     data = np.asarray(data)
#
#     def condition(lamda):
#         scores = ni_score(data, lamda)
#         return np.percentile(scores, 90) - upper_limit
#
#     # Binary search to find lamda where 95th percentile score is just below `upper_limit`
#     low, high = 0.0001, 100
#     for _ in range(10000):
#         mid = (low + high) / 2
#         if condition(mid) > 0:
#             high = mid
#         else:
#             low = mid
#     return low

