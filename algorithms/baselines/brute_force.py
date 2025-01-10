from algorithms.final_algorithm.new_greedy import parse_subpopulation, get_intersection
from algorithms.final_algorithm.find_treatment_new import find_best_treatment
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
import pandas as pd
import numpy as np
import warnings
from Utils import Dataset, choose_lamda
import itertools
from tqdm import tqdm
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


warnings.filterwarnings("ignore")

K = 5
alpha = 0.65

"""
1. for each subpopulation -> find his arg-max treatment
2. run all over subpopulations -> find arg-max group
"""

def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))

def get_score(group, alpha, d, max_subpopulation):
    intersection = 0
    g = []
    for x in group:
        _, row = x
        g.append(row)
    ni_score_sum = sum(x['ni_score'] for x in g)
    utility = ni_score_sum / len(group)
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d)
    f_intersection = ((max_subpopulation*len(group)*len(group)) - intersection) / (max_subpopulation*len(group)*len(group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def baseline(d: Dataset):
    df_treatments = find_best_treatment(d)
    df_treatments.to_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv", index=False)
    df_treatments = pd.read_csv(f"outputs/{d.name}/naive_subpopulations_and_treatments.csv")
    max_score = 0
    res_group = pd.DataFrame()
    scores_dict = {}
    lamda = choose_lamda([df_treatments["iscore"]])
    df_treatments['ni_score'] = df_treatments['iscore'].apply(lambda x: ni_score(x, lamda))
    df_facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    max_subpopulation = max(df_facts['size_subpopulation'])
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), K)):
        scores = get_score(group=group, alpha=alpha, d=d, max_subpopulation=max_subpopulation)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    g = []
    for x in res_group:
        _, row = x
        g.append(row)
    pd.DataFrame(g).to_csv(f'outputs/{d.name}/baselines/facts_naive.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/naive_scores.csv')


acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=False, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, can_ignore_treatments_filter=True)

baseline(acs)
