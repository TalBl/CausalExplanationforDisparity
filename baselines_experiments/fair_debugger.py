import math
import time
import copy
import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.model_selection import train_test_split
from algorithms.final_algorithm.new_greedy import get_intersection, ni_score
from algorithms.final_algorithm.find_treatment_new import findBestTreatment, get_subpopulation
from algorithms.debug_RF.DebugRF import Dataset_RF, FairnessMetric, FairnessDebuggingUsingMachineUnlearning
from algorithms.final_algorithm.find_treatment_new import changeDAG, getTreatmentATE
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from Utils import Dataset, choose_lamda
import itertools

THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5


'''Class for loading and preprocessing german credits dataset'''
class SODataset(Dataset_RF):
    def __init__(self, rootTrain, rootTest):
        Dataset_RF.__init__(self, rootTrain = rootTrain, rootTest = rootTest)
        self.train = self.trainDataset
        self.test = self.testDataset
        self.trainProcessed, self.testProcessed = self.__preprocessDataset(self.train), self.__preprocessDataset(self.test)
        self.trainLattice, self.testLattice = self.__preprocessDatasetForCategorization(self.train), self.__preprocessDatasetForCategorization(self.test)

    def getDataset(self):
        return self.dataset, self.train, self.test

    def getDatasetWithNormalPreprocessing(self):
        return self.trainProcessed, self.testProcessed

    def getDatasetWithCategorizationPreprocessing(self, decodeAttributeValues = False):
        if decodeAttributeValues == True:
            return self.__decodeAttributeCodeToRealValues(self.trainLattice), self.__decodeAttributeCodeToRealValues(self.testLattice)
        return self.trainLattice, self.testLattice

    def __preprocessDataset(self, dataset):
        df = copy.deepcopy(dataset)
        df['Gender'] = df['Gender'].apply(lambda x : 1 if x == "Male" else 0).astype(int, errors='ignore')
        df['Age'] = df['Age'].map({'Under 18 years old': 0,'18 - 24 years old': 1, '25 - 34 years old': 2,
                                   '35 - 44 years old': 3, '45 - 54 years old': 4, '55 - 64 years old': 5,
                                   '65 years or older': 6}).astype(int, errors='ignore')
        df['ConvertedSalary'] = df['ConvertedSalary'].apply(lambda x : 1 if x > 20000 else 0)
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('ConvertedSalary'))
        df = df[cols+['ConvertedSalary']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'ConvertedSalary' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['ConvertedSalary'] = df['ConvertedSalary'].apply(lambda x : 1 if x > 10000 else 0).astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('ConvertedSalary'))
        df = df[cols+['ConvertedSalary']]
        return df

    def __decodeAttributeCodeToRealValues(self, dataset):
        """df = copy.deepcopy(dataset)
        map_code_to_real = {
        }
        object_columns = [col for col in df.columns if df[col].dtypes == 'object']
        for col in object_columns:
            df[col] = df[col].map(map_code_to_real[col]).fillna(df[col])"""
        return dataset


class MEPSDataset(Dataset_RF):
    def __init__(self, rootTrain, rootTest):
        Dataset_RF.__init__(self, rootTrain = rootTrain, rootTest = rootTest)
        self.train = self.trainDataset
        self.test = self.testDataset
        self.trainProcessed, self.testProcessed = self.__preprocessDataset(self.train), self.__preprocessDataset(self.test)
        self.trainLattice, self.testLattice = self.__preprocessDatasetForCategorization(self.train), self.__preprocessDatasetForCategorization(self.test)

    def getDataset(self):
        return self.dataset, self.train, self.test

    def getDatasetWithNormalPreprocessing(self):
        return self.trainProcessed, self.testProcessed

    def getDatasetWithCategorizationPreprocessing(self, decodeAttributeValues = False):
        if decodeAttributeValues == True:
            return self.__decodeAttributeCodeToRealValues(self.trainLattice), self.__decodeAttributeCodeToRealValues(self.testLattice)
        return self.trainLattice, self.testLattice

    def __preprocessDataset(self, dataset):
        df = copy.deepcopy(dataset)
        df['MaritalStatus'] = df['MaritalStatus'].map({'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Separated': 3,
                                                       'NeverMarried': 4}).astype(int, errors='ignore')
        df['Region'] = df['Region'].map({'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3, "-1": -1}).astype(int, errors='ignore')
        df['Race'] = df['Race'].map({'White': 0, 'Black': 1, 'MultipleRaces': 2, ' Indian/Alaska': 3, 'Asian/Hawaiian/PacificIls': 4}).astype(int, errors='ignore')
        df['IsWorking'] = df['IsWorking'].map({'UnAcceptable': 0, 'WorkedAtSomePointDuringTheYear': 1, 'NotCurrentlyWorkingButHasAJob': 2, 'ActivelyWorking': 3, 'NotEmployed': 4, "-1": -1}).astype(int, errors='ignore')
        df['FeltNervous'] = df['FeltNervous'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('FeltNervous'))
        df = df[cols+['FeltNervous']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'FeltNervous' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['FeltNervous'] = df['FeltNervous'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('FeltNervous'))
        df = df[cols+['FeltNervous']]
        return df

    def __decodeAttributeCodeToRealValues(self, dataset):
        """df = copy.deepcopy(dataset)
        map_code_to_real = {
        }
        object_columns = [col for col in df.columns if df[col].dtypes == 'object']
        for col in object_columns:
            df[col] = df[col].map(map_code_to_real[col]).fillna(df[col])"""
        return dataset


class ACSDataset(Dataset_RF):
    def __init__(self, rootTrain, rootTest):
        Dataset_RF.__init__(self, rootTrain = rootTrain, rootTest = rootTest)
        self.train = self.trainDataset
        self.test = self.testDataset
        self.trainProcessed, self.testProcessed = self.__preprocessDataset(self.train), self.__preprocessDataset(self.test)
        self.trainLattice, self.testLattice = self.__preprocessDatasetForCategorization(self.train), self.__preprocessDatasetForCategorization(self.test)

    def getDataset(self):
        return self.dataset, self.train, self.test

    def getDatasetWithNormalPreprocessing(self):
        return self.trainProcessed, self.testProcessed

    def getDatasetWithCategorizationPreprocessing(self, decodeAttributeValues = False):
        if decodeAttributeValues == True:
            return self.__decodeAttributeCodeToRealValues(self.trainLattice), self.__decodeAttributeCodeToRealValues(self.testLattice)
        return self.trainLattice, self.testLattice

    def __preprocessDataset(self, dataset):
        df = copy.deepcopy(dataset)
        df['Sex'] = df['Sex'].map({'Man': 0, "Woman": 1}).astype(int, errors='ignore')
        df['Age'] = df['Age'].map({'<17.00': 0, "17.00-34.00": 1, "34.00-50.00": 2, "50.00-64.00": 3, ">64.00": 4}).astype(int, errors='ignore')
        df['With a disability'] = df['With a disability'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        # df['School enrollment'] = df['School enrollment'].map({'No, has not attended in the last 3 months': 0, 'Yes, private school, private college, or home school': 1,
        #                                                'Yes, public school or public college': 2}).astype(int, errors='ignore')
        #df['Language other than English spoken at home'] = df['Language other than English spoken at home'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        #df['Hearing difficulty'] = df['Percent of poverty status'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        # df['Citizenship status'] = df['Citizenship status'].map({'Born abroad of American parent(s)': 0, 'Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas': 1,
        #                                                        'Born in the U.S.': 2, 'Not a citizen of the U.S.': 3,
        #                                                          'U.S. citizen by naturalization': 4}).astype(int, errors='ignore')
        df['Region'] = df['Region'].map({'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3}).astype(int, errors='ignore')
        """df['state code'] = df['state code'].map({'California': 0, 'Florida': 1, 'Illinois': 2, ' New York': 3, 'Ohio': 4,
                                                 'Pennsylvania': 5, 'Texas': 6}).astype(int, errors='ignore')
        df['Percent of poverty status'] = df['Percent of poverty status'].map({'<150.00': 0, '150.00-275.00': 1, '275.00-425.00': 2, '>425.00': 3}).astype(int, errors='ignore')
        df['Marital status'] = df['Marital status'].map({'Divorced': 0, 'Married': 1, 'Never Married': 2, 'Separated': 3, 'Widowed': 4}).astype(int, errors='ignore')
        df['Hearing difficulty'] = df['Percent of poverty status'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        df['Related child'] = df['Related child'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        df['Nativity'] = df['Nativity'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')"""
        df['Health insurance coverage recode'] = df['Health insurance coverage recode'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('Health insurance coverage recode'))
        df = df[cols+['Health insurance coverage recode']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'Health insurance coverage recode' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['Health insurance coverage recode'] = df['Health insurance coverage recode'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('Health insurance coverage recode'))
        df = df[cols+['Health insurance coverage recode']]
        return df

    def __decodeAttributeCodeToRealValues(self, dataset):
        """df = copy.deepcopy(dataset)
        map_code_to_real = {
        }
        object_columns = [col for col in df.columns if df[col].dtypes == 'object']
        for col in object_columns:
            df[col] = df[col].map(map_code_to_real[col]).fillna(df[col])"""
        return dataset


def parse_treatment(treatment_str):
    treatments = ast.literal_eval(treatment_str)
    treatments_size = len(treatments)
    treatements_t = []
    for t in treatments:
        key, value = t
        try:
            value = float(value)
        except:
            value = value
        treatements_t.append([key, value])
    return treatements_t, treatments_size


def CalcIScore(treats: set, d: Dataset, dag, size, df_group1, df_group2, p_value):
    try:
        if size - 1 >= 1:
            dag = changeDAG(dag, [x[0] for x in treats])
        else:
            treatment_att_file_name = list(treats)[0][0]
            if treatment_att_file_name.startswith('DevType'):
                treatment_att_file_name = 'DevType'
            with open(f"data/{d.name}/causal_dags_files/{treatment_att_file_name}.pkl", 'rb') as file:
                dag = pickle.load(file)
        ate_group1 = getTreatmentATE(df_group1, dag, treats, d.outcome_col, p_value)
        ate_group2 = getTreatmentATE(df_group2, dag, treats, d.outcome_col, p_value)
        if ate_group1 and ate_group2: # pass p_value checks
            cate = abs(ate_group1 - ate_group2)
            return cate, ate_group1, ate_group2
        else:
            return None
    except Exception as e:
        print(e)


def stretch_power(x, alpha):
    return 1 - (1 - x)**alpha

def get_score(group, alpha, d, N, L):
    intersection = 0
    g = []
    iscore = 0
    for _, row in group.iterrows():
        g.append(row)
        if pd.notna(row['ni_score']):
            iscore += row['ni_score'] * row['support']
    utility = stretch_power(iscore / L, alpha=6.69998) if iscore > 0 else 0
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    no_overlap = stretch_power(f_intersection, alpha=0.11484)
    score = (alpha * utility) + ((1 - alpha) * no_overlap)
    return {"ni_score_sum": iscore, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "no_overlap": no_overlap, "score": score}


def parse_subpoplation(df_original, sub_str):
    sub_str = ast.literal_eval(sub_str)
    df = df_original.copy()
    for s in sub_str:
        try:
            k, v = s.split("=")
        except:
            print("here")
        try:
            v = int(v)
        except:
            v = v
        df = df.loc[df[k]==v]
    group1 = df.loc[df['group1']==1]
    group2 = df.loc[df['group2']==1]
    size = df.shape[0]
    belong_groups = df.loc[(df['group1'] == 1) | (df['group2'] == 1)]
    support = belong_groups.shape[0] / df_original.shape[0]
    return group1, group2, size, support


def baseline(d: Dataset, K):
    df_facts = pd.read_csv(f"outputs/{d.name}/baselines/facts_rf.csv")
    L = df_facts.shape[0]
    df = pd.read_csv(d.clean_path)
    N = df.shape[0]
    df = pd.read_csv(f"{d.name}_rf_base.csv")
    lamda = choose_lamda([x for x in df['iscore'] if pd.notna(x)])
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda))
    df = df.sort_values(by=['ni_score', 'support'], ascending=(False, False)).head(K)
    scores = get_score(group=df, alpha=ALPHA, d=d, N=N, L=L)
    return scores

meps = Dataset(name="meps", outcome_col="FeltNervous",
               treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'IsWorking'],
               columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
               dag_file="data/meps/causal_dag.txt")
so = Dataset(name="so", outcome_col="ConvertedSalary",
             treatments=['YearsCodingProf', 'Hobby', 'FormalEducation', 'WakeTime', 'DevType'],
             subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                             'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                             'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                             'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent'],
             columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True,
             dag_file="data/so/causal_dag.txt")
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Temporary absence from work', 'Worked last week',
                          'Widowed in the past 12 months', "Total person earnings",
                          'Educational attainment', 'Georgraphic division'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Region'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats, dag_file="data/acs/causal_dag.txt")


l = []
for k in [5]:
    d = baseline(so, k)
    l.append({"baseline": "fair-debugger", "dataset": "so", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(meps, k)
    l.append({"baseline": "fair-debugger", "dataset": "meps", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
    d = baseline(acs, k)
    l.append({"baseline": "fair-debugger", "dataset": "acs", "k": k, "score": d['score'], "utility": d['utility'], "no_overlap": d['no_overlap']})
pd.DataFrame(l).to_csv("outputs/baselines_comparison/fr_results2.csv", index=False)

