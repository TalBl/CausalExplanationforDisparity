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

THRESHOLD_SUPPORT = 0.05
ALPHA = 0.65
K = 5


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


def get_score(group, alpha, d, N, L):
    intersection = 0
    g = []
    ni_score_sum = 0
    for _, row in group.iterrows():
        g.append(row)
        if row["ni_score"]:
            ni_score_sum += row['ni_score'] * row['support']
    utility = ni_score_sum
    for row1, row2 in itertools.combinations(g, 2):
        intersection += get_intersection(row1, row2, d, {})
    f_intersection = ((N*L*L) - intersection) / (N*L*L)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"ni_score_sum": ni_score_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


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


def baseline(d: Dataset):
    df = pd.read_csv(d.clean_path)
    N = df.shape[0]
    with open(f"data/{d.name}/causal_dag.txt", "r") as f:
        lines = f.readlines()
    p_value_threshold = 0.05 if d.name != "meps" else 0.1
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    treatments, _ = findBestTreatment("", d, lines, p_value_threshold)
    print(treatments)
    df = pd.read_csv(d.clean_path)
    df['TempTreatment'] = df.apply(lambda row: int(all(row[attr] == val for attr, val in treatments)), axis=1)
    if len(treatments) > 1:
        dag = changeDAG(lines, [x[0] for x in treatments])
    else:
        treatment_att_file_name = list(treatments)[0][0]
        if treatment_att_file_name.startswith('DevType'):
            treatment_att_file_name = 'DevType'
        with open(f"data/{d.name}/causal_dags_files/{treatment_att_file_name}.pkl", 'rb') as file:
            dag = pickle.load(file)
    df = df.dropna(subset=d.subpopulations_atts)
    y = df[d.outcome_col]
    X = df.drop(d.outcome_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train[d.outcome_col] = y_train
    X_test[d.outcome_col] = y_test
    d.subpopulations_atts.extend(["TempTreatment", d.outcome_col])
    X_train = X_train[d.subpopulations_atts]
    X_train.to_csv(f"outputs/{d.name}/train.csv", index=False)
    X_test = X_test.reset_index()
    group1_idx = X_test.loc[X_test['group1']==1.0].index.tolist()
    group2_idx = X_test.loc[X_test['group2']==1.0].index.tolist()
    X_test = X_test[d.subpopulations_atts]
    X_test.to_csv(f"outputs/{d.name}/test.csv", index=False)
    if d.name == "so":
        myDataset = SODataset(rootTrain=f"outputs/{d.name}/train.csv", rootTest=f"outputs/{d.name}/test.csv")
    else:
        if d.name == "meps":
            myDataset = MEPSDataset(rootTrain=f"outputs/{d.name}/train.csv", rootTest=f"outputs/{d.name}/test.csv")
        else:
            myDataset = ACSDataset(rootTrain=f"outputs/{d.name}/train.csv", rootTest=f"outputs/{d.name}/test.csv")

    # fairnessDebug = FairnessDebuggingUsingMachineUnlearning(myDataset,
    #                                                         ["TempTreatment", 1, 0],
    #                                                         d.outcome_col,
    #                                                         FairnessMetric.CATE,
    #                                                         dag,
    #                                                         group1_idx,
    #                                                         group2_idx
    #                                                         )
    #
    # bias_inducing_subsets = fairnessDebug.latticeSearchSubsets(2, (0.05, 1), "normal", False)
    # print(bias_inducing_subsets)
    # bias_inducing_subsets.sort_values(by=["Parity_Reduction","Size"], ascending=[False, False], ignore_index=True)\
    #     .to_csv(f"outputs/{d.name}/baselines/facts_rf.csv", index=False)
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("sample", "clean")
    df_facts = pd.read_csv(f"outputs/{d.name}/baselines/facts_rf.csv")
    df = pd.read_csv(d.clean_path)
    L = df_facts.shape[0]
    res = []
    for _, row in df_facts.iterrows():
        sub = row["Subset"]
        group1, group2, size, support = parse_subpoplation(df, sub)
        result = CalcIScore(treatments, d, lines, len(treatments), group1, group2, p_value_threshold)
        size_group1 = group1.shape[0]
        size_group2 = group2.shape[0]
        diff_means = np.mean(group1[d.outcome_col]) - np.mean(group2[d.outcome_col])
        if result:
            res.append({'subpopulation': sub, 'treatment': treatments, 'cate1': result[1], 'cate2': result[2],
                        'iscore': result[0], 'size_group1': size_group1, "size_group2": size_group2,
                        "diff_means": diff_means, "avg_group1": np.mean(group1[d.outcome_col]),
                        "avg_group2": np.mean(group2[d.outcome_col]), "size": size, "support": support})
        else:
            res.append({'subpopulation': sub, 'treatment': treatments, 'cate1': None, 'cate2': None,
                        'iscore': None, 'size_group1': size_group1, "size_group2": size_group2,
                        "diff_means": diff_means, "avg_group1": np.mean(group1[d.outcome_col]),
                        "avg_group2": np.mean(group2[d.outcome_col]), "size": size, "support": support})
    df = pd.DataFrame(res)
    df_clean = pd.read_csv(d.clean_path)
    lamda = choose_lamda(df_clean[d.outcome_col])
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda) if x else None)
    df = df.sort_values(by=['ni_score', 'support'], ascending=(False, False)).head(K)
    df.to_csv(f'outputs/{d.name}/baselines/facts_final_rf.csv', index=False)
    scores = get_score(group=df, alpha=ALPHA, d=d, N=N, L=L)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/rf_scores.csv')


meps = Dataset(name="meps", outcome_col="FeltNervous",
               treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'IsWorking'],
               columns_to_ignore=[], clean_path="outputs/meps/clean_data.csv",
               func_filter_subs=meps_filter_facts, func_filter_treats=meps_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
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
             clean_path="outputs/so/clean_data.csv", func_filter_subs=so_filter_facts, func_filter_treats=so_filter_facts, need_filter_subpopulations=True, need_filter_treatments=True)
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Temporary absence from work', 'Worked last week', "person weight",
                          'Widowed in the past 12 months', "Total person's earnings",
                          'Educational attainment', 'Georgraphic division'],
              subpopulations=['Sex', 'Age', 'With a disability', 'Region'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

baseline(so)
baseline(meps)
baseline(acs)
