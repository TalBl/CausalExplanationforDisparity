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
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
from Cleaning_Datasets.clean_so import filter_facts as so_filter_facts
from Cleaning_Datasets.clean_meps import filter_facts as meps_filter_facts
from Utils import Dataset, choose_lamda
import itertools

THRESHOLD_SUPPORT = 0.05
# LAMDA = 0.0001
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
        df['Country'] = df['Country'].map({'Latvia': 0, 'Viet Nam': 1, 'Turkey': 2, 'Peru': 3, 'Bangladesh': 4, 'Iraq': 5,
                                           'Myanmar': 6, 'Estonia': 7, 'France': 8, 'Philippines': 9, 'Chile': 10,
                                           'Libyan Arab Jamahiriya': 11, 'The former Yugoslav Republic of Macedonia': 12,
                                           'Ecuador': 13, 'Canada': 14, 'Colombia': 15, 'Gambia': 16, 'Croatia': 17,
                                           'Luxembourg': 18, 'Brazil': 19, 'Portugal': 20, 'Uganda': 21, 'Indonesia': 22,
                                           'Ireland': 23, 'Japan': 24, 'Nicaragua': 25, 'Afghanistan': 26, 'Mongolia': 27,
                                           'Nigeria': 28, 'Slovenia': 29, 'Tunisia': 30, 'Russian Federation': 31,
                                           'Republic of Moldova': 32, 'Jordan': 33, 'Armenia': 34, 'Saudi Arabia': 35,
                                           'United Republic of Tanzania': 36, 'Argentina': 37, 'Malaysia': 38,
                                           'Venezuela, Bolivarian Republic of...': 39, 'Spain': 40, 'United Arab Emirates': 41,
                                           'Singapore': 42, 'Congo, Republic of the...': 43, 'Niger': 44, 'Botswana': 45,
                                           'Democratic Republic of the Congo': 46, 'Italy': 47, 'Andorra': 48, 'Finland': 49,
                                           'Algeria': 50, 'Lebanon': 51, 'Kyrgyzstan': 52, 'Cyprus': 53, 'Kenya': 54,
                                           'Cameroon': 55, 'Paraguay': 56, 'Hungary': 57, 'Denmark': 58, 'Senegal': 59,
                                           'Mauritius': 60, 'Republic of Korea': 61, 'Bulgaria': 62, 'Sweden': 63,
                                           'Netherlands': 64, 'United States': 65, 'Hong Kong (S.A.R.)': 66, 'Ghana': 67,
                                           'Costa Rica': 68, 'Uruguay': 69, 'Bahamas': 70, 'Egypt': 71, 'Romania': 72,
                                           'Lesotho': 73, 'Mexico': 74, 'Bolivia': 75, 'Malta': 76, 'Bosnia and Herzegovina': 77,
                                           'South Africa': 78, 'China': 79, 'Poland': 80, 'India': 81, 'Kazakhstan': 82,
                                           'Bahrain': 83, 'Czech Republic': 84, 'Azerbaijan': 85, 'Serbia': 86, 'El Salvador': 87,
                                           'Iceland': 88, 'Maldives': 89, 'Cuba': 90, 'Other Country (Not Listed Above)': 91,
                                           'Saint Lucia': 92, 'Montenegro': 93, 'United Kingdom': 94, 'Tajikistan': 95, 'Benin': 96,
                                           'Greece': 97, 'Ukraine': 98, 'Barbados': 99, 'Trinidad and Tobago': 100, 'New Zealand': 101,
                                           'Nepal': 102, 'Dominican Republic': 103, 'Yemen': 104, 'Iran, Islamic Republic of...': 105,
                                           'Somalia': 106, 'Haiti': 107, 'Sri Lanka': 108, 'Morocco': 109, 'Australia': 110,
                                           'Belarus': 111, 'Slovakia': 112, 'Ethiopia': 113, 'Norway': 114, 'Kuwait': 115,
                                           'Togo': 116, 'Albania': 117, 'Malawi': 118, 'Bhutan': 119, 'Sierra Leone': 120,
                                           'Uzbekistan': 121, 'Zimbabwe': 122, 'Guatemala': 123, 'Fiji': 124, 'Oman': 125,
                                           'Switzerland': 126, 'Rwanda': 127, 'Austria': 128, 'Cambodia': 129,
                                           'South Korea': 130, 'Turkmenistan': 131, 'Lithuania': 132, 'Jamaica': 133,
                                           'Taiwan': 134, 'Liechtenstein': 135, 'Sudan': 136, 'Syrian Arab Republic': 137,
                                           'Honduras': 138, 'Qatar': 139, 'Madagascar': 140, 'Thailand': 141,
                                           "Côte d'Ivoire": 142, 'Belgium': 143, 'Namibia': 144, 'Panama': 145, 'Mozambique': 146,
                                           'Israel': 147, 'Guyana': 148, 'Georgia': 149, 'Germany': 150, 'Pakistan': 151, 'Burundi': 152}).astype(int, errors='ignore')
        df['ConvertedCompYearly'] = df['ConvertedCompYearly'].apply(lambda x : 1 if x > 10000 else 0)
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('ConvertedCompYearly'))
        df = df[cols+['ConvertedCompYearly']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'ConvertedCompYearly' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['ConvertedCompYearly'] = df['ConvertedCompYearly'].apply(lambda x : 1 if x > 10000 else 0).astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('ConvertedCompYearly'))
        df = df[cols+['ConvertedCompYearly']]
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
        df['Region'] = df['Region'].map({'West': 0, 'Midwest': 1, 'South': 2, 'Northeast': 3}).astype(int, errors='ignore')
        df['Race'] = df['Race'].map({'White': 0, 'Black': 1, 'MultipleRaces': 2, ' Indian/Alaska': 3, 'Asian/Hawaiian/PacificIls': 4}).astype(int, errors='ignore')
        df['Education'] = df['Education'].map({'Bachelor': 0, 'Other': 1, 'UnAcceptable': 2, 'High school': 3, 'No degree': 4, 'Master': 5, 'Doctorate': 6, 'GED': 7}).astype(int, errors='ignore')
        df['IsWorking'] = df['IsWorking'].map({'UnAcceptable': 0, 'WorkedAtSomePointDuringTheYear': 1, 'NotCurrentlyWorkingButHasAJob': 2, 'ActivelyWorking': 3, 'NotEmployed': 4}).astype(int, errors='ignore')
        df['IsDiagnosedDiabetes'] = df['IsDiagnosedDiabetes'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('IsDiagnosedDiabetes'))
        df = df[cols+['IsDiagnosedDiabetes']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'IsDiagnosedDiabetes' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['IsDiagnosedDiabetes'] = df['IsDiagnosedDiabetes'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('IsDiagnosedDiabetes'))
        df = df[cols+['IsDiagnosedDiabetes']]
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
        df['School enrollment'] = df['School enrollment'].map({'No, has not attended in the last 3 months': 0, 'Yes, private school, private college, or home school': 1,
                                                       'Yes, public school or public college': 2}).astype(int, errors='ignore')
        df['Cognitive difficulty'] = df['Cognitive difficulty'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        df['Language other than English spoken at home'] = df['Language other than English spoken at home'].map({'no': 0, "yes": 1}).astype(int, errors='ignore')
        df['Citizenship status'] = df['Citizenship status'].map({'Born abroad of American parent(s)': 0, 'Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas': 1,
                                                               'Born in the U.S.': 2, 'Not a citizen of the U.S.': 3,
                                                                 'U.S. citizen by naturalization': 4}).astype(int, errors='ignore')
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


def get_score(group, alpha, d, max_subpopulation):
    intersection = 0
    g = []
    for x in group.iterrows():
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


def parse_subpoplation(df, sub_str):
    sub_str = ast.literal_eval(sub_str)
    for s in sub_str:
        k, v = s.split("=")
        try:
            v = int(v)
        except:
            v = v
        df = df.loc[df[k]==v]
    group1 = df.loc[df['group1']==1]
    group2 = df.loc[df['group2']==1]
    return group1, group2


def baseline(d: Dataset):
    df = pd.read_csv(d.clean_path)
    with open(f"data/{d.name}/causal_dag.txt", "r") as f:
        lines = f.readlines()
    treatments, _ = findBestTreatment("", d, lines, 0.5)
    df['TempTreatment'] = df.apply(lambda row: int(all(row[attr] == val for attr, val in treatments)), axis=1)
    if len(treatments) > 1:
        dag = changeDAG(lines, [x[0] for x in treatments])
    else:
        treatment_att_file_name = list(treatments)[0][0]
        if treatment_att_file_name.startswith('DevType'):
            treatment_att_file_name = 'DevType'
        with open(f"data/{d.name}/causal_dags_files/{treatment_att_file_name}.pkl", 'rb') as file:
            dag = pickle.load(file)
    """df = df.dropna()
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

    fairnessDebug = FairnessDebuggingUsingMachineUnlearning(myDataset,
                                                            ["TempTreatment", 1, 0],
                                                            d.outcome_col,
                                                            FairnessMetric.CATE,
                                                            dag,
                                                            group1_idx,
                                                            group2_idx
                                                            )

    bias_inducing_subsets = fairnessDebug.latticeSearchSubsets(2, (0.05, 1), "normal", False)
    print(bias_inducing_subsets)
    bias_inducing_subsets.sort_values(by=["Parity_Reduction","Size"], ascending=[False, False], ignore_index=True)\
        .to_csv(f"outputs/{d.name}/baselines/facts_rf.csv", index=False)"""
    df_facts = pd.read_csv(f"outputs/{d.name}/baselines/facts_rf.csv")
    res = []
    for _, row in df_facts.iterrows():
        sub = row["Subset"]
        group1, group2 = parse_subpoplation(df, sub)
        result = CalcIScore(treatments, d, lines, len(treatments), group1, group2, 1)
        if result:
            size_group1 = group1.shape[0]
            size_group2 = group2.shape[0]
            diff_means = np.mean(group1[d.outcome_col]) - np.mean(group2[d.outcome_col])
            res.append({'subpopulation': sub, 'treatment': treatments, 'cate1': result[1], 'cate2': result[2],
                        'iscore': result[0], 'size_group1': size_group1, "size_group2": size_group2,
                        "diff_means": diff_means, "avg_group1": np.mean(group1[d.outcome_col]),
                        "avg_group2": np.mean(group2[d.outcome_col])})
    df = pd.DataFrame(res)
    lamda = choose_lamda(list(df['iscore']))
    df['ni_score'] = df['iscore'].apply(lambda x: ni_score(x, lamda))
    facts = pd.read_csv(f"outputs/{d.name}/all_facts.csv")
    max_subpopulation = max(facts["size_subpopulation"])
    scores = get_score(group=df.head(K), alpha=ALPHA, d=d, max_subpopulation=max_subpopulation)
    df.head(K).to_csv(f'outputs/{d.name}/baselines/facts_final_rf.csv', index=False)
    pd.DataFrame([scores]).to_csv(f'outputs/{d.name}/baselines/rf_scores.csv')


"""so = Dataset(name="so", outcome_col="ConvertedCompYearly",
             treatments=['YearsCodingProf', 'Hobby', 'LastNewJob', 'Student', 'WakeTime', 'DevType'],
             subpopulations=['Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                             'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                             'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                             'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'],
             columns_to_ignore=['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                                'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                                'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                                'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0'],
             clean_path="outputs/so/clean_data.csv", func_filter_facts=so_filter_facts, need_filter_subpopulations=True)


meps = Dataset(name="meps", outcome_col="IsDiagnosedDiabetes",
               treatments=['DoesDoctorRecommendExercise', 'TakesAspirinFrequently', 'BMI', 'Exercise',
                           'CurrentlySmoke'],
               subpopulations=['MaritalStatus', 'Region', 'Race', 'Education',
                               'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],
               columns_to_ignore=['Education=UnAcceptable', 'IsWorking=UnAcceptable'], clean_path="outputs/meps/clean_data.csv",
               func_filter_facts=meps_filter_facts, need_filter_subpopulations=True)"""
# , 'Cognitive difficulty',
#                               'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
#                               'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'
acs = Dataset(name="acs", outcome_col="Health insurance coverage recode",
              treatments=['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
                          'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment'],
              subpopulations=['Sex', 'Age', 'With a disability', 'School enrollment', 'Cognitive difficulty', 'Region',
              'Language other than English spoken at home', 'Citizenship status'],
              columns_to_ignore=[], clean_path="outputs/acs/sample_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

#baseline(so) # p value 0.4
#baseline(meps)
baseline(acs)
