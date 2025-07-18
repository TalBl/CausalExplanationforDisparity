import math
import time
import copy
import numpy as np
import pandas as pd
import ast
import pickle
from tqdm import tqdm
import networkx as nx

import csv
from sklearn.model_selection import train_test_split
from algorithms.final_algorithm.new_greedy import get_intersection, ni_score, get_union, print_matrix
from algorithms.final_algorithm.find_treatment_new import findBestTreatment, get_subpopulation
from algorithms.debug_RF.DebugRF import Dataset_RF, FairnessMetric, FairnessDebuggingUsingMachineUnlearning
# from algorithms.final_algorithm.find_treatment_new import changeDAG, getTreatmentATE
from algorithms.final_algorithm.find_best_treatment import process_subpopulation
from cleaning_datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats
from cleaning_datasets.clean_so import filter_facts as so_filter_facts
from cleaning_datasets.clean_meps import filter_facts as meps_filter_facts
from Utils import Dataset, getTreatmentATE, get_indices
import itertools

THRESHOLD_SUPPORT = 0.01
ALPHA = 0.5
K = 5
THRESHOLD = 0.55


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
        df['Gender'] = df['Gender'].apply(lambda x : 1 if x == "Gender=Male" else 0).astype(int, errors='ignore')
        df['Age'] = df['Age'].map({'Age=Under 18 years old': 0,'Age=18 - 24 years old': 1, 'Age=25 - 34 years old': 2,
                                   'Age=35 - 44 years old': 3, 'Age=45 - 54 years old': 4, 'Age=55 - 64 years old': 5,
                                   'Age=65 years or older': 6}).astype(int, errors='ignore')
        df['Country'] = df['Country'].map({'Country=Philippines': 1, 'Country=Latvia': 2, 'Country=Nicaragua': 3, 'Country=Luxembourg': 4, 'Country=Honduras': 5, 'Country=Marshall Islands': 6, 'Country=Israel': 7, 'Country=Australia': 8, 'Country=Austria': 9, 'Country=Cambodia': 10, 'Country=Uzbekistan': 11, 'Country=Gambia': 12, 'Country=Indonesia': 13, 'Country=Afghanistan': 14, 'Country=Kazakhstan': 15, 'Country=Russian Federation': 16, 'Country=Jordan': 17, 'Country=Swaziland': 18, 'Country=Bulgaria': 19, 'Country=Lebanon': 20, 'Country=Monaco': 21, 'Country=France': 22, 'Country=Poland': 23, 'Country=Turkmenistan': 24, 'Country=Mexico': 25, 'Country=Congo, Republic of the...': 26, 'Country=Estonia': 27, 'Country=United Republic of Tanzania': 28, 'Country=Cameroon': 29, 'Country=Panama': 30, 'Country=Morocco': 31, 'Country=Mongolia': 32, 'Country=Trinidad and Tobago': 33, 'Country=Slovenia': 34, 'Country=Pakistan': 35, 'Country=United Arab Emirates': 36, 'Country=Syrian Arab Republic': 37, 'Country=Tajikistan': 38, 'Country=Ukraine': 39, 'Country=South Korea': 40, 'Country=Colombia': 41, 'Country=Viet Nam': 42, 'Country=Uganda': 43, 'Country=Bahrain': 44, 'Country=Eritrea': 45, 'Country=Kenya': 46, 'Country=Nepal': 47, 'Country=Malawi': 48, 'Country=Guyana': 49, 'Country=Belgium': 50, 'Country=Bhutan': 51, 'Country=Madagascar': 52, 'Country=Serbia': 53, 'Country=Namibia': 54, 'Country=Lesotho': 55, 'Country=Malaysia': 56, 'Country=Bosnia and Herzegovina': 57, 'Country=Spain': 58, 'Country=Saudi Arabia': 59, 'Country=Kuwait': 60, 'Country=Canada': 61, 'Country=Dominican Republic': 62, 'Country=Senegal': 63, 'Country=Argentina': 64, 'Country=Benin': 65, 'Country=Costa Rica': 66, 'Country=Mauritius': 67, 'Country=Somalia': 68, 'Country=Portugal': 69, 'Country=Germany': 70, 'Country=China': 71, 'Country=Malta': 72, 'Country=Montenegro': 73, 'Country=Croatia': 74, 'Country=Iraq': 75, 'Country=Mozambique': 76, 'Country=Andorra': 77, 'Country=Iran, Islamic Republic of...': 78, 'Country=Nigeria': 79, 'Country=Norway': 80, 'Country=Fiji': 81, 'Country=Singapore': 82, 'Country=United Kingdom': 83, 'Country=Slovakia': 84, 'Country=Ghana': 85, 'Country=Iceland': 86, 'Country=Other Country (Not Listed Above)': 87, 'Country=Libyan Arab Jamahiriya': 88, 'Country=Albania': 89, 'Country=Algeria': 90, 'Country=Myanmar': 91, 'Country=Oman': 92, "Country=Côte d'Ivoire": 93, 'Country=Saint Lucia': 94, 'Country=Liechtenstein': 95, 'Country=Bolivia': 96, 'Country=Peru': 97, 'Country=Democratic Republic of the Congo': 98, 'Country=Rwanda': 99, 'Country=Suriname': 100, 'Country=Dominica': 101, 'Country=Ireland': 102, 'Country=Yemen': 103, 'Country=Jamaica': 104, 'Country=Bangladesh': 105, 'Country=Zambia': 106, 'Country=Azerbaijan': 107, 'Country=Sweden': 108, 'Country=Greece': 109, 'Country=Paraguay': 110, 'Country=Tunisia': 111, 'Country=Sierra Leone': 112, 'Country=Ecuador': 113, 'Country=Thailand': 114, 'Country=Cuba': 115, 'Country=Cyprus': 116, 'Country=Sri Lanka': 117, 'Country=Denmark': 118, 'Country=Egypt': 119, 'Country=Switzerland': 120, 'Country=Turkey': 121, 'Country=Italy': 122, 'Country=Ethiopia': 123, 'Country=Japan': 124, 'Country=Romania': 125, 'Country=India': 126, 'Country=New Zealand': 127, 'Country=Hungary': 128, 'Country=Armenia': 129, 'Country=Lithuania': 130, 'Country=Netherlands': 131, 'Country=Botswana': 132, 'Country=South Africa': 133, 'Country=Hong Kong (S.A.R.)': 134, 'Country=Qatar': 135, 'Country=Bahamas': 136, 'Country=Barbados': 137, 'Country=Uruguay': 138, 'Country=El Salvador': 139, 'Country=Togo': 140, 'Country=Czech Republic': 141, 'Country=Brazil': 142, 'Country=Zimbabwe': 143, 'Country=Belarus': 144, 'Country=Sudan': 145, 'Country=Republic of Korea': 146, 'Country=Maldives': 147, 'Country=Chile': 148, 'Country=Taiwan': 149, 'Country=Finland': 150, 'Country=Guatemala': 151, 'Country=Republic of Moldova': 152, 'Country=Georgia': 153, 'Country=Venezuela, Bolivarian Republic of...': 154, 'Country=United States': 155, 'Country=The former Yugoslav Republic of Macedonia': 156, 'Country=Kyrgyzstan': 157}).astype(int, errors='ignore')
        df['SexualOrientation'] = df['SexualOrientation'].map({'SexualOrientation=Straight or heterosexual;Gay or Lesbian;Bisexual or Queer;Asexual': 1, 'SexualOrientation=Asexual': 2, 'SexualOrientation=Straight or heterosexual;Bisexual or Queer': 3, 'SexualOrientation=Straight or heterosexual;Bisexual or Queer;Asexual': 4, 'SexualOrientation=Gay or Lesbian;Asexual': 5, 'SexualOrientation=Gay or Lesbian;Bisexual or Queer;Asexual': 6, 'SexualOrientation=Bisexual or Queer': 7, 'SexualOrientation=Bisexual or Queer;Asexual': 8, 'SexualOrientation=Straight or heterosexual;Asexual': 9, 'SexualOrientation=Straight or heterosexual;Gay or Lesbian': 10, 'SexualOrientation=Gay or Lesbian;Bisexual or Queer': 11, 'SexualOrientation=Gay or Lesbian': 12, 'SexualOrientation=Straight or heterosexual;Gay or Lesbian;Bisexual or Queer': 13, 'SexualOrientation=Straight or heterosexual': 14}).astype(int, errors='ignore')
        df['EducationParents'] = df['EducationParents'].map({'EducationParents=Associate degree': 1, 'EducationParents=They never completed any formal education': 2, 'EducationParents=Master’s degree (MA, MS, M.Eng., MBA, etc.)': 3, 'EducationParents=Some college/university study without earning a degree': 4, 'EducationParents=Professional degree (JD, MD, etc.)': 5, 'EducationParents=Bachelor’s degree (BA, BS, B.Eng., etc.)': 6, 'EducationParents=Primary/elementary school': 7, 'EducationParents=Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 8, 'EducationParents=Other doctoral degree (Ph.D, Ed.D., etc.)': 9})
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
        df['MaritalStatus'] = df['MaritalStatus'].map({'MaritalStatus=Married': 0, 'MaritalStatus=Divorced': 1, 'MaritalStatus=Widowed': 2, 'MaritalStatus=Separated': 3,
                                                       'MaritalStatus=NeverMarried': 4}).astype(int, errors='ignore')
        df['Region'] = df['Region'].map({'Region=West': 0, 'Region=Midwest': 1, 'Region=South': 2, 'Region=Northeast': 3, "Region=-1": -1}).astype(int, errors='ignore')
        df['Race'] = df['Race'].map({'Race=White': 0, 'Race=Black': 1, 'Race=MultipleRaces': 2, 'Race=Indian/Alaska': 3, 'Race=Asian/Hawaiian/PacificIls': 4}).astype(int, errors='ignore')
        df['Age'] = df['Age'].map({"Age=<30.00": 0, "Age=30.00-42.00": 1, "Age=42.00-54.00": 2, "Age=54.00-66.00": 3, "Age=>66.00": 4}).astype(int, errors='ignore')
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
        df['Sex'] = df['Sex'].map({'Sex=Man': 0, "Sex=Woman": 1}).astype(int, errors='ignore')
        df['Age'] = df['Age'].map({'Age=<17.00': 0, "Age=17.00-34.00": 1, "Age=34.00-50.00": 2, "Age=50.00-64.00": 3, "Age=>64.00": 4}).astype(int, errors='ignore')
        df['WithADisability'] = df['WithADisability'].map({'WithADisability=no': 0, "WithADisability=yes": 1}).astype(int, errors='ignore')
        df['RaceEthnicity'] = df['RaceEthnicity'].map(
            {'RaceEthnicity=Black alone': 0, "RaceEthnicity=Am Indian and Alaskan Native tribes specified": 1, "RaceEthnicity=Asian alone": 2, "RaceEthnicity=American Indian alone": 3, "RaceEthnicity=White alone": 4,
             'RaceEthnicity=Alaskan Native alone': 5, "RaceEthnicity=2+ major race groups": 6, "RaceEthnicity=Native Hawaiian and other PI alone": 7, "RaceEthnicity=Some other race alone": 8}).astype(int, errors='ignore')
        df['LanguageOtherThanEnglishSpokenAtHome'] = df['LanguageOtherThanEnglishSpokenAtHome'].map({'LanguageOtherThanEnglishSpokenAtHome=Telugu': 1, 'LanguageOtherThanEnglishSpokenAtHome=Apache languages': 2, 'LanguageOtherThanEnglishSpokenAtHome=Portuguese': 3, 'LanguageOtherThanEnglishSpokenAtHome=Hmong': 4, 'LanguageOtherThanEnglishSpokenAtHome=India N.E.C.': 5, 'LanguageOtherThanEnglishSpokenAtHome=Jamaican Creole English': 6, 'LanguageOtherThanEnglishSpokenAtHome=Min Nan Chinese': 7, 'LanguageOtherThanEnglishSpokenAtHome=Hindi': 8, 'LanguageOtherThanEnglishSpokenAtHome=Gbe languages': 9, 'LanguageOtherThanEnglishSpokenAtHome=Indonesian': 10, 'LanguageOtherThanEnglishSpokenAtHome=Spanish': 11, 'LanguageOtherThanEnglishSpokenAtHome=Burmese': 12, 'LanguageOtherThanEnglishSpokenAtHome=German': 13, 'LanguageOtherThanEnglishSpokenAtHome=Arabic': 14, 'LanguageOtherThanEnglishSpokenAtHome=Other Bantu languages': 15, 'LanguageOtherThanEnglishSpokenAtHome=Other languages of Africa': 16, 'LanguageOtherThanEnglishSpokenAtHome=Bosnian': 17, 'LanguageOtherThanEnglishSpokenAtHome=Swahili': 18, 'LanguageOtherThanEnglishSpokenAtHome=Mandarin': 19, 'LanguageOtherThanEnglishSpokenAtHome=Kannada': 20, 'LanguageOtherThanEnglishSpokenAtHome=Dutch': 21, 'LanguageOtherThanEnglishSpokenAtHome=Latvian': 22, 'LanguageOtherThanEnglishSpokenAtHome=Cajun French': 23, 'LanguageOtherThanEnglishSpokenAtHome=Igbo': 24, 'LanguageOtherThanEnglishSpokenAtHome=Other Mande languages': 25, 'LanguageOtherThanEnglishSpokenAtHome=Iu Mien': 26, 'LanguageOtherThanEnglishSpokenAtHome=Other Afro-Asiatic languages': 27, 'LanguageOtherThanEnglishSpokenAtHome=Nilo-Saharan languages': 28, 'LanguageOtherThanEnglishSpokenAtHome=Filipino': 29, 'LanguageOtherThanEnglishSpokenAtHome=Chaldean Neo-Aramaic': 30, 'LanguageOtherThanEnglishSpokenAtHome=Other Central and South American languages': 31, 'LanguageOtherThanEnglishSpokenAtHome=Ojibwa': 32, 'LanguageOtherThanEnglishSpokenAtHome=Greek': 33, 'LanguageOtherThanEnglishSpokenAtHome=Hebrew': 34, 'LanguageOtherThanEnglishSpokenAtHome=Aleut languages': 35, 'LanguageOtherThanEnglishSpokenAtHome=Albanian': 36, 'LanguageOtherThanEnglishSpokenAtHome=Bulgarian': 37, 'LanguageOtherThanEnglishSpokenAtHome=Japanese': 38, 'LanguageOtherThanEnglishSpokenAtHome=Tibetan': 39, 'LanguageOtherThanEnglishSpokenAtHome=Cantonese': 40, 'LanguageOtherThanEnglishSpokenAtHome=Finnish': 41, 'LanguageOtherThanEnglishSpokenAtHome=Hungarian': 42, 'LanguageOtherThanEnglishSpokenAtHome=Pashto': 43, 'LanguageOtherThanEnglishSpokenAtHome=Danish': 44, 'LanguageOtherThanEnglishSpokenAtHome=Samoan': 45, 'LanguageOtherThanEnglishSpokenAtHome=Other Native North American languages': 46, 'LanguageOtherThanEnglishSpokenAtHome=Thai': 47, 'LanguageOtherThanEnglishSpokenAtHome=Oromo': 48, 'LanguageOtherThanEnglishSpokenAtHome=Tigrinya': 49, 'LanguageOtherThanEnglishSpokenAtHome=Dari': 50, 'LanguageOtherThanEnglishSpokenAtHome=Punjabi': 51, 'LanguageOtherThanEnglishSpokenAtHome=Wolof': 52, 'LanguageOtherThanEnglishSpokenAtHome=Khmer': 53, 'LanguageOtherThanEnglishSpokenAtHome=Other Niger-Congo languages': 54, 'LanguageOtherThanEnglishSpokenAtHome=Other Philippine languages': 55, 'LanguageOtherThanEnglishSpokenAtHome=Tamil': 56, 'LanguageOtherThanEnglishSpokenAtHome=Karen languages': 57, 'LanguageOtherThanEnglishSpokenAtHome=Manding languages': 58, 'LanguageOtherThanEnglishSpokenAtHome=Malay': 59, 'LanguageOtherThanEnglishSpokenAtHome=Afrikaans': 60, 'LanguageOtherThanEnglishSpokenAtHome=Marathi': 61, 'LanguageOtherThanEnglishSpokenAtHome=Tagalog': 62, 'LanguageOtherThanEnglishSpokenAtHome=Other Indo-European languages': 63, 'LanguageOtherThanEnglishSpokenAtHome=Other English-based Creole languages': 64, 'LanguageOtherThanEnglishSpokenAtHome=Swedish': 65, 'LanguageOtherThanEnglishSpokenAtHome=Hawaiian': 66, 'LanguageOtherThanEnglishSpokenAtHome=Ga': 67, 'LanguageOtherThanEnglishSpokenAtHome=Other and unspecified languages': 68, 'LanguageOtherThanEnglishSpokenAtHome=Other Eastern Malayo-Polynesian languages': 69, 'LanguageOtherThanEnglishSpokenAtHome=Mongolian': 70, 'LanguageOtherThanEnglishSpokenAtHome=Ilocano': 71, 'LanguageOtherThanEnglishSpokenAtHome=Slovak': 72, 'LanguageOtherThanEnglishSpokenAtHome=Muskogean languages': 73, 'LanguageOtherThanEnglishSpokenAtHome=Yoruba': 74, 'LanguageOtherThanEnglishSpokenAtHome=Chamorro': 75, 'LanguageOtherThanEnglishSpokenAtHome=Tongan': 76, 'LanguageOtherThanEnglishSpokenAtHome=Ganda': 77, 'LanguageOtherThanEnglishSpokenAtHome=Swiss German': 78, 'LanguageOtherThanEnglishSpokenAtHome=Norwegian': 79, 'LanguageOtherThanEnglishSpokenAtHome=Uto-Aztecan languages': 80, 'LanguageOtherThanEnglishSpokenAtHome=Armenian': 81, 'LanguageOtherThanEnglishSpokenAtHome=Other Indo-Iranian languages': 82, 'LanguageOtherThanEnglishSpokenAtHome=Other languages of Asia': 83, 'LanguageOtherThanEnglishSpokenAtHome=Fulah': 84, 'LanguageOtherThanEnglishSpokenAtHome=Serbian': 85, 'LanguageOtherThanEnglishSpokenAtHome=Kabuverdianu': 86, 'LanguageOtherThanEnglishSpokenAtHome=Chuukese': 87, 'LanguageOtherThanEnglishSpokenAtHome=Haitian': 88, 'LanguageOtherThanEnglishSpokenAtHome=Nepali': 89, 'LanguageOtherThanEnglishSpokenAtHome=Pennsylvania German': 90, 'LanguageOtherThanEnglishSpokenAtHome=Yiddish': 91, 'LanguageOtherThanEnglishSpokenAtHome=Serbocroatian': 92, 'LanguageOtherThanEnglishSpokenAtHome=Assyrian Neo-Aramaic': 93, 'LanguageOtherThanEnglishSpokenAtHome=Dakota languages': 94, 'LanguageOtherThanEnglishSpokenAtHome=Navajo': 95, 'LanguageOtherThanEnglishSpokenAtHome=Malayalam': 96, 'LanguageOtherThanEnglishSpokenAtHome=Kurdish': 97, 'LanguageOtherThanEnglishSpokenAtHome=Cebuano': 98, 'LanguageOtherThanEnglishSpokenAtHome=Marshallese': 99, 'LanguageOtherThanEnglishSpokenAtHome=Edoid languages': 100, 'LanguageOtherThanEnglishSpokenAtHome=French': 101, 'LanguageOtherThanEnglishSpokenAtHome=Russian': 102, 'LanguageOtherThanEnglishSpokenAtHome=Bengali': 103, 'LanguageOtherThanEnglishSpokenAtHome=Polish': 104, 'LanguageOtherThanEnglishSpokenAtHome=Turkish': 105, 'LanguageOtherThanEnglishSpokenAtHome=Farsi': 106, 'LanguageOtherThanEnglishSpokenAtHome=Irish': 107, 'LanguageOtherThanEnglishSpokenAtHome=Lao': 108, 'LanguageOtherThanEnglishSpokenAtHome=Gujarati': 109, 'LanguageOtherThanEnglishSpokenAtHome=Somali': 110, 'LanguageOtherThanEnglishSpokenAtHome=Akan (incl. Twi)': 111, 'LanguageOtherThanEnglishSpokenAtHome=Amharic': 112, 'LanguageOtherThanEnglishSpokenAtHome=Konkani': 113, 'LanguageOtherThanEnglishSpokenAtHome=Croatian': 114, 'LanguageOtherThanEnglishSpokenAtHome=Vietnamese': 115, 'LanguageOtherThanEnglishSpokenAtHome=Lithuanian': 116, 'LanguageOtherThanEnglishSpokenAtHome=Urdu': 117, 'LanguageOtherThanEnglishSpokenAtHome=Ukrainian': 118, 'LanguageOtherThanEnglishSpokenAtHome=Cherokee': 119, 'LanguageOtherThanEnglishSpokenAtHome=Shona': 120, 'LanguageOtherThanEnglishSpokenAtHome=Sinhala': 121, 'LanguageOtherThanEnglishSpokenAtHome=Macedonian': 122, 'LanguageOtherThanEnglishSpokenAtHome=Pakistan N.E.C.': 123, 'LanguageOtherThanEnglishSpokenAtHome=Chin languages': 124, 'LanguageOtherThanEnglishSpokenAtHome=Korean': 125, 'LanguageOtherThanEnglishSpokenAtHome=Czech': 126, 'LanguageOtherThanEnglishSpokenAtHome=Chinese': 127, 'LanguageOtherThanEnglishSpokenAtHome=Romanian': 128, 'LanguageOtherThanEnglishSpokenAtHome=Italian': 129})
        df['StateCode'] = df['StateCode'].map({'StateCode=Illinois': 1, 'StateCode=Florida': 2, 'StateCode=California': 3, 'StateCode=Ohio': 4, 'StateCode=Texas': 5, 'StateCode=Pennsylvania': 6, 'StateCode=New York': 7}).astype(int, errors='ignore')
        df['MaritalStatus'] = df['MaritalStatus'].map({'MaritalStatus=Divorced': 0, 'MaritalStatus=Married': 1, 'MaritalStatus=Never Married': 2, 'MaritalStatus=Separated': 3, 'MaritalStatus=Widowed': 4}).astype(int, errors='ignore')
        df['Nativity'] = df['Nativity'].map({'Nativity=no': 0, "Nativity=yes": 1}).astype(int, errors='ignore')
        df['RelatedChild'] = df['RelatedChild'].map({'RelatedChild=no': 0, "RelatedChild=yes": 1}).astype(int, errors='ignore')
        df['CitizenshipStatus'] = df['CitizenshipStatus'].map({'CitizenshipStatus=Born abroad of American parent(s)': 0,
                                                                 'CitizenshipStatus=Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas': 1,
                                                                 'CitizenshipStatus=Born in the U.S.': 2, 'CitizenshipStatus=Not a citizen of the U.S.': 3,
                                                                 'CitizenshipStatus=U.S. citizen by naturalization': 4}).astype(int, errors='ignore')
        df['Region'] = df['Region'].map({'Region=West': 0, 'Region=Midwest': 1, 'Region=South': 2, 'Region=Northeast': 3}).astype(int, errors='ignore')
        df['HealthInsuranceCoverageRecode'] = df['HealthInsuranceCoverageRecode'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('HealthInsuranceCoverageRecode'))
        df = df[cols+['HealthInsuranceCoverageRecode']]
        return df

    def __preprocessDatasetForCategorization(self, dataset):
        df = copy.deepcopy(dataset)
        non_object_columns = [col for col in df.columns if df[col].dtypes != 'object']
        quantiles = self.train[non_object_columns].quantile([0, .25, .5, .75, 1.0], axis = 0)
        for col in non_object_columns:
            if col == 'HealthInsuranceCoverageRecode' or col == "TempTreatment":
                continue
            else:
                df[col] = pd.cut(df[col],
                                 [quantiles[col][0.0] - 1, 0.5, math.inf],
                                 labels=[str(col) + ' = low', str(col) + ' = high'],
                                 right=True,
                                 include_lowest=True)
        df['HealthInsuranceCoverageRecode'] = df['HealthInsuranceCoverageRecode'].astype(int, errors='ignore')
        '''Moving outcome column at the end'''
        cols = list(df.columns.values)
        cols.pop(cols.index('HealthInsuranceCoverageRecode'))
        df = df[cols+['HealthInsuranceCoverageRecode']]
        return df

    def __decodeAttributeCodeToRealValues(self, dataset):
        """df = copy.deepcopy(dataset)
        map_code_to_real = {
        }
        object_columns = [col for col in df.columns if df[col].dtypes == 'object']
        for col in object_columns:
            df[col] = df[col].map(map_code_to_real[col]).fillna(df[col])"""
        return dataset


def get_score(group, d, calc_intersection, calc_union, max_outcome):
    g = []
    iscore = 0
    for _, row in group:
        g.append(row)
        iscore += row['iscore']
    for row1, row2 in itertools.combinations(g, 2):
        intersection = get_intersection(row1, row2, d, calc_intersection)
        union = get_union(row1, row2, d, calc_union)
        jaccard = intersection / union
        if jaccard > THRESHOLD:
            return {"score": 0}
    return {"score": iscore}


def run_search(d, k, df_treatments, calc_intersection, calc_union, max_outcome):
    max_score = 0
    for group in tqdm(itertools.combinations(df_treatments.iterrows(), k)):
        scores = get_score(group=group, d=d, calc_intersection=calc_intersection, calc_union=calc_union, max_outcome=max_outcome)
        if scores["score"] > max_score:
            max_score = scores["score"]
            res_group = group
            scores_dict = scores
    if max_score > 0:
        return max_score, res_group, scores_dict
    else:
        return 0, [], {}


def parse_subpoplation(df_original, sub_str, outcome_col):
    sub_str = ast.literal_eval(sub_str)
    df = df_original.copy()
    parsed_subs = {}
    for s in sub_str:
        if " = low" in s:
            s = s.replace(" = low", "=0")
        if " = high" in s:
            s = s.replace(" = high", "=1")
        try:
            k, v = s.split("=")
        except:
            print("here")
        try:
            v = int(v)
        except:
            v = v
        df = df.loc[df[k]==v]
        parsed_subs[k]=v
    group1 = df.loc[df['group1']==1]
    group2 = df.loc[df['group2']==1]
    belong_groups = df.loc[(df['group1'] == 1) | (df['group2'] == 1)]
    support = belong_groups.shape[0] / df_original.shape[0]
    return group1, group2, df.shape[0], support, set(df.index), np.mean(group1[outcome_col]), np.mean(group2[outcome_col]),parsed_subs

def naive_filter(n1, n2):
    return True

def baseline(d: Dataset):
    df = pd.read_csv(d.clean_path)
    max_outcome = max(df[d.outcome_col])
    treatments = process_subpopulation(pd.Series(), d, nx.DiGraph(nx.nx_pydot.read_dot(d.dag_file)), max_outcome)['treatment_combo']
    print(treatments)
    if d.name == "acs":
        d.clean_path = d.clean_path.replace("clean", "sample")
    df = pd.read_csv(d.clean_path)
    df['TempTreatment'] = df.apply(lambda row: int(all(row[attr] == val for attr, val in treatments)), axis=1)
    treatment_att_file_name = list(treatments)[0][0]
    dag = f"data/{d.name}/causal_dags_files/graph_{treatment_att_file_name}.dot"
    df = df.dropna(subset=d.subpopulations_atts)
    y = df[d.outcome_col]
    X = df.drop(d.outcome_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train[d.outcome_col] = y_train
    X_test[d.outcome_col] = y_test
    d.subpopulations_atts.extend(["TempTreatment", d.outcome_col])
    X_train = X_train[d.subpopulations_atts]
    cats_names= X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train = X_train.apply(lambda col: col.map(lambda val: f"{col.name}={val}") if col.name in cats_names else col)
    X_train.to_csv(f"outputs/{d.name}/train.csv", index=False)
    X_test = X_test.reset_index()
    group1_idx = X_test.loc[X_test['group1']==1.0].index.tolist()
    group2_idx = X_test.loc[X_test['group2']==1.0].index.tolist()
    X_test = X_test[d.subpopulations_atts]
    X_test = X_test.apply(lambda col: col.map(lambda val: f"{col.name}={val}") if col.name in cats_names else col)
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

    bias_inducing_subsets = fairnessDebug.latticeSearchSubsets(4, (0.05, 1), "normal", False)
    print(bias_inducing_subsets)
    bias_inducing_subsets.sort_values(by=["Parity_Reduction","Size"], ascending=[False, False], ignore_index=True)\
        .to_csv(f"outputs/{d.name}/baselines/facts_rf.csv", index=False)
    df_facts = pd.read_csv(f"outputs/{d.name}/baselines/facts_rf.csv")
    data = pd.read_csv(d.clean_path)
    res = []
    for _, row in df_facts.iterrows():
        sub = row["Subset"]
        group1, group2, size, support, idxs, avg1, avg2, parsed_subs = parse_subpoplation(data, sub, d.outcome_col)
        if not d.func_filter_subs(avg1, avg2):
            continue
        res1 = getTreatmentATE(group1, dag, treatments, d.outcome_col)
        if not res1:
            continue
        ate1, _ = res1
        res2 = getTreatmentATE(group2, dag, treatments, d.outcome_col)
        if not res2:
            continue
        ate2, _ = res2
        if not d.func_filter_treats(ate1, ate2):
            continue
        result = abs(ate1 - ate2) / max_outcome
        if result:
            res.append({'subpop': str(parsed_subs), 'treatment': treatments, 'ate1': ate1, 'ate2': ate2,
                        'iscore': result,
                        "avg_group1": avg1, "avg_group2": avg2, "size": size, "support": support,
                        'indices': idxs})
    df = pd.DataFrame(res, columns=["subpop", "treatment", "ate1", "ate2", "iscore", "avg_group1", "avg_group2", "size", "support", "indices"])
    df.drop(columns=['indices']).to_csv(f"outputs/{d.name}/baselines/rf_subpopulations_and_treatments.csv", index=False)
    res_group = []
    k = K
    calc_intersection, calc_union, scores_dict = {}, {}, {}
    while res_group == [] and k > 0:
        max_score, res_group, scores_dict = run_search(d, k, df, calc_intersection, calc_union, max_outcome)
        k -= 1
    g = []
    for x in res_group:
        _, row = x
        g.append(row)
    df_results = pd.DataFrame(g, columns=["subpop", "treatment", "ate1", "ate2", "iscore", "avg_group1", "avg_group2", "size", "support", "indices"])
    jaccard_matrix = print_matrix({}, {}, [[x['subpop'], x['indices']] for _, x in df_results.iterrows()])
    jaccard_matrix.to_csv(f"outputs/{d.name}/baselines/rf_jaccard_matrix.csv", quoting=csv.QUOTE_NONNUMERIC)
    df_results = df_results.sort_values(by=['iscore'], ascending=False)
    df_results = df_results.drop(columns=['indices'])
    df_results.to_csv(f'outputs/{d.name}/baselines/facts_final_rf.csv', index=False)
    pd.DataFrame([scores_dict]).to_csv(f'outputs/{d.name}/baselines/rf_scores.csv', index=False)



from algorithms.final_algorithm.full import acs, so, meps
import time
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


"""
acs took 717.3514559268951
so took 2546.943838119507
meps took 103.76292014122009
"""
