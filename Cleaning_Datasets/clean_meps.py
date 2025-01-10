import pandas as pd
import numpy as np
INPUT_DF_PATH = "data/meps/h181.csv"

# Example mappings based on the MEPS codebook
column_mappings = {
    "SEX": {1: "Male", 2: "Female"},
    "RACEV1X": {1: "White", 2: "Black", 3: " Indian/Alaska", 4: "Asian/Hawaiian/PacificIls", 6: "MultipleRaces"},
    "BORNUSA": {1: 1, 2: 0},
    "HIDEG": {1: "No degree", 2: "GED", 3: "High school", 4: "Bachelor", 5: "Master", 6: "Doctorate", 7: "Other", 8: "Under18",
              -1: "UnAcceptable", -7: "UnAcceptable", -8: "UnAcceptable", -9: "UnAcceptable"},
    "REGION15": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    "MARRY15X": {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "NeverMarried"},
    "EXRCIS53": {1: 1, 2: 0},
    "PHYEXE53": {1: 1, 2: 0},
    "ADSMOK42": {1: 1, 2: 0},
    "ASPRIN53": {1: 1, 2: 0},
    "ASTHDX": {1: 1, 2: 0},
    "EMPST53": {1: "ActivelyWorking", 2: "NotCurrentlyWorkingButHasAJob", 3: "WorkedAtSomePointDuringTheYear", 4: "NotEmployed",
                -7: "UnAcceptable", -8: "UnAcceptable", -9: "UnAcceptable"},
    "DIABDX": {1: 1, 2: 0}
}


# Function to map categorical flags to their actual values
def map_flags_to_values(df, column_mappings):
    for col, mapping in column_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])  # Retain original if no mapping is provided
    return df



def build_mini_df():
    df = pd.read_csv(INPUT_DF_PATH)
    df = map_flags_to_values(df, column_mappings)
    df['group2'] = df['SEX'].apply(lambda x: 1 if x != "Male" else 0)
    df['group1'] = df['SEX'].apply(lambda x: 1 if x == "Male" else 0)
    df["BMI"] = df["BMINDX53"]
    # Subpopulations
    df["Age"] = df["AGELAST"]
    df["Race"] = df["RACEV1X"]
    df["IsBornInUSA"] = df["BORNUSA"]
    df["Education"] = df["HIDEG"]
    df['Region'] = df['REGION15']
    df["MaritalStatus"] = df["MARRY15X"]

    # Treatments """['DoesDoctorRecommendExercise', 'TakesAspirinFrequently', 'BMI', 'Exercise',
    #                            'CurrentlySmoke'],
    #                subpopulations=['Married', 'Region', 'Race',
    #                                'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking'],"
    df["Student"] = df["FTSTU15X"]
    df["DoesDoctorRecommendExercise"] = df["EXRCIS53"]
    df["HowOftenCheckInDoctor"] = df["MESVIS42"]
    df["HoldHealthInsurance"] = df["INSURC15"]
    df["Exercise"] = df["PHYEXE53"]
    df["FamilyIncomeWage"] = df["POVCAT15"]
    df["IsWorking"] = df["EMPST53"]
    df['CurrentlySmoke'] = df['ADSMOK42'].fillna(0).apply(lambda x: 1 if x and x == 1 else 0) # smokers
    df["IsADHD/ADD_Diagnisos"] = df["ADHDADDX"]
    df["FeltNervous"] = df["ADNERV42"]
    df["IsHadHeartAttack"] = df["MIDX"]
    df["IsHadStroke"] = df["STRKDX"]
    df["IsDiagnosedCancer"] = df["CANCERDX"]
    df["IsDiagnosedDiabetes"] = df["DIABDX"]
    df["IsDiagnosedAsthma"] = df["ASTHDX"]
    df["LongSinceLastFluVaccination"] = df["FLUSHT53"]
    df["TakesAspirinFrequently"] = df["ASPRIN53"]
    df["WearsSeatBelt"] = df["SEATBE53"]

    df = df.dropna()
    df = df[['group1', 'group2', 'BMI', 'Age', 'Region', 'MaritalStatus', 'Student', 'Race', 'Education', 'FamilyIncomeWage',
               'IsHadHeartAttack', 'IsHadStroke', 'IsDiagnosedCancer', 'IsADHD/ADD_Diagnisos',
                'IsDiagnosedAsthma', 'IsBornInUSA',
               'DoesDoctorRecommendExercise', 'LongSinceLastFluVaccination', 'TakesAspirinFrequently', 'Exercise',
               'WearsSeatBelt', 'FeltNervous', 'HoldHealthInsurance', 'IsWorking', 'CurrentlySmoke', 'IsDiagnosedDiabetes', 'HowOftenCheckInDoctor']]
    df = df.loc[(df["group1"] == 1) | (df["group2"] == 1)]
    proportion_of_neg1 = (df == -1).mean()
    df = df.loc[:, proportion_of_neg1 <= 0.5]
    for column in ['Age', 'BMI']:
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].dropna().nunique()
            if unique_values > 5:
                # Calculate percentiles, ensuring unique bin edges
                percentiles = np.percentile(df[column].dropna(), [0, 20, 40, 60, 80, 100])
                percentiles = np.unique(percentiles)  # Remove duplicate edges

                if len(percentiles) > 2:  # At least two unique bin edges are required
                    bin_labels = [
                                     f"<{percentiles[1]:.2f}"
                                 ] + [
                                     f"{percentiles[i]:.2f}-{percentiles[i + 1]:.2f}" for i in range(1, len(percentiles) - 2)
                                 ] + [f">{percentiles[-2]:.2f}"]
                    df[column] = pd.cut(
                        df[column], bins=percentiles, labels=bin_labels, include_lowest=True
                    )
    numerical_cols = df.select_dtypes(include=['number']).columns
    df = df[(df[numerical_cols] >= 0).all(axis=1)]
    return df



SUBPOPULATIONS = ['MaritalStatus', 'Region', 'Race',
                  'IsDiagnosedAsthma', 'IsBornInUSA', 'IsWorking']


def create_value_dict(df: pd.DataFrame, columns: list) -> dict:
    result = {}

    for col in columns:
        unique_values = df[col].dropna().unique()
        for value in unique_values:
            key = f"{col}_{value}"
            result[key] = {
                "att": col,
                "value": lambda x, v=value: 1 if pd.notna(x) and x == v else 0
            }
    return result

TREATMENTS_COLUMNS = ['IsHadStroke', 'DoesDoctorRecommendExercise', 'TakesAspirinFrequently',
                     'WearsSeatBelt', 'Exercise', 'LongSinceLastFluVaccination', 'CurrentlySmoke']


OUTCOME_COLUMN = "BMI"

COLUMNS_TO_IGNORE = []

def filter_facts(avg1, avg2):
    return avg2 > avg1

#d = build_mini_df()
#d.to_csv("outputs/meps/clean_data.csv", index=False)
