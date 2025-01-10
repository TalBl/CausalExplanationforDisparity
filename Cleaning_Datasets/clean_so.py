import pandas as pd
import numpy as np

def convert_to_yearly(row):
    if row['SalaryType'] == 'Weekly':
        return row['ConvertedSalary'] * 52  # Assuming 52 weeks per year
    elif row['SalaryType'] == 'Monthly':
        return row['ConvertedSalary'] * 12  # Assuming 12 months in a year
    else:  # Already yearly
        return 0


INPUT_DF_PATH = "data/so/2018_data.csv"
def build_mini_df():
    df = pd.read_csv(INPUT_DF_PATH)
    df['group1'] = np.where(df['FormalEducation'].str.contains('Bachelor', case=False, na=False), 1, 0)
    df['group2'] = np.where(df['FormalEducation'].str.contains('Master', case=False, na=False), 1, 0)
    df['ConvertedCompYearly'] = df.apply(convert_to_yearly, axis=1)
    options_df = df["DevType"].str.get_dummies(sep=";")
    l = ["DevType_"+x.replace(" ", "") for x in list(options_df.columns)]
    options_df.columns = l
    df = pd.concat([df.drop(columns=["DevType"]), options_df], axis=1)
    options_df = df["RaceEthnicity"].str.get_dummies(sep=";")
    l2 = ["RaceEthnicity_"+x.replace(" ", "") for x in list(options_df.columns)]
    options_df.columns = l2
    df = pd.concat([df.drop(columns=["RaceEthnicity"]), options_df], axis=1)
    df = df.loc[(df["group1"] == 1) | (df["group2"] == 1)]
    wanted_columns = ['group1', 'group2', 'YearsCodingProf', 'HopeFiveYears','Age',
                      'Gender', 'JobSatisfaction', 'Hobby', 'Student',
                      'LastNewJob', 'WakeTime', 'Exercise', 'ConvertedCompYearly', 'Country'] + l + l2
    df = df[wanted_columns]
    for column in ['Age']:
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
    df = df.dropna(subset=['ConvertedCompYearly'])
    return df


SUBPOPULATIONS_raw = ['Gender', 'Country','Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                                              'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                                              'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                                              'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent']

SUBPOPULATIONS = ['Gender', 'FormalEducation', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
                  'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
                  'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
                  'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country']

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


def filter_facts(avg1, avg2):
    return avg1 > avg2


OUTCOME_COLUMN = "ConvertedCompYearly"

TREATMENTS_COLUMNS = ['YearsCodingProf', 'JobSatisfaction', 'Hobby', 'LastNewJob', 'Exercise', 'Student',
                      'WakeTime', 'DevType', 'HopeFiveYears']

COLUMNS_TO_IGNORE = ['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                     'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                     'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                     'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0']


def clean2():
    df = pd.read_csv(INPUT_DF_PATH)
    df['group1'] = df['RaceEthnicity'].apply(lambda x: 1 if x != "White or of European descent" else 0)
    df['group2'] = 1
    df['ConvertedCompYearly'] = df.apply(convert_to_yearly, axis=1)
    options_df = df["DevType"].str.get_dummies(sep=";")
    l = ["DevType_"+x.replace(" ", "") for x in list(options_df.columns)]
    options_df.columns = l
    df = pd.concat([df.drop(columns=["DevType"]), options_df], axis=1)
    wanted_columns = ['group1', 'group2', 'YearsCodingProf', 'HopeFiveYears','Age', 'FormalEducation', 'MilitaryUS', 'CurrencySymbol', 'HoursOutside', 'HoursComputer',
                      'Gender', 'JobSatisfaction', 'Hobby', 'Student', 'Employment', 'UndergradMajor', 'StackOverflowRecommend', 'SkipMeals', 'EducationParents',
                      'LastNewJob', 'WakeTime', 'Exercise', 'ConvertedCompYearly', 'Country'] + l
    subpopulations = ['HopeFiveYears', 'Age', 'FormalEducation', 'MilitaryUS', 'CurrencySymbol', 'HoursOutside', 'HoursComputer', 'Gender',
                      'JobSatisfaction', 'Employment', 'UndergradMajor', 'StackOverflowRecommend', 'SkipMeals', 'EducationParents', 'Exercise', 'Country']
    df = df[wanted_columns]
    for column in ['Age']:
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
    df = df.dropna(subset=['ConvertedCompYearly'])
    return df

# d = clean2()
# d.to_csv("outputs/so/clean_data2.csv")