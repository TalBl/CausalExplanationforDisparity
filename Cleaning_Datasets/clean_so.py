import pandas as pd
import numpy as np


INPUT_DF_PATH = "data/so/2018_data.csv"
def build_mini_df():
    df = pd.read_csv(INPUT_DF_PATH)
    df['group1'] = np.where(df['DevType'].str.contains('Data or business analyst', case=False, na=False), 1, 0)
    df['group2'] = np.where(df['DevType'].str.contains('Back-end developer', case=False, na=False), 1, 0)
    options_df = df["RaceEthnicity"].str.get_dummies(sep=";")
    l2 = ["RaceEthnicity_"+x.replace(" ", "") for x in list(options_df.columns)]
    options_df.columns = l2
    df = pd.concat([df.drop(columns=["RaceEthnicity"]), options_df], axis=1)
    wanted_columns = ['group1', 'group2', 'YearsCodingProf', 'HopeFiveYears','Age',
                      'Gender', 'JobSatisfaction', 'Hobby', 'Student',
                      'FormalEducation', 'WakeTime', 'Exercise', 'ConvertedSalary', 'Country', 'EducationParents',
                      'SexualOrientation', 'Dependents', 'HoursComputer', 'UndergradMajor', 'CompanySize'] + l2
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
    df = df.dropna(subset=['ConvertedSalary'])
    df = df[['group1', 'group2', 'Gender', 'Age', 'RaceEthnicity_BlackorofAfricandescent', 'RaceEthnicity_EastAsian',
            'RaceEthnicity_HispanicorLatino/Latina', 'RaceEthnicity_MiddleEastern',
            'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian',
            'RaceEthnicity_SouthAsian', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country', 'SexualOrientation', 'EducationParents',
            'YearsCodingProf', 'HopeFiveYears', 'JobSatisfaction', 'Hobby', 'Student',
            'FormalEducation', 'WakeTime', 'Exercise', 'Dependents', 'HoursComputer', 'UndergradMajor', 'CompanySize', 'ConvertedSalary']]

    return df


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


# d = build_mini_df()
# d.to_csv("outputs/so/clean_data.csv", index=False)
