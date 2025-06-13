import os
import pandas as pd
import numpy as np
from folktables import ACSDataSource
def get_Multiple_States_2018_All(state_code_list=["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                                                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                                                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                                                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                                                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]):
    #Last Done for states:["SD","NE","ND","AL","MT","NH","UT", "MO", "WI", "FL", "OK", "AR", "KS", "MN", "IA", "CO", "VT", "MD", "ME", "ID"]
    final_df = pd.DataFrame()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    for state_code in state_code_list:
        print(state_code)
        state_file_string=f"2018_{state_code}_data.csv"
        if not os.path.exists(state_file_string):
            state_data = data_source.get_data(states=[state_code], download=True)
            state_data.to_csv()
        else:
            state_data = pd.read_csv(state_file_string)
        final_df = pd.concat([final_df, state_data], ignore_index=True)
    final_df.to_csv(f"2018_all_data.csv", index=False)
def read_ACS_value_data():
    text = open('data/acs/cepr_acs_2018_varlabels_plus.log', 'r').read().split("-" * 73)[2]
    lines = text.split("\n")
    field_to_value_dict = {}  # field name-> {field value code-> string}
    d = {}
    field = None
    for i, line in enumerate(lines):
        line_text = line.strip()
        if line_text == '':
            continue
        if line_text.endswith(":"):  # new field
            if field is not None:
                field_to_value_dict[field] = d
            field = line_text[:-1]
            d = {}
        elif line_text.startswith("> "):  # continuation of previous line
            d[k] += line_text[2:]
        else:
            try:
                parts = line_text.split()
                k = parts[0]
                v = " ".join(parts[1:])
                d[k] = v
            except:
                print(f"line num {i}: {line_text}")
                return
    if field is not None:
        field_to_value_dict[field] = d
    return field_to_value_dict
def make_translation_for_ACS(fields_list):
    # df = read_ACS_fields_data(year=2018)
    df = pd.read_csv('data/acs/field_map.tsv', sep='\t')
    df['field label'] = df['field label'].apply(lambda s: s.strip('* '))
    trans = {}
    unmatched = []
    matched = 0
    unmatched_value_mapping = []
    field_to_value_dict = read_ACS_value_data()
    for field in fields_list:
        subset = df[df['field name'] == field]
        if len(subset) == 1:
            trans[field] = subset.iloc[0]['field label']
            matched += 1
        else:
            unmatched.append(field)
            continue
        # look for value mapping
        value_map_needed = subset.iloc[0]['field value map needed?']
        exclude = subset.iloc[0]['exclude?']
        if value_map_needed == 'no' or exclude == 'yes':
            continue
        elif value_map_needed == 'binary':
            if field == 'SEX':
                trans[(field, 1)] = 'Man'
                trans[(field, 2)] = 'Woman'
            else:
                trans[(field, 1)] = 'yes'
                trans[(field, 2)] = 'no'
        elif value_map_needed == 'binary(0,1)':
            trans[(field, 1)] = 'yes'
            trans[(field, 0)] = 'no'
        elif field.lower() in field_to_value_dict:
            value_to_meaning = field_to_value_dict[field.lower()]
            for v, meaning in value_to_meaning.items():
                value_for_trans = v
                if value_for_trans.isdigit():
                    value_for_trans = int(value_for_trans)
                trans[(field, value_for_trans)] = meaning
        else:  # mapping needed but not found
            unmatched_value_mapping.append(field)
    print(f"matched field names: {matched}/{len(fields_list)}. \nUnmatched: {unmatched}")
    print(f"missing value mapping for: {unmatched_value_mapping}")
    return trans
def convert_df_clean(df, dict_translation):
    df_new = pd.DataFrame()
    for column in list(df.columns):
        if column in dict_translation:
            new_column_name = dict_translation[column]
            matching_key = next((key for key in dict_translation.keys() if key[0] == column), None)
            if matching_key is not None:
                df_new[new_column_name] = df.apply(lambda row: dict_translation.get((column, row[column]), None), axis=1)
            else:
                df_new[new_column_name] = df[column]
    return df_new
SUBPOPULATIONS = ['Ancestry recode', 'Region', 'Available for Work',
                  'Ancestry recode - first entry', 'Self-employment income past 12 months',
                  'Language other than English spoken at home', 'Citizenship status', 'Retirement income past 12 months',
                  'All other income past 12 months',
                  'Percent of poverty status', 'Marital status', 'Social Security income past 12 months',
                  "Person's weight replicate 4", 'Number of times married', 'Adjustment factor for income and earnings dollar amounts',
                  'Mobility status (lived here 1 year ago)', 'Usual hours worked per week past 12 months', 'Vision difficulty', 'Related child',
                  'When last worked', 'Ancestry recode - second entry', 'Hearing difficulty', 'Raw labor-force status',
                  'Medicare, for people 65 and older, or people with certain disabilities', 'person weight', 'VA (Health Insurance through VA Health Care)',
                  'Self-care difficulty', 'Interest, dividends, and net rental income past 12 months',
                  "Person's weight replicate 2", "Total person's income", 'state code', 'Married, spouse present/spouse absent',
                  'Married in the past 12 months', 'Ambulatory difficulty', 'Health insurance coverage recode', 'Supplementary Security Income past 12 months',
                  'Divorced in the past 12 months', "Person's weight replicate 3", 'Independent living difficulty', 'Widowed in the past 12 months',
                  'Georgraphic division', 'Public assistance income past 12 months', "Person's weight replicate 1", "Person's weight replicate 5",
                  'Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability']

TREATMENTS = ['Educational attainment', 'Wages or salary income past 12 months',
              'Sex', 'Quarter of birth', 'Insurance purchased directly from an insurance company',
              'Year last married', 'Temporary absence from work', 'Informed of recall',
              'Age', 'With a disability', 'Indian Health Service', 'Place of birth', 'On layoff from work',
              'Weeks worked during past 12 months', 'Hispanic, Detailed', 'Relationship to reference person',
              'Insurance through a current or former employer or union', 'Nativity', 'Cognitive difficulty', 'Looking for work',
              "Total person's earnings", 'Class of Worker', 'School enrollment', 'TRICARE or other military health care',
              'Occupation recode', 'Worked last week']

SUBPOPULATIONS = ['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                  'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                  'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity']

TREATMENTS = ['Wages or salary income past 12 months', 'Temporary absence from work', "Total person's earnings", 'Occupation recode', 'Worked last week',
              'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment',
              'Insurance through a current or former employer or union']

education_mapping = {
    'No schooling completed': 'No Formal Education',
    'Nursery school, preschool': 'No Formal Education',
    'Kindergarten': 'No Formal Education',
    'Grade 1': 'Elementary School',
    'Grade 2': 'Elementary School',
    'Grade 3': 'Elementary School',
    'Grade 4': 'Elementary School',
    'Grade 5': 'Elementary School',
    'Grade 6': 'Elementary School',
    'Grade 7': 'Middle School',
    'Grade 8': 'Middle School',
    'Grade 9': 'High School',
    'Grade 10': 'High School',
    'Grade 11': 'High School',
    '12th grade - no diploma': 'High School',
    'Regular high school diploma': 'High School',
    'GED or alternative credential': 'High School',
    'Some college, but less than 1 year': 'Some College',
    '1 or more years of college credit, no degree': 'Some College',
    "Associate's degree": "Associate's Degree",
    "Bachelor's degree": "Bachelor's Degree",
    "Master's degree": "Master's Degree",
    "Professional degree beyond a bachelor's degree": "Professional Degree",
    "Doctorate degree": "Doctorate Degree",
}

occupation_mapping = {
    'TRN': 'Transportation',
    'ENT': 'Entertainment',
    'SCI': 'Science',
    'CMS': 'Communications',
    'CMM': 'Community and Social Services',
    'SAL': 'Sales',
    'MIL': 'Military',
    'EAT': 'Food Preparation and Serving',
    'PRD': 'Production',
    'PRS': 'Personal Care and Service',
    'CLN': 'Cleaning and Maintenance',
    'CON': 'Construction',
    'BUS': 'Business Operations',
    'EDU': 'Education',
    'ENG': 'Engineering',
    'OFF': 'Office and Administrative Support',
    'MGR': 'Management',
    'Unemployed': 'Unemployed',
    'PRT': 'Protective Services',
    'RPR': 'Repair and Maintenance',
    'FFF': 'Farming, Fishing, and Forestry',
    'HLS': 'Healthcare Support',
    'MED': 'Healthcare Practitioners and Technical',
    'FIN': 'Financial Operations',
    'EXT': 'Extraction',
    'LGL': 'Legal',
}

def clean_and_transform_data():
    cleaned_df = pd.read_csv("outputs/acs/2018_all_data_clean.csv")
    cleaned_df['Total person earnings'] = cleaned_df["Total person's earnings"]
    cleaned_df = cleaned_df[['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                             'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                             'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity',
                             'Wages or salary income past 12 months', 'Temporary absence from work', "Total person's income", 'Occupation recode', 'Worked last week',
                             'Insurance purchased directly from an insurance company', 'Indian Health Service', 'Class of Worker', 'Informed of recall', 'Educational attainment',
                             'Insurance through a current or former employer or union','Race/Ethnicity','Health insurance coverage recode',
                             'Gave birth within past year', 'Place of work - State or foreign country recode', 'Ability to speak English',
                             'Widowed in the past 12 months', 'person weight', 'When last worked', 'Georgraphic division', 'Raw labor-force status',
                             'Adjustment factor for income and earnings dollar amounts', 'Employment status of parents', "Total person earnings",
                             "Usual hours worked per week past 12 months", "Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability",
                             "Field of degree - Science and Engineering flag", "Weeks worked during past 12 months", "Looking for work"]]
    for column in cleaned_df.columns:
        if column == "Total person income":
            continue
        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
            unique_values = cleaned_df[cleaned_df[column] >= 0][column].dropna().nunique()
            if unique_values > 5:
                # Calculate percentiles, ensuring unique bin edges
                percentiles = np.percentile(cleaned_df[column].dropna(), [0, 20, 40, 60, 80, 100])
                percentiles = np.unique(percentiles)  # Remove duplicate edges

                if len(percentiles) > 2:  # At least two unique bin edges are required
                    bin_labels = [
                                     f"<{percentiles[1]:.2f}"
                                 ] + [
                                     f"{percentiles[i]:.2f}-{percentiles[i + 1]:.2f}" for i in range(1, len(percentiles) - 2)
                                 ] + [f">{percentiles[-2]:.2f}"]
                    cleaned_df[column] = pd.cut(
                        cleaned_df[column], bins=percentiles, labels=bin_labels, include_lowest=True
                    )
                else:
                    # If there aren't enough unique percentiles, skip binning
                    cleaned_df[column] = cleaned_df[column].astype(str)
    cleaned_df['Occupation recode'] = cleaned_df['Occupation recode'].apply(lambda x: x.split("-")[0] if type(x)==str else x)
    cleaned_df['Educational attainment'] = cleaned_df['Educational attainment'].map(education_mapping)
    cleaned_df['Occupation recode'] = cleaned_df['Occupation recode'].map(occupation_mapping)
    cleaned_df['Health insurance coverage recode'] = cleaned_df['Health insurance coverage recode'].apply(lambda x: 1 if x == "yes" else 0)
    #cleaned_df['group1'] = cleaned_df['Race/Ethnicity'].apply(lambda x: 1 if x != "White alone" else 0)
    cleaned_df['group1'] = cleaned_df['Occupation recode'].apply(lambda x: 1 if x in ["Cleaning and Maintenance", "Farming, Fishing, and Forestry", "Repair and Maintenance", "Construction"] else 0)
    cleaned_df['group2'] = 1
    cleaned_df = cleaned_df[['Temporary absence from work', 'Worked last week', "person weight",
                             'Widowed in the past 12 months', "Total person earnings",
                             'Educational attainment', 'Georgraphic division', 'Sex', 'Age', 'With a disability', "Race/Ethnicity",
                             'Region', 'Language other than English spoken at home', 'state code',
                             'Marital status', 'Nativity', 'Related child', 'group1', 'group2', 'Health insurance coverage recode',
                             'Gave birth within past year', 'Field of degree - Science and Engineering flag']]
    #threshold = 0.5 * len(cleaned_df)  # 50% of the number of rows
    #cleaned_df = cleaned_df.loc[:, cleaned_df.isnull().sum() <= threshold]
    return cleaned_df



def calc_raw():
    # get_Multiple_States_2018_All()
    fields = ['AGEP', 'CIT', 'MAR', 'SCHL', 'PINCP', 'DIS', 'RAC1P', 'WKL', "SEX"]
    df = pd.read_csv("outputs/acs/raw_data.csv")
    fields_list = [
        # Demographics
        "AGEP", "SEX", "HISP", "RAC1P", "CIT", "REGION", "ST", "POBP", "YOEP", "MSP",
        "DIVISION", "MIG", "MIGSP", "MIGPUMA", "NATIVITY", "QTRBIR", "ANC", "ANC1P", "ANC2P", "DECADE",

        # Education
        "SCH", "SCHL", "SCHG", "FOD1P", "FOD2P",

        # Employment and Work
        "COW", "OCCP", "INDP", "WAGP", "WKHP", "WKL", "WKW", "WRK", "ESR", "JWMNP",
        "JWRIP", "JWTR", "LANX", "LANP", "INTP", "SEMP", "PERNP", "PINCP",

        # Income and Poverty
        "POVPIP", "RETP", "SSP", "OIP", "PAP", "SSIP", "ADJINC", "PWGTP", "HICOV", "HINS1",
        "HINS2", "HINS3", "HINS4", "HINS5", "HINS6", "HINS7",

        # Housing and Location
        "RC", "ESP", "NWLK", "NWAB", "NWAV", "NWLA", "NWRE", "DRIVESP", "POWPUMA", "POWSP",

        # Family and Relationships
        "RELP", "NOP", "MAR", "MARHD", "MARHM", "MARHT", "MARHW", "MARHYP",

        # Health
        "DIS", "DPHY", "DEAR", "DEYE", "DOUT", "DDRS", "DREM", "HICOV", "FER",

        # Transportation
        "JWAP", "JWDP", "DRAT", "DRATX", "JWMNP",

        # Derived Attributes
        "ENG", "GCR", "WKHP", "HINS4", "SCIENGP"
    ]
    df = df[list(set(fields_list))]
    dict_translation = make_translation_for_ACS(list(df.columns))
    df_clean = convert_df_clean(df, dict_translation)
    df_clean.to_csv("outputs/acs/2018_all_data_clean.csv", index=False)

#calc_raw()
#df = pd.read_csv("outputs/acs/2018_all_data_clean.csv")
# df_clean = clean_and_transform_data()
# df_clean.to_csv("outputs/acs/clean_data.csv", index=False)

"""
# Path to your text file
file_path = 'data/acs/edges.txt'

# Read the file and process the edges
with open(file_path, 'r') as file:
    edges = [eval(line.strip()) for line in file]  # Convert each line to a tuple using eval

# Generate the graph description text
graph_text = "\n".join([f"'{edge[0]} -> {edge[1]};'," for edge in edges])

# Print or save the result
print(graph_text)

# Optionally, save the output to another file
output_path = 'data/acs/causal_dag.txt'
with open(output_path, 'w') as file:
    file.write(graph_text)
group2 = "White alone"
OUTCOME_COLUMN = "Health insurance coverage recode"
df = pd.read_csv("outputs/acs/clean_data.csv")
treatments = ['Weeks worked during past 12 months', 'Veteran Service Disability Rating (percentage)', 'Looking for work', 'Ancestry recode', 'Cognitive difficulty', 'Travel time to work (mins)', 'Year of Entry', 'Self-employment income past 12 months', 'Language other than English spoken at home', 'Retirement income past 12 months', 'Vehicle occupancy (to work)', 'Recoded field of degree - second entry', 'Nativity of parent', 'Year last married', 'All other income past 12 months', 'Gave birth within past year', 'Percent of poverty status', 'With a disability', 'Worked last week', 'Marital status', 'Social Security income past 12 months', "Person's weight replicate 4", 'Number of times married', 'Adjustment factor for income and earnings dollar amounts', 'Usual hours worked per week past 12 months', "Total person's earnings", 'Class of Worker', 'Vision difficulty', 'Related child', 'When last worked', 'Hispanic, Detailed', 'Hearing difficulty', 'Raw labor-force status', 'Medicare, for people 65 and older, or people with certain disabilities', 'person weight', 'Indian Health Service', 'VA (Health Insurance through VA Health Care)', 'Self-care difficulty', 'Veteran Service Disability Rating (checkbox)', 'Interest, dividends, and net rental income past 12 months', 'Place of work PUM area code', "Person's weight replicate 2", "Total person's income", 'Means of transportation to work', 'Wages or salary income past 12 months', 'Number of vehicles calculated from JWRI', 'Married in the past 12 months', 'Ambulatory difficulty', 'Insurance through a current or former employer or union', 'Supplementary Security Income past 12 months', 'Informed of recall', 'Divorced in the past 12 months', "Person's weight replicate 3", 'Independent living difficulty', 'School enrollment', 'Widowed in the past 12 months', 'Educational attainment', 'Public assistance income past 12 months', 'Grade level attending', 'Relationship to reference person', "Person's weight replicate 1", 'Occupation recode', 'Insurance purchased directly from an insurance company', "Person's weight replicate 5", 'Employment status of parents', 'Temporary absence from work', 'Time of departure for work (hr & min)', 'Recoded field of degree - first entry', 'Time of arrival at work (hr & min)', 'On layoff from work', 'TRICARE or other military health care', 'Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability', 'Place of work - State or foreign country recode', 'Industry recode', 'Available for Work']
for c in df.columns:
    if c != OUTCOME_COLUMN and c not in SUBPOPULATIONS:
        treatments.append(c)
print(treatments)"""

def filter_subs(avg1, avg2):
    return True #avg2 < avg1

def filter_treats(avg1, avg2):
    return avg2 * avg1 < 0
