import pandas as pd
from Utils import Dataset
from algorithms.final_algorithm.find_treatment_new import write_pickle_files
df = pd.read_csv("outputs/acs/clean_data.csv")
c = list(df.columns)
c.remove("group1")
c.remove("group2")
from Cleaning_Datasets.clean_acs import filter_subs as acs_filter_subs, filter_treats as acs_filter_treats


acs = Dataset(name="acs", outcome_col="Total person's earnings",
              treatments=c,
              subpopulations=['Sex', 'Age', 'With a disability', 'Place of birth', 'School enrollment', 'Cognitive difficulty',
                              'Region', 'Language other than English spoken at home', 'Citizenship status', 'state code',
                              'Percent of poverty status', 'Marital status', 'Hearing difficulty', 'Related child', 'Nativity'],
              columns_to_ignore=[], clean_path="outputs/acs/clean_data.csv", func_filter_subs=acs_filter_subs, need_filter_subpopulations=True, need_filter_treatments=True,
              func_filter_treats=acs_filter_treats)

write_pickle_files(acs)

