from algorithms.final_algorithm.full import acs, meps, so
import networkx as nx
from networkx.drawing.nx_pydot import write_dot


def create_default_causal_graph(list_A, list_B, outcome_attr):
    G = nx.DiGraph()

    # A → B edges
    for a in list_A:
        for b in list_B:
            G.add_edge(a, b)

    # A → B edges
    for a in list_A:
        G.add_edge(a, outcome_attr)

    # B → Outcome edge
    for b in list_B:
        G.add_edge(b, outcome_attr)

    return G

#['so', ['YearsCodingProf', 'Hobby', 'ConvertedSalary', 'Gender', 'Age', 'EducationParents', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country']]]:
                      # ['acs', ['TotalPersonEarnings', 'EducationalAttainment', 'MaritalStatus', 'WithADisability', 'Sex', 'HealthInsuranceCoverageRecode']]:
    # meps = Dataset(name="meps", outcome_col="FeltNervous",
    #                treatments=['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking',
    #                            'LongSinceLastFluVaccination', 'WearsSeatBelt', 'TakesAspirinFrequently'],
    #                subpopulations=['MaritalStatus', 'Region', 'Race', 'Age', 'IsDiagnosedDiabetes',
    #                                'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
dicty = {'acs': [['MaritalStatus', 'WithADisability', 'Sex'], ['TotalPersonEarnings', 'EducationalAttainment'], 'HealthInsuranceCoverageRecode'],
         'so': [['Gender', 'Age', 'EducationParents', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country'], ['YearsCodingProf', 'Hobby'], 'ConvertedSalary'],
         'meps': [['MaritalStatus', 'Region', 'Race', 'Age', 'IsDiagnosedDiabetes', 'IsDiagnosedAsthma', 'IsBornInUSA', 'DoesDoctorRecommendExercise'],
                  ['Exercise', 'CurrentlySmoke', 'HoldHealthInsurance', 'Student', 'IsWorking', 'LongSinceLastFluVaccination', 'WearsSeatBelt', 'TakesAspirinFrequently'], 'FeltNervous']}

for dataset, value in dicty.items():
    l1, l2, outcome = value
    G = create_default_causal_graph(l1, l2, outcome)
    write_dot(G, f"outputs/dags/causal_graph_default_{dataset}.dot")

