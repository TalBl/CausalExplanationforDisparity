import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from causallearn.search.FCMBased.lingam.utils import make_dot


def fix_duplicate_labels(dot_path, cleaned_path):
    """
    Fixes duplicate node labels and rewrites edges and node IDs
    using the provided column name labels.

    Args:
        dot_path (str): Path to original DOT file.
        cleaned_path (str): Path to save cleaned DOT file.
        labels (list): List of variable names (index = node ID).
    """
    with open(dot_path, "r") as f:
        lines = f.readlines()

    node_map = {}  # old ID -> new label
    edges = []
    header_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Node declaration
        if line.endswith("];") and "[label=" in line and "->" not in line:
            node_id = line.split("[")[0].strip()
            label_val = line.split("label=")[1].strip("];").strip('"')

            # Only store if not an X label
            if not label_val.startswith("X"):
                node_map[node_id] = label_val

        # Edge line
        elif "->" in line:
            src, rest = line.split("->")
            tgt = rest.split("[")[0].strip()
            attrs = "[" + rest.split("[", 1)[1] if "[" in rest else ""
            edges.append((src.strip(), tgt, attrs))

        # Header lines
        elif any(line.startswith(x) for x in ["digraph", "graph", "rankdir"]):
            header_lines.append(line)

    # Start writing the cleaned DOT
    with open(cleaned_path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")

        # Write renamed node definitions
        for new_name in node_map.values():
            f.write(f'"{new_name}" [label="{new_name}"];\n')

        # Write edges with renamed node IDs
        for src, tgt, attrs in edges:
            if src in node_map and tgt in node_map:
                src_name = node_map[src]
                tgt_name = node_map[tgt]
                f.write(f'"{src_name}" -> "{tgt_name}" {attrs}\n')

        f.write("}\n")
#['so', ['YearsCodingProf', 'Hobby', 'ConvertedSalary', 'Gender', 'Age', 'EducationParents', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country']]]:
                      # ['acs', ['TotalPersonEarnings', 'EducationalAttainment', 'MaritalStatus', 'WithADisability', 'Sex', 'HealthInsuranceCoverageRecode']]:

for d, cols_names in [['so', ['YearsCodingProf', 'Hobby', 'Gender', 'Age', 'EducationParents', 'RaceEthnicity_WhiteorofEuropeandescent', 'Country', 'ConvertedSalary']]]:
    data_mpg = pd.read_csv(f"outputs/{d}/clean_data2.csv").dropna()
    if cols_names:
        data_mpg = data_mpg[cols_names]
    if 'Unnamed: 0' in data_mpg.columns:
        data_mpg = data_mpg.drop(columns=['Unnamed: 0'])
    # data_mpg = data_mpg.drop(columns=['group1', 'group2'])
    data_np = data_mpg.values  # convert pandas to numpy
    labels = list(data_mpg.columns)  # preserve original names


    def convert_to_continuous(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df_copy = df.copy()

        # Identify column types
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = df_copy.select_dtypes(include=['bool']).columns.tolist()

        # Convert boolean columns to integers
        df_copy[boolean_cols] = df_copy[boolean_cols].astype(int)

        # Ordinal encode categorical columns

        encoder = OrdinalEncoder()
        if categorical_cols:
            df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])

        # Save metadata for reversing
        metadata = {
            'categorical_cols': categorical_cols,
            'boolean_cols': boolean_cols,
            'encoder': encoder
        }

        return df_copy.astype(float), metadata

    data_mpg, dict_metadata = convert_to_continuous(data_mpg)

    labels = [f'{col}' for i, col in enumerate(data_mpg.columns)]
    labels = [label.replace(',', '_') for label in labels]
    data = data_mpg.to_numpy()

    cg = pc(data)
    #
    # # Visualization using pydot
    from causallearn.utils.GraphUtils import GraphUtils
    #
    pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    pyd.write_raw(f"outputs/dags/causal_graph_pc_{d}.dot")

    # Usage
    # fix_duplicate_labels(f"outputs/dags/causal_graph_pc_{d}.dot", f"outputs/dags/causal_graph_pc_{d}.dot")

    #
    from causallearn.search.ScoreBased.GES import ges
    try:
        Record = ges(data)

        pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
        pyd.write_raw(f"outputs/dags/causal_graph_ges_{d}.dot")
    except Exception as e:
        print(f"failed to generate GES graph : {e}")
    # if d != 'acs':
    #     fix_duplicate_labels(f"outputs/dags/causal_graph_ges_{d}.dot", f"outputs/dags/causal_graph_ges_{d}.dot")


    from causallearn.search.FCMBased import lingam
    # model_lingam = lingam.ICALiNGAM()
    # model_lingam.fit(data)
    #
    # dot = make_dot(model_lingam.adjacency_matrix_, labels=labels)
    # dot.save(f'outputs/dags/causal_graph_lingam_{d}.dot')

    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    graph, sep_set = fci(data, test_method=fisherz, alpha=0.05)

    # Visualize and export the graph
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    pyd.write_raw(f"outputs/dags/causal_graph_fci_{d}.dot")
    # fix_duplicate_labels(f"outputs/dags/causal_graph_fci_{d}.dot", f"outputs/dags/causal_graph_fci_{d}.dot")
