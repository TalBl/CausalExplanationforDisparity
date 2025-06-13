from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

for d in ['meps','so', 'acs']:
    data_mpg = pd.read_csv(f"outputs/{d}/clean_data.csv").dropna()
    if 'Unnamed: 0' in data_mpg.columns:
        data_mpg = data_mpg.drop(columns=['Unnamed: 0'])
    data_mpg = data_mpg.drop(columns=['group1', 'group2'])

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
    pyd.write_raw(f"causal_graph_pc_{d}.dot")
    #
    from causallearn.search.ScoreBased.GES import ges

    Record = ges(data)

    pyd = GraphUtils.to_pydot(Record['G'], labels=labels)
    pyd.write_png("causal_graph_ges.png")
    pyd.write_raw(f"causal_graph_ges_{d}.dot")


    from causallearn.search.FCMBased import lingam
    model_lingam = lingam.ICALiNGAM()
    model_lingam.fit(data)

    from causallearn.search.FCMBased.lingam.utils import make_dot
    dot = make_dot(model_lingam.adjacency_matrix_, labels=labels)

    dot.save(f'causal_graph_lingam_{d}.dot')

    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    graph, sep_set = fci(data, test_method=fisherz, alpha=0.05)

    # Visualize and export the graph
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    pyd.write_raw(f"causal_graph_fci_{d}.dot")
