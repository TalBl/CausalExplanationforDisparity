import streamlit as st
import pandas as pd
import ast
import openai
from openai import OpenAI

OPEN_AI_API_KEY = None
with open('demo/gpt_api_key.txt') as f:
    OPEN_AI_API_KEY = f.read()
client = OpenAI(api_key=OPEN_AI_API_KEY)
from Utils import Dataset

st.set_page_config(page_title="DisEx", layout='wide', initial_sidebar_state="expanded")

# st.markdown(
#     f"""
#         <style>
#         .stApp {{
#             background-color: #FFFFFF;
#             color: #000000;
#         }}
#         </style>
#         """,
#     unsafe_allow_html=True
# )
# import base64
#
# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
# page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/jpg;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % get_base64("b.jpg")
# st.markdown(page_bg_img, unsafe_allow_html=True)

def exists_group(row, group, check):
    if check:
        return True
    for condition in group:
        if condition['operator'] == "=":
            if row[condition['column']] != condition['value']:
                return False
        else:
            if row[condition['column']] == condition['value']:
                return False
    return True

def display_input_row(df, index, group):
    att, op, value = st.columns([2,1,5])
    att.selectbox("Pick an attribute", df.columns, key=f"att_{group}_{index}")
    op.selectbox("", ["=", "!="], key=f"op_{group}_{index}")
    value.selectbox("Pick a value", list(sorted(set(df[att]))), key=f"value_{group}_{index}")


def increase_rows1():
    st.session_state[f'rows_group1'] += 1

def increase_rows2():
    st.session_state[f'rows_group2'] += 1

def calc_algorithm():
    df = pd.read_csv("outputs/so/find_k/5_0.65.csv")
    st.dataframe(df, use_container_width=True)

def valid_group(group_name):
    if f'clear_{group_name}' in st.session_state and st.session_state[f"clear_{group_name}"]:
        return True
    if group_name in st.session_state:
        if st.session_state[group_name]:
            return True
    return False

# todo: update clean path without groups + insert process groups to full.py

file = output_col = group1_query = group2_query = file_dag = imutable_atts = mutable_atts = None

if "page" not in st.session_state:
    st.session_state.page = "Input"

def switch_page(page_name):
    st.session_state.page = page_name

if st.session_state.page == "Input":
    with st.container():
        # First row: Centered title
        _, title_col, _ = st.columns([1, 6, 1])
        with title_col:
            st.markdown("<h1 style='text-align: center; font-size: 60px; color: whit;'>DisEx</h1>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 7])
        with col1:
            css = '''
                <style>
                    [data-testid='stFileUploader'] {
                        width: max-content;
                    }
                    [data-testid='stFileUploader'] section {
                        padding: 0;
                        float: left;
                    }
                    [data-testid='stFileUploader'] section > input + div {
                        display: none;
                    }
                    [data-testid='stFileUploader'] section + div {
                        float: right;
                        padding-top: 0;
                    }
                
                </style>
            '''
            st.markdown(css, unsafe_allow_html=True)

            file = st.file_uploader(label="Upload your dataset", type=["csv"])
            if file is None:
                st.error("No found file.")
            else:
                df = pd.read_csv(file).head(20)
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if not num_cols:
                    st.error("No numeric attributes found.")
                l = [""]
                l.extend(num_cols)
                st.markdown(
                    """
                    <style>
                    /* Change text color of the label */
                    div[data-testid="stSelectBox"] label {
                        color: black !important;  /* Set font color to black */
                        font-weight: bold;  /* Make it bold */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                output_col = st.selectbox("Choose output attribute:", l)
                if output_col:
                    if "Group_A" not in st.session_state:
                        st.session_state.Group_A = []
                    if "Group_B" not in st.session_state:
                        st.session_state.Group_B = []

                    def remove_condition(group, index):
                        st.session_state[group].pop(index)
                        on_conditions_changed(group)

                    # Function to add a new condition to a given group
                    def add_condition(group):
                        st.session_state[group].append({"column": None, "operator": "=", "value": None})
                        on_conditions_changed(group)

                    def clear_all_conditions(group):
                        while st.session_state[group]:
                            st.session_state[group].pop()
                        st.session_state[group] = []
                        st.rerun()
                        on_conditions_changed(group)

                    def on_conditions_changed(group):
                        st.write(f"🔄 **Conditions Updated for {group}**:")

                    for group_name in ["Group_A", "Group_B"]:
                        with st.expander(f"{group_name.replace('_', ' ').title()}"):
                            clear_all = st.checkbox(f"Overall dataset for {group_name}", key=f"clear_{group_name}")
                            if st.session_state[f"clear_{group_name}"]:
                                clear_all_conditions(group_name)

                            # **Add Condition Button**
                            if st.button(f"➕ Add Condition to {group_name}", key=f"add_{group_name}"):
                                add_condition(group_name)

                            # Display condition selections dynamically
                            for i, condition in enumerate(st.session_state[group_name]):
                                cols = st.columns([3, 2, 3, 1])  # Layout

                                # Column selection dropdown
                                condition["column"] = cols[0].selectbox(
                                    "Column", df.columns, key=f"{group_name}_col_{i}",
                                    index=0 if condition["column"] is None else df.columns.get_loc(condition["column"])
                                )

                                # Operator selection dropdown (= or !=)
                                condition["operator"] = cols[1].selectbox(
                                    "Operator", ["=", "!="], key=f"{group_name}_op_{i}",
                                    index=["=", "!="].index(condition["operator"])
                                )

                                # Value selection dropdown (based on chosen column)
                                if condition["column"]:
                                    unique_values = df[condition["column"]].unique().tolist()
                                    condition["value"] = cols[2].selectbox(
                                        "Value", unique_values, key=f"{group_name}_val_{i}"
                                    )

                                # Remove condition button
                                if cols[3].button("🗑️", key=f"del_{group_name}_{i}"):
                                    remove_condition(group_name, i)
                                    st.rerun()
                if valid_group('Group_A') and valid_group('Group_B'):
                    file_dag = st.file_uploader("Upload DAG file", type=["txt"])
                if file_dag:
                    cols = df.columns.tolist()
                    excluded_cols = ['group1', 'group2', output_col]
                    available_cols = [col for col in cols if col not in excluded_cols]
                    if "imutable_atts" not in st.session_state:
                        st.session_state.imutable_atts = []
                    if "mutable_atts" not in st.session_state:
                        st.session_state.mutable_atts = []
                    imutable_atts = st.multiselect("Pick immutable atts:", available_cols, key=imutable_atts)
                if imutable_atts:
                    cols = df.columns.tolist()
                    mutable_atts = st.multiselect("Pick mutable atts:", available_cols, key=mutable_atts)
                if mutable_atts:
                    alpha = st.slider("Please select an alpha", min_value=0.0, max_value=1.0, value=0.5)
                    threshold = st.slider("Please select a threshold", min_value=0.0, max_value=1.0, value=0.01)
                    k = st.slider("Please select a k", min_value=1, max_value=20, value=5)
                    min_cate = st.slider("Please select a min ABS CATE", min_value=0.0, max_value=100.0, value=0.0)
                if file and output_col and group1_query and group2_query and file_dag and imutable_atts and mutable_atts:
                    calc_algorithm()
        with col2:
            if file and not output_col:
                st.dataframe(df)
            if output_col and 'Group_A' not in st.session_state:
                column_config = {
                    output_col: st.column_config.Column(
                        label=f"👉 {output_col}",  # Add an arrow for emphasis
                        pinned=True,
                    )
                }
                st.data_editor(df, column_config=column_config, height=400)
            if valid_group('Group_A'):
                def highlight_groups(s):
                    # if st.session_state["group1"] and st.session_state["group2"]:
                    if (exists_group(s, st.session_state["Group_A"], st.session_state[f"clear_Group_A"])) and (exists_group(s, st.session_state['Group_B'], st.session_state[f"clear_Group_B"])):
                        return ['background-color: #FABEDB']*len(s)
                    if exists_group(s, st.session_state["Group_A"], st.session_state[f"clear_Group_A"]):
                        return ['background-color: #BEEBFA']*len(s)
                    if exists_group(s, st.session_state['Group_B'], st.session_state[f"clear_Group_B"]):
                        return ['background-color: #DCBEFA']*len(s)
                    return ['background-color: #BEFAD3']*len(s)
                column_config = {
                    output_col: st.column_config.Column(
                        label=f"👉 {output_col}",  # Add an arrow for emphasis
                        pinned=True,
                    )
                }
                st.dataframe(df.style.apply(highlight_groups, axis=1), column_config=column_config)

                color_legend = {"Group A": "#BEEBFA",  "Group B": "#DCBEFA", "Both": "#FABEDB", "None": "#BEFAD3"}
                st.markdown("### Color Legend")

                legend, group_avg_col, filter_scenario_select, _ = st.columns([1, 1, 2, 1])
                with legend:
                    legend_html = ""
                    for label, color in color_legend.items():
                        legend_html += f"""
                        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
                            <div style='width: 20px; height: 20px; background-color: {color}; margin-right: 10px;'></div>
                            <span>{label}</span>
                        </div>
                        """
                    st.markdown(legend_html, unsafe_allow_html=True)
                if valid_group('Group_A') and valid_group("Group_B") and st.button("Calculate"):
                    switch_page("Output")

                with group_avg_col:
                    group_a_avg = df[df.apply(lambda r: exists_group(r, st.session_state["Group_A"], st.session_state[f"clear_Group_A"]), axis=1)].loc[:, output_col].mean()
                    group_b_avg = df[df.apply(lambda r: exists_group(r, st.session_state["Group_B"], st.session_state[f"clear_Group_B"]), axis=1)].loc[:, output_col].mean()
                    st.write(f"**Average Group A:** {group_a_avg:.2f}")
                    st.write(f"**Average Group B:** {group_b_avg:.2f}")

                with filter_scenario_select:
                    filter_scenario = st.selectbox("Choose filter scenario:", ['None', 'Investigate a disparate trend', 'Debugging bias', 'Discovering reverse trends'], key='filter_scenario')
if st.session_state.page == "Output":
    df2 = pd.read_csv("demo/res2.csv")
    st.session_state['df2'] = df2
    def on_change():
        """Function called when DataFrame changes."""
        st.write("Data has been updated!")

    def edit_treatment(index):
        """Edit treatment for a specific row."""
        with st.expander(f"Edit Treatment for Row {index + 1}"):
            new_treatment = []
            for i in range(3):  # Allow up to 3 conditions
                col1, col2, col3 = st.columns(3)
                attr = col1.selectbox(f"Attribute {i+1}", st.session_state.mutable_atts, key=f"attr_{index}_{i}")
                op = col2.selectbox(f"Operation {i+1}", ["=", "!="], key=f"op_{index}_{i}")
                val = col3.text_input(f"Value {i+1}", key=f"val_{index}_{i}")
                if attr and val:
                    new_treatment.append((attr, op, val))

            if st.button("Save Treatment", key=f"save_{index}"):
                st.session_state.df2.at[index, "Treatment"] = str(new_treatment)
                on_change()

    def remove_row(index):
        """Remove a row from the DataFrame."""
        st.session_state.df2.drop(index, inplace=True)
        st.session_state.df2.reset_index(drop=True, inplace=True)
        on_change()

    def add_row():
        """Add a new subpopulation row."""
        new_id = len(st.session_state.df2) + 1
        new_row = pd.DataFrame([{"ID": new_id, "Treatment": "None", "Subpopulation": "None"}])
        st.session_state.df2 = pd.concat([st.session_state.df2, new_row], ignore_index=True)
        on_change()

    def parse_column(value):
        """Safely parse set or tuple strings from the dataframe."""
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    def generate_explanations(group1, group2, outcome, df):
        """
        Generates formatted natural language explanations comparing treatment effects between two groups.
        The response from OpenAI includes HTML styling for Streamlit.
        """
        # Convert string columns to actual Python objects
        df["subpopulation"] = df["subpopulation"].apply(parse_column)
        df["treatment"] = df["treatment"].apply(parse_column)

        # Define the system prompt with formatting instructions
        system_message = (
            "You are an expert data analyst who provides insights in natural language. Your task is to compare the effect of a treatment between two groups in different subpopulations. Format the response using HTML with the following color styles:\n"
            "- Subpopulation conditions: `<span style='color:orange;'></span>`\n"
            "- Treatment variables: `<span style='color:blue;'></span>`\n"
            "- Group 1 (e.g., individuals with more experience): `<span style='background-color:yellow;'></span>`\n"
            "- Group 2 (e.g., individuals with less experience): `<span style='background-color:purple;color:white;'></span>`\n"
            "- All other words: `<span style='color:black;'></span>`\n"
            "The dataframe has the following fields: subpopulation, treatment, ate1, ate2."
            "Convert subpopulation, treatment, groups and outcome column to descriptive text with proper grammar.\n"
            "Do not return the values of ate, just specify for who the treatment effects more.\n"
            "The explanation of a row from the df must be returned in a single line with only the required HTML styling and no additional markdown or explanation.\n"
            "Here is an example of the format of an explanation that should be returned where group1 is (DevType, 'Analyst') and group2 is (DevType, 'Backend developer'):\n"
            "For <span style='color:orange;'>individuals who identify as White or of European descent</span>, income growth is more influenced by <span style='color:blue;'>having 24-26 years of coding experience</span> as an <span style='background-color:yellow;'>analyst</span> compared to <span style='background-color:purple;color:white;'>back-end developers</span>."
        )

        # Construct user prompt
        user_message = f"Compare the impact of the treatment between these groups:\n"
        user_message += f"Group 1: {group1} (highlighted in yellow)\n"
        user_message += f"Group 2: {group2} (highlighted in purple)\n"
        user_message += f"Outcome: {outcome}\n\n"

        for _, row in df.iterrows():
            subpop_str = ", ".join(row["subpopulation"])
            treatment_str = ", ".join([f"{attr} is {val}" for attr, val in row["treatment"]])
            user_message += (
                f"Subpopulation: {subpop_str}\n"
                f"Treatment: {treatment_str}\n"
                f"Effect for Group 1: {row['ate1']}\n"
                f"Effect for Group 2: {row['ate2']}\n\n"
            )

        user_message += "Generate five natural language explanations using the specified HTML formatting."

        client = openai.OpenAI(api_key=OPEN_AI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        explanations = response.choices[0].message.content.split('\n')
        explanations = [e for e in explanations if "<span" in e]
        return explanations

    all_columns = list(st.session_state.df2.columns)  # Get all column names
    all_columns.insert(0, "explanation")
    always_visible_cols = ["explanation"]
    default_selected = [*always_visible_cols, "avg_group1", "avg_group2", "support"]  # Default visible columns

    # Initialize session state for column selections
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = {col: col in default_selected for col in all_columns}

    # Initialize session state for explanations
    if "explanations" not in st.session_state:
        df_mini = st.session_state.df2[['subpopulation', 'treatment', 'ate1', 'ate2']]
        st.session_state.explanations = generate_explanations(st.session_state.Group_A, st.session_state.Group_B, output_col, df_mini)

    # Create an expander with checkboxes
    with st.expander("Select columns to display", expanded=False):
        for col in [col for col in all_columns if col not in always_visible_cols]:
            st.session_state.selected_columns[col] = st.checkbox(col, value=st.session_state.selected_columns[col])

    # Get the list of currently selected columns
    selected_cols = [col for col, selected in st.session_state.selected_columns.items() if selected]

    # Display the table row by row with buttons
    st.write("### Dataset")

    # Ensure enough space for buttons
    header_cols = st.columns([3] * len(selected_cols) + [3, 2])

    # Display headers
    for i, col in enumerate(selected_cols):
        header_cols[i].markdown(f"**{col}**")

    # Display data dynamically based on selected columns
    for index, row in st.session_state.df2.iterrows():
        cols = st.columns([3] * len(selected_cols) + [3, 2])

        for i, col in enumerate(selected_cols):
            if col == "explanation":
                cols[i].write(st.session_state.explanations[index], unsafe_allow_html=True)
            elif col == 'delta':
                cols[i].write(f"<span style='background-color:{'#B3FFAE' if row['delta'] >= 0 else '#FF6464'};'>{row[col]}</span>", unsafe_allow_html=True)
            else:
                cols[i].write(row[col])

        # Buttons for each row
        if cols[len(selected_cols)].button("✏️ Change Treatment", key=f"edit_{index}"):
            edit_treatment(index)
        if cols[len(selected_cols) + 1].button("❌ Remove", key=f"remove_{index}"):
            remove_row(index)

    st.write("---")
    scores_df = pd.read_csv("demo/5_0.65.csv")
    scores_row = scores_df.iloc[-1].to_dict()
    st.markdown(f"<h1><b>Utility</b>: {scores_row['utility']:.2f} <b>Intersection</b>: {scores_row['final_intersection']:.2f} <b>Score</b>: {scores_row['score']:.2f}</h1>", unsafe_allow_html=True)

    # Add new row button
    st.button("➕ Add Subpopulation", on_click=add_row)

    if st.button("Return to input settings"):
        switch_page("Input")


