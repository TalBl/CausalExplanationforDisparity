import math

import streamlit as st
import pandas as pd
import ast
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re


OPEN_AI_API_KEY = None
with open('demo/gpt_api_key.txt') as f:
    OPEN_AI_API_KEY = f.read()
client = OpenAI(api_key=OPEN_AI_API_KEY)

st.set_page_config(page_title="DisEx", layout='wide', initial_sidebar_state="expanded")


def split_camel_case(text):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

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
            if row[condition['column']] == condition['value']:
                return True
        else:
            if row[condition['column']] != condition['value']:
                return True
    return False

def display_input_row(df, index, group):
    att, op, value = st.columns([2,1,5])
    att.selectbox("Pick an attribute", df.columns, key=f"att_{group}_{index}")
    op.selectbox("", ["=", "!="], key=f"op_{group}_{index}")
    value.selectbox("Pick a value", list(sorted(set(df[att]))), key=f"value_{group}_{index}")

def create_pie_chart(percentage):
    fig, ax = plt.subplots(figsize=(0.8, 0.8))  # Small size
    ax.pie([percentage, 1 - percentage], colors=["#4CAF50", "#E0E0E0"], startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_xticks([])  # Hide X ticks
    ax.set_yticks([])  # Hide Y ticks
    plt.axis("off")  # Hide axes
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" width="30"/>'

def create_colored_dot(s, min_value, max_value):
    normalized = (s - min_value) / (max_value - min_value)

    # Generate color gradient (Red to Green)
    r = (255 - normalized * 155).astype(int)  # Red decreases
    g = (100 + normalized * 155).astype(int)  # Green increases
    b = 100  # Fixed Blue

    return [f"background-color: rgb({r[i]}, {g[i]}, {b})" for i in range(len(s))]

def format_currency(value):
    return f"${int(value):,}"  # Convert to int, add thousand separators

def color_ate_cell(value, min_value, max_value):
    value = int(value.replace("$", "").replace(",", ""))
    normalized = (value - min_value) / (max_value - min_value)

    # When the value is in the middle, the color should be white
    if normalized < 0.5:
        r = int(255 - normalized * 510)  # Red decreases from 255 to 0
        g = int(normalized * 510)  # Green increases from 0 to 255
    else:
        r = 0  # Red stays at 0
        g = int((1 - normalized) * 510)  # Green decreases from 255 to 0

    b = 100  # Fixed Blue

    return f"background-color: rgb({r}, {g}, {b})"

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


def prompt_descriptive_group_name(group_conditions, overall_dataset_check=False):
    if overall_dataset_check:
        return "Overall dataset"
    if group_conditions is None or len(group_conditions) == 0:
        return None
    prompt = f"""
    Convert the following list of conditions into a concise, natural-language description:

    Conditions:
    {group_conditions}

    Formatting Guidelines:
    - Use simple, natural phrasing.
    - Omit unnecessary words like "The data includes" or "Only individuals with".
    - Keep the sentence compact while preserving meaning.
    - Ensure correct grammar and readability.

    Example Output:
    - "Coding professionally for 18-20 years and age is other than 18-24 years old."
    - "18-20 years of professional coding experience and not aged 18-24."

    Generate the description following these guidelines.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # return response["choices"][0]["message"]["content"].strip()
    return response.choices[0].message.content.replace(".", "").replace('"', "")

# todo: update clean path without groups + insert process groups to full.py

file_to_use = output_col = group1_query = group2_query = file_dag = imutable_atts = mutable_atts = clicked = None


with st.container():
    # First row: Centered title
    st.markdown("<h1 style='text-align: center; font-size: 60px; color: whit;'>DisEx</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 6], border=True)

    with col1:
        st.markdown("<h1 style='font-size:36px;'>Settings</h1>", unsafe_allow_html=True)
        # st.markdown('<div class="rounded-box">Column 1 Content', unsafe_allow_html=True)
        # css = '''
        #     <style>
        #         [data-testid='stFileUploader'] {
        #             width: max-content;
        #         }
        #         [data-testid='stFileUploader'] section {
        #             padding: 0;
        #             float: left;
        #         }
        #         [data-testid='stFileUploader'] section > input + div {
        #             display: none;
        #         }
        #         [data-testid='stFileUploader'] section + div {
        #             float: right;
        #             padding-top: 0;
        #         }
        #
        #     </style>
        # '''
        # st.markdown(css, unsafe_allow_html=True)
        _col1, _col2 = st.columns([1, 1], vertical_alignment='center')
        with _col1:
            file_to_use = st.file_uploader(label="Upload your dataset", type=["csv"], key="dataset")
        with _col2:
            file_dag = st.file_uploader("We will automatically discover causal DAG from your data once you upload it. If you want to use your own causal DAG, you can manually upload it here", type=["txt"], key="dag_file")
        # with _col3:
        #     st.button("Discover Causal DAG")
        # file_to_use = st.file_uploader(label="Upload your dataset", type=["csv"])
        # existing_files = {
        #     "Stack Overflow": "demo/clean_data_so.csv",
        #     "MEPS": "demo/sample2.csv",
        #     "ACS": "demo/sample3.csv",
        # }
        # existing_dag_files = {
        #     "Stack Overflow": "demo/causal_dag_so.txt",
        #     "MEPS": "sample2.csv",
        #     "ACS": "sample3.csv",
        # }
        # st.markdown(
        #     """
        #     <style>
        #     /* Reduce file uploader size */
        #     div[data-testid="stFileUploader"] section {
        #         padding: 8px !important;
        #         max-width: 300px !important;
        #         border-radius: 10px !important;
        #         background-color: #f8f9fa !important;
        #     }
        #
        #     /* Align elements */
        #     .file-container {
        #         display: flex;
        #         gap: 20px;
        #     }
        #
        #     /* Style the dropdown */
        #     .stSelectbox {
        #         max-width: 300px !important;
        #     }
        #
        #     /* Improve button appearance */
        #     div[data-testid="stButton"] button {
        #         border-radius: 8px !important;
        #         background-color: #4CAF50 !important;
        #         color: white !important;
        #         padding: 8px 15px !important;
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )
        #
        # st.write("Upload or Choose a File")
        #
        # col11, col22 = st.columns([1, 1])
        # with col11:
        #     selected_file_key = st.selectbox("Choose a file:", ["Upload manually"] + list(existing_files.keys()))
        # with col22:
        #     uploaded_file = st.file_uploader("Or upload your own file", type=["csv"])
        #
        # # Determine which file to use
        # if uploaded_file is not None:
        #     file_to_use = uploaded_file
        #     st.success("Using uploaded file.")
        # elif selected_file_key != "Upload manually":
        #     file_to_use = existing_files[selected_file_key]
        #     file_dag = existing_dag_files[selected_file_key]
        #     st.success(f"Using selected file: {file_to_use}")
        # else:
        #     file_to_use = None

        # if file_to_use:
        #     df = pd.read_csv(file_to_use)
        # _col1, _col2 = st.columns([2, 1], vertical_alignment='bottom')
        # with _col1:
        #     if file_dag:
        #         st.success(f"Using selected DAG file: {file_dag}")
        #     if not file_dag:
        #         file_dag = st.file_uploader("Upload DAG file", type=["txt"])
        # with _col2:
        #     st.button("Discover Causal DAG")
        if file_to_use:
            df = pd.read_csv(file_to_use).head(40)
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
                    # st.rerun()
                    on_conditions_changed(group)

                def on_conditions_changed(group):
                    st.write(f"🔄 **Conditions Updated for {group}**:")
                with st.container(border=True):
                    st.write("Group A")
                    group_name = "Group_A"
                    clear_all = st.checkbox(f"Overall dataset for {group_name}", key=f"clear_{group_name}")
                    if st.session_state[f"clear_{group_name}"]:
                        clear_all_conditions(group_name)

                    # **Add Condition Button**
                    if st.button(f"➕ Add Condition to {group_name}", key=f"add_{group_name}"):
                        add_condition(group_name)

                    # Display condition selections dynamically
                    for i, condition in enumerate(st.session_state[group_name]):
                        cols = st.columns([3, 2, 3, 1], vertical_alignment='bottom')  # Layout

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
                with st.container(border=True):
                    st.write("Group B")
                    group_name = "Group_B"
                    clear_all = st.checkbox(f"Overall dataset for {group_name}", key=f"clear_{group_name}")
                    if st.session_state[f"clear_{group_name}"]:
                        clear_all_conditions(group_name)

                    # **Add Condition Button**
                    if st.button(f"➕ Add Condition to {group_name}", key=f"add_{group_name}"):
                        add_condition(group_name)

                    # Display condition selections dynamically
                    for i, condition in enumerate(st.session_state[group_name]):
                        cols = st.columns([3, 2, 3, 1], vertical_alignment='bottom')  # Layout

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
                # for group_name in ["Group_A", "Group_B"]:
                #     with st.expander(f"{group_name.replace('_', ' ').title()}"):
                #         clear_all = st.checkbox(f"Overall dataset for {group_name}", key=f"clear_{group_name}")
                #         if st.session_state[f"clear_{group_name}"]:
                #             clear_all_conditions(group_name)
                #
                #         # **Add Condition Button**
                #         if st.button(f"➕ Add Condition to {group_name}", key=f"add_{group_name}"):
                #             add_condition(group_name)
                #
                #         # Display condition selections dynamically
                #         for i, condition in enumerate(st.session_state[group_name]):
                #             cols = st.columns([3, 2, 3, 1])  # Layout
                #
                #             # Column selection dropdown
                #             condition["column"] = cols[0].selectbox(
                #                 "Column", df.columns, key=f"{group_name}_col_{i}",
                #                 index=0 if condition["column"] is None else df.columns.get_loc(condition["column"])
                #             )
                #
                #             # Operator selection dropdown (= or !=)
                #             condition["operator"] = cols[1].selectbox(
                #                 "Operator", ["=", "!="], key=f"{group_name}_op_{i}",
                #                 index=["=", "!="].index(condition["operator"])
                #             )
                #
                #             # Value selection dropdown (based on chosen column)
                #             if condition["column"]:
                #                 unique_values = df[condition["column"]].unique().tolist()
                #                 condition["value"] = cols[2].selectbox(
                #                     "Value", unique_values, key=f"{group_name}_val_{i}"
                #                 )
                #
                #             # Remove condition button
                #             if cols[3].button("🗑️", key=f"del_{group_name}_{i}"):
                #                 remove_condition(group_name, i)
                #                 st.rerun()
            if valid_group('Group_A') and valid_group('Group_B'):
                cols = df.columns.tolist()
                excluded_cols = ['group1', 'group2', output_col]
                available_cols = [col for col in cols if col not in excluded_cols]
                if "imutable_atts" not in st.session_state:
                    st.session_state.imutable_atts = []
                if "mutable_atts" not in st.session_state:
                    st.session_state.mutable_atts = []
                imutable_atts = st.multiselect("Select immutable attributes", available_cols, key=imutable_atts)
            if imutable_atts:
                cols = df.columns.tolist()
                mutable_atts = st.multiselect("Select attributes that are actionable", available_cols, key=mutable_atts)
            if mutable_atts:
                options = [5, 10, 15, 20, 25, 30]
                options2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                # Render a row of tick marks above the slider
                value = st.select_slider("How many explanations do you want?", options=options)
                st.markdown(
                    """
                    <style>
                    div[data-testid="stSliderTickBarMin"],
                    div[data-testid="stSliderTickBarMax"] {
                        display: none;
                    }
                        .tick-marks { 
                            display: flex;
                            justify-content: space-between;
                            padding: 0;
                            margin-top: -20px;  /* Moves labels up */
                            font-size: 14px;
                            color: gray;
                        }
                    </style>
                    <div class="tick-marks">
                        <span>5</span><span>10</span><span>15</span><span>20</span><span>25</span><span>30</span>
                    </div>
                    """, unsafe_allow_html=True)
                value2 = st.select_slider("Balance parameter (Diversity - Utility)", options=options2, value=0.5)
                st.markdown(
                    """
                    <style>
                    div[data-testid="stSliderTickBarMin"],
                    div[data-testid="stSliderTickBarMax"] {
                        display: none;
                    }
                        .tick-marks { 
                            display: flex;
                            justify-content: space-between;
                            padding: 0;
                            margin-top: -20px;  /* Moves labels up */
                            font-size: 14px;
                            color: gray;
                        }
                    </style>
                    <div class="tick-marks">
                        <span>0</span><span>0.1</span><span>0.2</span><span>0.3</span><span>0.4</span><span>0.5</span><span>0.6</span><span>0.7</span><span>0.8</span><span>0.9</span><span>1</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    div[data-testid="stSliderTickBarMin"],
                    div[data-testid="stSliderTickBarMax"] {
                        display: none;
                    }
                        .tick-marks { 
                            display: flex;
                            justify-content: space-between;
                            padding: 0;
                            margin-top: -20px;  /* Moves labels up */
                            font-size: 14px;
                            color: black;
                        }
                    </style>
                    <div class="tick-marks">
                        <span>Diversity</span><span>Utility</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Create a slider with selectable values (5, 10, ..., 30)
                # value = st.slider("How many explanations do you want?", min_value=5, max_value=30, step=5)

                # k = st.slider("How many explanations do you want?", min_value=1, max_value=30, value=5, step=1)
                col3, col4 = st.columns([1, 2], vertical_alignment='center')
                with col3:
                    st.html("<p style='font-size: 24px; color: black; text-align: center; margin-top: 10px;'>Choose filter scenario</p>")

                    # st.write("Choose filter scenario")
                with col4:
                    filter_scenario = st.selectbox("c", ['Investigate a disparate trend', 'Debugging bias', 'Discovering reverse trends'], key='filter_scenario', label_visibility="collapsed")
                col5 = st.columns(1)[0]
                with col5:
                    button_html = """
                    <style>
                        .custom-button {
                            width: 100%;
                            background-color: green;
                            color: white;
                            font-size: 16px;
                            padding: 10px;
                            border-radius: 5px;
                            border: none;
                            cursor: pointer;
                        }
                        .custom-button:hover {
                            background-color: darkgreen;
                        }
                    </style>
                    <button class="custom-button" onclick="sendClick()">Explain</button>
                    <script>
                        function sendClick() {
                            var xhr = new XMLHttpRequest();
                            xhr.open("GET", "/button_clicked", true);
                            xhr.send();
                        }
                    </script>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)
        if file_to_use and output_col and group1_query and group2_query and imutable_atts and mutable_atts and st.experimental_get_query_params().get("button_clicked"):
                calc_algorithm()
    with col2:
        with st.container():
            st.markdown("<h1 style='font-size:36px;'>Data</h1>", unsafe_allow_html=True)
            if file_to_use and not output_col:
                st.dataframe(df)
            if output_col and 'Group_A' not in st.session_state:
                column_config = {
                    output_col: st.column_config.Column(
                        label=f"👉 {output_col}",  # Add an arrow for emphasis
                        pinned=True,
                    )
                }
                st.data_editor(df, column_config=column_config, height=400)
            group_a_text = 'Group A'
            group_b_text = 'Group B'
            if valid_group('Group_A'):
                group_a_text = prompt_descriptive_group_name(st.session_state.Group_A, st.session_state["clear_Group_A"])
            if valid_group('Group_B'):
                group_b_text = prompt_descriptive_group_name(st.session_state.Group_B, st.session_state["clear_Group_B"])
            if valid_group('Group_A'):
                def highlight_groups(s):
                    # if st.session_state["group1"] and st.session_state["group2"]:
                    # if (exists_group(s, st.session_state["Group_A"], st.session_state[f"clear_Group_A"])) and (exists_group(s, st.session_state['Group_B'], st.session_state[f"clear_Group_B"])):
                    #     return ['background-color: #FABEDB']*len(s)
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
                st.dataframe(df.style.apply(highlight_groups, axis=1), column_config=column_config, height=750)
                # "Both": "#FABEDB",
                color_legend = {group_a_text: "#BEEBFA",  group_b_text: "#DCBEFA",  "None": "#BEFAD3"}
                # st.markdown("### Color Legend")

                legend, group_avg_col = st.columns([2, 2])
                with legend:
                    legend_html = "<div style='display: flex; align-items: center; gap: 15px;'>"
                    for label, color in color_legend.items():
                        legend_html += f"<div style='display: flex; align-items: center;'>"
                        legend_html += f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px;'></div>"
                        legend_html += f"<span>{label}</span></div>"
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)
                if valid_group('Group_A') and valid_group("Group_B") and clicked:
                    # calc alg
                    x = 1
                with group_avg_col:
                    group_a_avg = df[df.apply(lambda r: exists_group(r, st.session_state["Group_A"], st.session_state[f"clear_Group_A"]), axis=1)].loc[:, output_col].mean()
                    group_b_avg = df[df.apply(lambda r: exists_group(r, st.session_state["Group_B"], st.session_state[f"clear_Group_B"]), axis=1)].loc[:, output_col].mean()
                    st.write(f"**Average {'salary' if output_col == 'ConvertedSalary' else split_camel_case(output_col).lower()} for {group_a_text}:** \${group_a_avg:.2f} \n  **Average {'salary' if output_col == 'ConvertedSalary' else split_camel_case(output_col).lower()} for {group_b_text}:** \${group_b_avg:.2f}")
                    # st.write(f"**Average Group B:** {group_b_avg:.2f}")
        with st.container():
            st.markdown("<h1 style='font-size:36px;'>Causal Explanations</h1>", unsafe_allow_html=True)
            df2 = pd.read_csv("demo/res2.csv")
            df2["support"] = df2["support"].apply(create_pie_chart)
            df2['affect_group1'] = df2['ate1'] / df2['avg_group1']
            df2['affect_group2'] = df2['ate2'] / df2['avg_group2']
            #df2["ate_group1"] = df2["affect_group1"].apply(create_colored_dot)
            #df2["ate_group2"] = df2["affect_group2"].apply(create_colored_dot)
            # column_descriptions = {
            #     "ATE1": f"Average Treatment Effect for {group_a_text}.",
            #     "ATE2": f"Average Treatment Effect for {group_b_text}.",
            #     "AVG_O1": f"Average salary for {group_a_text}.",
            #     "AVG_O2": f"Average salary for {group_b_text}.",
            # }
            # def create_tooltip(column_name, description):
            #     return f"""
            #         <div style="display: flex; align-items: center; gap: 5px;">
            #             <span>{column_name}</span>
            #             <span title="{description}" style="cursor: help; color: blue; font-weight: bold;">❓</span>
            #         </div>
            #         """

            # min_ate = min(df2["ate1"].min(), df2["ate2"].min())
            # max_ate = max(df2["ate1"].max(), df2["ate2"].max())
            df2["Avg Treatment Effect for Group A"] = df2["ate1"].apply(format_currency)
            df2["Avg Treatment Effect for Group B"] = df2["ate2"].apply(format_currency)
            df2["Avg TC for Group A"] = df2["avg_group1"].apply(format_currency)
            df2["Avg TC for Group B"] = df2["avg_group2"].apply(format_currency)
            df2["subpopulation"] = df2["subpopulation"].apply(prompt_descriptive_group_name)
            df2["treatment"] = df2["treatment"].apply(prompt_descriptive_group_name)
            # ll = [f"E{x}" for x in range(1, len(df2)+1)]
            # df2["new_index"] = [f"E{x}" for x in range(1, len(df2)+1)]
            df2.index = [f"E{x}" for x in range(1, len(df2)+1)]
            df2 = df2[["subpopulation", "support", "treatment", "Avg TC for Group A", "Avg TC for Group B", "Avg Treatment Effect for Group A", "Avg Treatment Effect for Group B"]]
            # df2.columns = ["subpopulation", "support", "AVG_O1", "AVG_O2", "treatment", "ATE1", "ATE2"]
            styled_df = df2.style.map(lambda v: color_ate_cell(v, min_value=-200000, max_value=200000), subset=["Avg Treatment Effect for Group A", "Avg Treatment Effect for Group B"])\
                        .set_properties(
                            **{"text-align": "left"}, subset=["subpopulation", "treatment"]
                        ).set_properties(
                            **{"text-align": "right"}, subset=["Avg TC for Group A", "Avg TC for Group B", "Avg Treatment Effect for Group A",
                                                               "Avg Treatment Effect for Group B"]
                        ) \
                .set_table_styles(
                [
                    {'selector': 'th', 'props': [('text-align', 'center')]},  # Center the header
                ]
            )

            html_output = styled_df.to_html(escape=False, index=False)
            # for column, tooltip in column_descriptions.items():
            #     # Create a regex pattern to match the <th> with the column name
            #     pattern = re.compile(rf'<th .*>{column}</th>')
            #     orig_header = re.findall(pattern, html_output)[0]
            #     new_header = orig_header.replace(column, create_tooltip(column, tooltip))
            #     html_output = html_output.replace(orig_header, new_header)

            st.markdown(html_output, unsafe_allow_html=True)
            # styled_headers = {col: create_tooltip(col, column_descriptions.get(col, "")) for col in df2.columns}
            # Add the gradient legend to the second column
            st.write("Legend for Average Treatment Effect")
            gradient_html = """
            <div style="width: 100%; height: 20px;
                background: linear-gradient(to right, rgb(139,0,0), rgb(255,255,255), rgb(0,128,0));
                border: 2px solid #808080; "></div>
            """
            st.markdown(gradient_html, unsafe_allow_html=True)

            # Add labels under the gradient in the second column
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; font-size: 18px;">
                    <span style="color: black;">Deterioration</span>
                    <span style="color: black;">No Effect</span>
                    <span style="color: black;">Improvement</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # st.dataframe(df3)
            scores_df = pd.read_csv("demo/5_0.65.csv")
            scores_df = scores_df[['utility', 'final_intersection', 'score']]
            scores_df = scores_df.rename(columns={'utility': 'Utility', 'final_intersection': 'Diversity', 'score': 'Overall quality'})
            scores_row = scores_df.iloc[-1].to_dict()
            colored_star = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" width="31.9727px" height="35.5859px" fill='yellow'>
              <path fill='#F6DC43' d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z" fill="gray"/>
            </svg>
            """
            grayed_star = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" width="31.9727px" height="35.5859px" fill='yellow'>
              <path d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z" fill="gray"/>
            </svg>
            """
            for i,score in enumerate(['Overall quality', 'Diversity', 'Utility']):
                col4, col5, col6 = st.columns([1,1,1], vertical_alignment='center')
                value_s = scores_row[score]
                num_yellow_stars = math.ceil(value_s / 0.2)
                num_grey_stars = 5 - num_yellow_stars
                with col4:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center;">
                            <h1 style="font-size: 25px; margin-right: 10px;">{score}</h1>
                    """, unsafe_allow_html=True)
                with col5:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center;">
                            <div style="display: flex;">
                                {''.join([colored_star for _ in range(num_yellow_stars)])}
                                {''.join([grayed_star for _ in range(num_grey_stars)])}
                    """, unsafe_allow_html=True)
                with col6:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center;">
                            <h1 style="font-size: 25px; margin-right: 10px;">{int(value_s*100)}%</h1>
                    """, unsafe_allow_html=True)


            # st.markdown(f"<div>{fa_star_svg}</div>", unsafe_allow_html=True)
            # st.bar_chart(scores_row, horizontal=True)
            # st.write("Utility:")
            # st.progress(scores_row['utility'])
            # st.write("Intersection:")
            # st.progress(scores_row['final_intersection'])
            # st.write("Score:")
            # st.progress(scores_row['score'])


# if st.session_state.page == "Output":
#     df2 = pd.read_csv("demo/res2.csv")
#     st.session_state['df2'] = df2
#     def on_change():
#         """Function called when DataFrame changes."""
#         st.write("Data has been updated!")
#
#     def edit_treatment(index):
#         """Edit treatment for a specific row."""
#         with st.expander(f"Edit Treatment for Row {index + 1}"):
#             new_treatment = []
#             for i in range(3):  # Allow up to 3 conditions
#                 col1, col2, col3 = st.columns(3)
#                 attr = col1.selectbox(f"Attribute {i+1}", st.session_state.mutable_atts, key=f"attr_{index}_{i}")
#                 op = col2.selectbox(f"Operation {i+1}", ["=", "!="], key=f"op_{index}_{i}")
#                 val = col3.text_input(f"Value {i+1}", key=f"val_{index}_{i}")
#                 if attr and val:
#                     new_treatment.append((attr, op, val))
#
#             if st.button("Save Treatment", key=f"save_{index}"):
#                 st.session_state.df2.at[index, "Treatment"] = str(new_treatment)
#                 on_change()
#
#     def remove_row(index):
#         """Remove a row from the DataFrame."""
#         st.session_state.df2.drop(index, inplace=True)
#         st.session_state.df2.reset_index(drop=True, inplace=True)
#         on_change()
#
#     def add_row():
#         """Add a new subpopulation row."""
#         new_id = len(st.session_state.df2) + 1
#         new_row = pd.DataFrame([{"ID": new_id, "Treatment": "None", "Subpopulation": "None"}])
#         st.session_state.df2 = pd.concat([st.session_state.df2, new_row], ignore_index=True)
#         on_change()
#
#     def parse_column(value):
#         """Safely parse set or tuple strings from the dataframe."""
#         try:
#             return ast.literal_eval(value)
#         except (SyntaxError, ValueError):
#             return value
#
#     def generate_explanations(group1, group2, outcome, df):
#         """
#         Generates formatted natural language explanations comparing treatment effects between two groups.
#         The response from OpenAI includes HTML styling for Streamlit.
#         """
#         # Convert string columns to actual Python objects
#         df["subpopulation"] = df["subpopulation"].apply(parse_column)
#         df["treatment"] = df["treatment"].apply(parse_column)
#
#         # Define the system prompt with formatting instructions
#         system_message = (
#             "You are an expert data analyst who provides insights in natural language. Your task is to compare the effect of a treatment between two groups in different subpopulations. Format the response using HTML with the following color styles:\n"
#             "- Subpopulation conditions: `<span style='color:orange;'></span>`\n"
#             "- Treatment variables: `<span style='color:blue;'></span>`\n"
#             "- Group 1 (e.g., individuals with more experience): `<span style='background-color:yellow;'></span>`\n"
#             "- Group 2 (e.g., individuals with less experience): `<span style='background-color:purple;color:white;'></span>`\n"
#             "- All other words: `<span style='color:black;'></span>`\n"
#             "The dataframe has the following fields: subpopulation, treatment, ate1, ate2."
#             "Convert subpopulation, treatment, groups and outcome column to descriptive text with proper grammar.\n"
#             "Do not return the values of ate, just specify for who the treatment effects more.\n"
#             "The explanation of a row from the df must be returned in a single line with only the required HTML styling and no additional markdown or explanation.\n"
#             "Here is an example of the format of an explanation that should be returned where group1 is (DevType, 'Analyst') and group2 is (DevType, 'Backend developer'):\n"
#             "For <span style='color:orange;'>individuals who identify as White or of European descent</span>, income growth is more influenced by <span style='color:blue;'>having 24-26 years of coding experience</span> as an <span style='background-color:yellow;'>analyst</span> compared to <span style='background-color:purple;color:white;'>back-end developers</span>."
#         )
#
#         # Construct user prompt
#         user_message = f"Compare the impact of the treatment between these groups:\n"
#         user_message += f"Group 1: {group1} (highlighted in yellow)\n"
#         user_message += f"Group 2: {group2} (highlighted in purple)\n"
#         user_message += f"Outcome: {outcome}\n\n"
#
#         for _, row in df.iterrows():
#             subpop_str = ", ".join(row["subpopulation"])
#             treatment_str = ", ".join([f"{attr} is {val}" for attr, val in row["treatment"]])
#             user_message += (
#                 f"Subpopulation: {subpop_str}\n"
#                 f"Treatment: {treatment_str}\n"
#                 f"Effect for Group 1: {row['ate1']}\n"
#                 f"Effect for Group 2: {row['ate2']}\n\n"
#             )
#
#         user_message += "Generate five natural language explanations using the specified HTML formatting."
#
#         client = openai.OpenAI(api_key=OPEN_AI_API_KEY)
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ]
#         )
#
#         explanations = response.choices[0].message.content.split('\n')
#         explanations = [e for e in explanations if "<span" in e]
#         return explanations
#
#     all_columns = list(st.session_state.df2.columns)  # Get all column names
#     all_columns.insert(0, "explanation")
#     always_visible_cols = ["explanation"]
#     default_selected = [*always_visible_cols, "avg_group1", "avg_group2", "support"]  # Default visible columns
#
#     # Initialize session state for column selections
#     if "selected_columns" not in st.session_state:
#         st.session_state.selected_columns = {col: col in default_selected for col in all_columns}
#
#     # Initialize session state for explanations
#     if "explanations" not in st.session_state:
#         df_mini = st.session_state.df2[['subpopulation', 'treatment', 'ate1', 'ate2']]
#         st.session_state.explanations = generate_explanations(st.session_state.Group_A, st.session_state.Group_B, output_col, df_mini)
#
#     # Create an expander with checkboxes
#     with st.expander("Select columns to display", expanded=False):
#         for col in [col for col in all_columns if col not in always_visible_cols]:
#             st.session_state.selected_columns[col] = st.checkbox(col, value=st.session_state.selected_columns[col])
#
#     # Get the list of currently selected columns
#     selected_cols = [col for col, selected in st.session_state.selected_columns.items() if selected]
#
#     # Display the table row by row with buttons
#     st.write("### Dataset")
#
#     # Ensure enough space for buttons
#     header_cols = st.columns([3] * len(selected_cols) + [3, 2])
#
#     # Display headers
#     for i, col in enumerate(selected_cols):
#         header_cols[i].markdown(f"**{col}**")
#
#     # Display data dynamically based on selected columns
#     for index, row in st.session_state.df2.iterrows():
#         cols = st.columns([3] * len(selected_cols) + [3, 2])
#
#         for i, col in enumerate(selected_cols):
#             if col == "explanation":
#                 cols[i].write(st.session_state.explanations[index], unsafe_allow_html=True)
#             elif col == 'delta':
#                 cols[i].write(f"<span style='background-color:{'#B3FFAE' if row['delta'] >= 0 else '#FF6464'};'>{row[col]}</span>", unsafe_allow_html=True)
#             else:
#                 cols[i].write(row[col])
#
#         # Buttons for each row
#         if cols[len(selected_cols)].button("✏️ Change Treatment", key=f"edit_{index}"):
#             edit_treatment(index)
#         if cols[len(selected_cols) + 1].button("❌ Remove", key=f"remove_{index}"):
#             remove_row(index)
#
#     st.write("---")
#     scores_df = pd.read_csv("demo/5_0.65.csv")
#     scores_row = scores_df.iloc[-1].to_dict()
#     st.markdown(f"<h1><b>Utility</b>: {scores_row['utility']:.2f} <b>Intersection</b>: {scores_row['final_intersection']:.2f} <b>Score</b>: {scores_row['score']:.2f}</h1>", unsafe_allow_html=True)
#
#     # Add new row button
#     st.button("➕ Add Subpopulation", on_click=add_row)
#
#     if st.button("Return to input settings"):
#         switch_page("Input")


