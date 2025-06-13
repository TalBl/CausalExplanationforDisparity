import re

def convert_graph(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    label_map = {}

    # Step 1: Build node label map from lines like: 0 [label="real name"];
    for line in lines:
        match = re.match(r'^(\d+)\s+\[label=(.+?)\]', line)
        if match:
            node_id = match.group(1)
            label = match.group(2)
            label_map[node_id] = label

    # Step 2: Convert edge definitions to 'Label1 -> Label2;'
    output_lines = []
    for line in lines:
        match = re.match(r'^(\d+)\s*->\s*(\d+)', line)
        if match:
            source_id, target_id = match.groups()
            source_label = label_map.get(source_id, f"X{source_id}")
            target_label = label_map.get(target_id, f"X{target_id}")
            formatted = f"'{source_label} -> {target_label};'\n"
            output_lines.append(formatted)
    with open(f"outputs/dags/{file_name}","w") as f:
        f.writelines(output_lines)

for file in ["causal_graph_fci_meps.dot"]:
    convert_graph(file)
