from collections import defaultdict

def parse_edges(file_path):
    """Parse edges from a file where each edge line is wrapped in double quotes and ends with a comma."""
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().replace('"', '').strip(',')  # Remove double quotes and trailing comma
            if '->' in line:
                source, target = line.strip(';').split(' -> ')
                edges.append((source, target))
    return edges

def remove_vertices_and_bypass(edges, vertices_to_remove):
    """Remove edges involving specific vertices and create bypassing edges."""
    adjacency_list = defaultdict(list)
    for source, target in edges:
        adjacency_list[source].append(target)

    new_edges = []  # Store the bypassing edges

    for vertex in vertices_to_remove:
        # Identify all sources and targets for the current vertex
        sources = [src for src, targets in adjacency_list.items() if vertex in targets]
        targets = adjacency_list.get(vertex, [])

        # Create bypassing edges
        for src in sources:
            for tgt in targets:
                if tgt not in vertices_to_remove:  # Only create valid bypassing edges
                    new_edges.append((src, tgt))

        # Remove all references to the current vertex in the adjacency list
        for src in sources:
            adjacency_list[src].remove(vertex)
        if vertex in adjacency_list:
            del adjacency_list[vertex]

    # Combine the original edges with the bypassing edges
    # and exclude any edges that were removed
    updated_edges = [
        (src, tgt) for src, tgt in edges
        if src not in vertices_to_remove and tgt not in vertices_to_remove
    ]
    updated_edges.extend(new_edges)

    return updated_edges

def write_edges(file_path, edges):
    """Write edges to a file in the specified format, with each line wrapped in double quotes and ending with a comma."""
    with open(file_path, 'w') as file:
        for source, target in edges:
            file.write(f'"{source} -> {target};",\n')

# Example Usage
input_file = "data/acs/causal_dag.txt"  # Replace with your input file name
output_file = "data/acs/updated_graph.txt"  # Replace with your desired output file name

# vertices_to_remove = ["B"]  # Replace with the vertices to remove
vertices_to_remove = ['TRICARE or other military health care', 'Usual hours worked per week past 12 months', 'Mobility status (lived here 1 year ago)', 'When last worked', "Person's weight replicate 5", 'Ancestry recode', 'Raw labor-force status', 'Number of times married', 'Hispanic, Detailed', 'Married in the past 12 months', 'Retirement income past 12 months', 'Medicare, for people 65 and older, or people with certain disabilities', 'Independent living difficulty', "Person's weight replicate 2", 'Ancestry recode - second entry', "Person's weight replicate 1", 'person weight', 'Adjustment factor for income and earnings dollar amounts', "Person's weight replicate 3", "Person's weight replicate 4", 'Social Security income past 12 months', 'Ambulatory difficulty', 'Public assistance income past 12 months', 'Self-employment income past 12 months', 'Self-care difficulty', 'All other income past 12 months', 'Interest, dividends, and net rental income past 12 months', 'Divorced in the past 12 months', 'Quarter of birth', 'Supplementary Security Income past 12 months', 'Year last married', 'Married, spouse present/spouse absent', 'Georgraphic division', 'Relationship to reference person', 'Available for Work', 'Looking for work', 'Ancestry recode - first entry', 'Widowed in the past 12 months', 'On layoff from work', "Total person's income", 'VA (Health Insurance through VA Health Care)', 'Weeks worked during past 12 months', 'Medicaid, Medical Assistance, or any kind of government-assistance plan for those with low incomes or a disability']

# Parse edges from the file
edges = parse_edges(input_file)

# Remove vertices and create bypassing edges
updated_edges = remove_vertices_and_bypass(edges, vertices_to_remove)

# Write updated edges to a new file
write_edges(output_file, updated_edges)

print(f"Updated edges written to {output_file}.")
