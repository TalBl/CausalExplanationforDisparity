from algorithms.final_algorithm.full import acs, meps, so

for name, dataset in [['acs', acs], ['so', so], ['meps', meps]]:
    lines = []
    for att1 in dataset.subpopulations_atts:
        for att2 in dataset.treatments_atts:
            lines.append(f"'{att1} -> {att2};',\n")
    for att in dataset.treatments_atts:
        lines.append(f"'{att} -> {dataset.outcome_col};',\n")
    with open(f"outputs/dags/causal_graph_default_{name}.dot", "w") as f:
        f.writelines(lines)

