# One table with the main numbers
summary_rows = []

for dataset in REGION_NAMES:
    forecast_row = v2_table[v2_table["Region"] == REGION_NAMES[dataset]].iloc[0]
    rule_row = rule_table[rule_table["Region"] == REGION_NAMES[dataset]].iloc[0]

    summary_rows.append({
        "Region": REGION_NAMES[dataset],
        "Tested on": forecast_row["Tested on"],
        "Skill: trend forecast": forecast_row["Skill: trend"],
        "Skill: Model A": forecast_row["Skill: Model A"],
        "Skill: Model B": forecast_row["Skill: Model B"],
        "Severity: capsule (%)": rule_row["Capsule network alone (%)"],
        "Severity: capsule + rules (%)": rule_row["Capsule network + rules (%)"],
    })

summary_table = pd.DataFrame(summary_rows)
display(summary_table)
save_table(summary_table, "08_summary")
