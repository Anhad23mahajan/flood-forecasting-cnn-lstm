# How often each rule made the final decision
reason_rows = []

for dataset in test_tiles:
    reasons = test_tiles[dataset]["reasons"]

    # The rule name is the text before the ":" in each reason
    rule_names = []
    for reason in reasons:
        rule_names.append(reason.split(":")[0])

    row = {"Region": REGION_NAMES[dataset]}
    for rule in ["No rule needed", "Rule 1", "Rule 2", "Rule 3"]:
        row[rule] = rule_names.count(rule)
    reason_rows.append(row)

reason_table = pd.DataFrame(reason_rows)
display(reason_table)
save_table(reason_table, "07_rules_used")
