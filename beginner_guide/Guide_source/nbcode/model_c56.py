# Accuracy with and without the rules
rule_rows = []

for dataset in test_tiles:
    data = test_tiles[dataset]
    rule_rows.append({
        "Region": REGION_NAMES[dataset],
        "Test tiles": len(data["labels"]),
        "Capsule network alone (%)": round(100 * np.mean(data["capsule_classes"] == data["labels"]), 1),
        "Class of the forecast map alone (%)": round(100 * np.mean(data["forecast_classes"] == data["labels"]), 1),
        "Capsule network + rules (%)": round(100 * np.mean(data["final_classes"] == data["labels"]), 1),
        "Decisions changed by the rules (%)": round(100 * np.mean(data["final_classes"] != data["capsule_classes"]), 1),
    })

rule_table = pd.DataFrame(rule_rows)
display(rule_table)
save_table(rule_table, "07_neurosymbolic_accuracy")
