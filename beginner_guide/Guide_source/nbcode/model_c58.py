# Some example decisions with their reasons (never-seen floods)
example_rows = []

for pair in PAIRS:
    data = test_tiles[pair["test"]]

    # First the tiles where a rule changed the answer, then a few others
    changed = np.where(data["final_classes"] != data["capsule_classes"])[0]
    unchanged = np.where(data["final_classes"] == data["capsule_classes"])[0]
    chosen = list(changed[:3]) + list(unchanged[:2])

    for tile in chosen:
        example_rows.append({
            "Region": REGION_NAMES[pair["test"]],
            "Hours into flood": data["positions"][tile][0] * STEP_HOURS,
            "True severity": SEVERITY_NAMES[data["labels"][tile]],
            "Capsule network": SEVERITY_NAMES[data["capsule_classes"][tile]],
            "Final decision": SEVERITY_NAMES[data["final_classes"][tile]],
            "Reason": data["reasons"][tile],
        })

example_table = pd.DataFrame(example_rows)
pd.set_option("display.max_colwidth", None)
display(example_table)
save_table(example_table, "07_example_decisions")
