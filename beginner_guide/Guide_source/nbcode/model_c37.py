# Test Model B on whole maps, and keep the trend forecast's errors for the plots
v2_evaluations = {}
v2_examples = {}
trend_squared_errors = {}
v2_rows = []

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]
    trend_factor = trend_factors[pair["name"]]

    parts = [
        (pair["train"], "unseen time (last 15%)", train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], "never-seen region", FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, part, first_target, last_target in parts:
        region = regions[dataset]

        # Model B on whole maps
        whole_map_model = v2_for_whole_map(v2_forecasters[pair["name"]], region, trend_factor)
        v2_evaluations[dataset], v2_examples[dataset] = evaluate_region(whole_map_model, region, first_target, last_target)
        del whole_map_model

        # Error of the trend forecast at every step
        trend_errors = []
        for step in range(first_target, last_target):
            last = region["depth"][step - 1]
            trend = np.maximum(last + trend_factor * (last - region["depth"][step - 2]), 0)
            trend_errors.append(float(np.sum((trend - region["depth"][step]) ** 2, dtype="float64")))
        trend_squared_errors[dataset] = np.array(trend_errors)

        evaluation = v2_evaluations[dataset]
        copy_error = evaluation["copy_squared_error"].sum()
        simple_row = simple_table[simple_table["Region"] == REGION_NAMES[dataset]].iloc[0]

        v2_rows.append({
            "Region": REGION_NAMES[dataset],
            "Tested on": part,
            "Skill: trend": simple_row["Skill: trend"],
            "Skill: Model A": simple_row["Skill: Model A"],
            "Skill: Model B": round(1 - evaluation["model_squared_error"].sum() / copy_error, 3),
            "Skill: Model B + rule": round(1 - evaluation["rule_squared_error"].sum() / copy_error, 3),
        })

    print(pair["name"], "done")

v2_table = pd.DataFrame(v2_rows)
display(v2_table)
save_table(v2_table, "04_model_B_results")
