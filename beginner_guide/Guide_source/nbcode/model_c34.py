# Score the simple forecasts on the same test parts as Model A
simple_rows = []

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]

    parts = [
        (pair["train"], "unseen time (last 15%)", train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], "never-seen region", FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, part, first_target, last_target in parts:
        skills = score_simple_forecasts(regions[dataset], first_target, last_target, trend_factors[pair["name"]])
        model_a_row = result_table[result_table["Region"] == REGION_NAMES[dataset]].iloc[0]

        simple_rows.append({
            "Region": REGION_NAMES[dataset],
            "Tested on": part,
            "Trend number": round(trend_factors[pair["name"]], 3),
            "Skill: trend": skills["trend"],
            "Skill: trend + rule": skills["trend + rule"],
            "Skill: copy + rule": skills["copy + rule"],
            "Skill: Model A": model_a_row["CNN-LSTM: skill"],
            "Skill: Model A + rule": model_a_row["CNN-LSTM + water rule: skill"],
        })

simple_table = pd.DataFrame(simple_rows)
display(simple_table)
save_table(simple_table, "04_simple_forecasts")
