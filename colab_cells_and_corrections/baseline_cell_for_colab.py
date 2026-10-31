# HOW TO USE: paste this as ONE new cell in Flood_model_updated.ipynb, right after the cell that builds `result_table`
# (section 3, "Results on whole maps"). It needs these names from the notebook: regions, PAIRS, REGION_NAMES, FIRST_TARGET,
# apply_water_balance, result_table, save_table. It trains nothing, so it runs in about 5 minutes (480 m rain maps are big).
#
# Tested 4 Oct 2026 on the real data: on UK it gives the same numbers as an independent script
# (trend 0.947, copy + rule 0.053, trend + rule 0.947).

# Simple forecasts to compare the CNN-LSTM with. They are scored exactly like the CNN-LSTM:
# whole maps, 30 minutes ahead, the same test parts, skill against "copy the last map".
#   trend       : the water keeps changing the way it changed in the last 30 minutes (one number, fitted on the training part)
#   copy + rule : copy the last map, then apply the water-balance rule (so the rule alone, without any network)
#   trend + rule: the trend forecast, then the water-balance rule


def fit_trend_factor(region, first_target, last_target):
    """The one number of the trend forecast: how much of the last 30-minute change continues in the next 30 minutes."""

    top = 0.0
    bottom = 0.0

    for step in range(first_target, last_target):
        last = region["depth"][step - 1]
        previous_change = (last - region["depth"][step - 2]).astype("float64")
        next_change = (region["depth"][step] - last).astype("float64")

        top += float(np.sum(previous_change * next_change))
        bottom += float(np.sum(previous_change ** 2))

    return top / bottom


def score_simple_forecasts(region, first_target, last_target, trend_factor):
    """Skill of the simple forecasts over copy-the-last-map, summed over all forecasts of a region."""

    errors = {"copy": 0.0, "trend": 0.0, "copy + rule": 0.0, "trend + rule": 0.0}

    for step in range(first_target, last_target):
        true = region["depth"][step]
        last = region["depth"][step - 1]
        change = last - region["depth"][step - 2]

        # Rain of the 30 minutes before the last map: the only rain a forecast may use (same as in evaluate_region)
        past_rain = region["rain"][step - 2].astype("float32")

        trend = np.maximum(last + trend_factor * change, 0)

        forecasts = {
            "copy": last,
            "trend": trend,
            "copy + rule": apply_water_balance(last, last, past_rain),
            "trend + rule": apply_water_balance(trend, last, past_rain),
        }

        for name, forecast in forecasts.items():
            errors[name] += float(np.sum((forecast - true) ** 2, dtype="float64"))

    return {name: round(1 - errors[name] / errors["copy"], 3) for name in errors if name != "copy"}


simple_rows = []

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]

    # The trend number comes from the training part of the training flood only
    trend_factor = fit_trend_factor(train_region, FIRST_TARGET, train_region["train_end"])

    parts = [
        (pair["train"], "unseen time (last 15%)", train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], "never-seen region", FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, part, first_target, last_target in parts:
        skills = score_simple_forecasts(regions[dataset], first_target, last_target, trend_factor)
        cnn_row = result_table[result_table["Region"] == REGION_NAMES[dataset]].iloc[0]

        simple_rows.append({
            "Region": REGION_NAMES[dataset],
            "Tested on": part,
            "Trend number": round(trend_factor, 3),
            "Skill: trend": skills["trend"],
            "Skill: trend + rule": skills["trend + rule"],
            "Skill: copy + rule": skills["copy + rule"],
            "Skill: CNN-LSTM": cnn_row["CNN-LSTM: skill"],
            "Skill: CNN-LSTM + rule": cnn_row["CNN-LSTM + water rule: skill"],
        })

simple_table = pd.DataFrame(simple_rows)
display(simple_table)
save_table(simple_table, "03_simple_forecasts")
