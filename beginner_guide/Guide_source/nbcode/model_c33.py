def fit_trend_factor(region, first_target, last_target):
    # Find beta: how much of the last 30-minute change carries on into the next 30 minutes
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
    # Skill of the simple forecasts against copying the last map
    errors = {"copy": 0.0, "trend": 0.0, "copy + rule": 0.0, "trend + rule": 0.0}

    for step in range(first_target, last_target):
        true = region["depth"][step]
        last = region["depth"][step - 1]
        change = last - region["depth"][step - 2]

        # Rain of the 30 minutes before the last map
        past_rain = region["rain"][step - 2].astype("float32")

        trend = np.maximum(last + trend_factor * change, 0)

        forecasts = {
            "copy": last,
            "trend": trend,
            "copy + rule": apply_water_balance(last, last, past_rain),
            "trend + rule": apply_water_balance(trend, last, past_rain),
        }

        for name in forecasts:
            errors[name] += float(np.sum((forecasts[name] - true) ** 2, dtype="float64"))

    skills = {}
    for name in ["trend", "copy + rule", "trend + rule"]:
        skills[name] = round(1 - errors[name] / errors["copy"], 3)
    return skills


# Fit beta for each model, on the training part of its training flood
trend_factors = {}

for pair in PAIRS:
    train_region = regions[pair["train"]]
    trend_factors[pair["name"]] = fit_trend_factor(train_region, FIRST_TARGET, train_region["train_end"])

print("Trend numbers (beta):", trend_factors)
