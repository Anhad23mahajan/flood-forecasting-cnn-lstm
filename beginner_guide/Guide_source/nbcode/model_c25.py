def critical_success_index(predicted, true):
    # Of the squares flooded in the forecast or in reality, the share where both agree
    predicted_flooded = predicted > FLOOD_DEPTH_M
    true_flooded = true > FLOOD_DEPTH_M

    hits = np.sum(predicted_flooded & true_flooded)
    misses = np.sum(~predicted_flooded & true_flooded)
    false_alarms = np.sum(predicted_flooded & ~true_flooded)

    # Nothing flooded anywhere: the score is undefined, so this step is skipped
    if hits + misses + false_alarms == 0:
        return np.nan

    return hits / (hits + misses + false_alarms)


def evaluate_region(forecaster, region, first_target, last_target):
    # Forecast every whole map, with and without the water rule, and compare with copying
    rows = []
    example = None
    biggest_change = -1.0

    for target_step in range(first_target, last_target):
        model_input = make_input(region, target_step - 1)[None]
        predicted = forecaster(model_input, training=False).numpy()[0, :, :, 0]

        true = region["depth"][target_step]
        last = region["depth"][target_step - 1]

        # Rain of the last 30 minutes (back in mm per hour) and the corrected forecast
        past_rain = model_input[0, -1, :, :, 1] * RAIN_SCALE
        corrected = apply_water_balance(predicted, last, past_rain)

        rows.append({
            "hours": target_step * STEP_HOURS,
            "model_squared_error": float(np.sum((predicted - true) ** 2)),
            "rule_squared_error": float(np.sum((corrected - true) ** 2)),
            "copy_squared_error": float(np.sum((last - true) ** 2)),
            "cells": true.size,
            "model_csi": critical_success_index(predicted, true),
            "rule_csi": critical_success_index(corrected, true),
            "copy_csi": critical_success_index(last, true),
        })

        # Keep the step where the water changes the most, for the pictures
        total_change = float(np.abs(true - last).sum())
        if total_change > biggest_change:
            biggest_change = total_change
            example = {"step": target_step, "last": last, "true": true, "predicted": predicted, "corrected": corrected}

    return pd.DataFrame(rows), example
