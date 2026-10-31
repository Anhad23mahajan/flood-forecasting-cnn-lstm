def apply_water_balance(predicted, last, past_rain):
    # Shift the forecast so the total rise of the water matches the rain of the last 30 minutes

    predicted_change = predicted - last

    # Rise allowed by the rain, in metres (mm per hour x 0.5 hours / 1000)
    allowed_rise = 0.5 * float(np.mean(past_rain)) / 1000

    # Extra water the network added (positive) or rain water it missed (negative)
    extra_water = (float(np.mean(predicted_change)) - allowed_rise) * predicted.size

    corrected = predicted.copy()

    if extra_water < 0:
        # Missing rain water: rain falls everywhere, so add it to every square
        corrected = corrected - extra_water / predicted.size
    else:
        # Extra water: take it away only from squares that have water
        wet = corrected > WET_DEPTH_M
        if wet.sum() > 0:
            corrected[wet] = corrected[wet] - extra_water / wet.sum()

    # Depth can't be negative
    return np.maximum(corrected, 0)
