# Limits used by the rules
ALMOST_DRY_SHARE = 0.01
RAIN_RULE_MM_PER_HOUR = 1.0


def apply_rules(network_class, share_now, share_forecast, rain_last_hour):
    # Check the network's class with the three rules; return the final class and the reason

    now_class = severity_class(share_now)
    forecast_class = severity_class(share_forecast)

    # Rule 1: no water, no flood
    if share_now < ALMOST_DRY_SHARE and share_forecast < ALMOST_DRY_SHARE:
        return 0, "Rule 1: under 1% of the land flooded now and in the forecast, so None"

    # Rule 2: rising water during rain can't be less severe than now
    if share_forecast >= share_now and rain_last_hour > RAIN_RULE_MM_PER_HOUR and network_class < now_class:
        return now_class, "Rule 2: water rising during rain, so at least " + SEVERITY_NAMES[now_class] + " (the current level)"

    # Rule 3: the network and the forecast map are more than one class apart
    if abs(network_class - forecast_class) > 1:
        return forecast_class, ("Rule 3: network said " + SEVERITY_NAMES[network_class] + " but the forecast map says "
                                + SEVERITY_NAMES[forecast_class] + ", so the forecast map wins")

    return network_class, "No rule needed: the network's answer agrees with the rules"
