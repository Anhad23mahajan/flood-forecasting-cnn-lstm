# Apply the rules to every test tile
for dataset in test_tiles:
    data = test_tiles[dataset]
    region = regions[dataset]

    final_classes = []
    forecast_classes = []
    reasons = []

    for tile in range(len(data["positions"])):
        target_step, top, left = data["positions"][tile]
        dry_land = region["dry_land"][top:top + TILE_SIZE, left:left + TILE_SIZE]

        # Facts the rules use
        share_now = flooded_share(data["inputs"][tile, -1, :, :, 0], dry_land)
        share_forecast = flooded_share(data["forecast"][tile, :, :, 0], dry_land)
        rain_last_hour = data["inputs"][tile, -2:, :, :, 1].mean() * RAIN_SCALE

        final_class, reason = apply_rules(data["capsule_classes"][tile], share_now, share_forecast, rain_last_hour)

        final_classes.append(final_class)
        forecast_classes.append(severity_class(share_forecast))
        reasons.append(reason)

    data["final_classes"] = np.array(final_classes)
    data["forecast_classes"] = np.array(forecast_classes)
    data["reasons"] = reasons
