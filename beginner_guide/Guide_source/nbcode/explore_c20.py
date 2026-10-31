# Rain and water measurements every 30 minutes, for every dataset
measurements = {}

for dataset in DEPTH_FOLDERS:
    region = REGION_OF[dataset]

    # One depth map every 30 minutes, the same spacing as the rain maps
    depth_files = get_depth_files(dataset)[::DEPTH_MAPS_PER_RAIN_MAP]
    rain_files = get_rain_files(region)

    average_depth = []
    deepest_point = []
    wet_area = []
    average_rain = []
    impossible_values = 0

    for step, depth_file in enumerate(depth_files):
        depth = read_map(depth_file)

        # Values that should never exist: missing values or negative depth
        impossible_values += int(np.isnan(depth).sum())
        impossible_values += int((depth < 0).sum())

        average_depth.append(np.nanmean(depth))
        deepest_point.append(np.nanmax(depth))

        # Share of the map with more than 1 cm of water
        wet_area.append(np.nanmean(depth > WET_DEPTH_M))

        # Average rain in the rain map with the same step number
        if step < len(rain_files):
            rain = read_map(rain_files[step], max_size=400)
            average_rain.append(np.nanmean(rain))
        else:
            average_rain.append(np.nan)

    measurements[dataset] = {
        "hours": np.arange(len(depth_files)) * 0.5,
        "average_depth": np.array(average_depth),
        "deepest_point": np.array(deepest_point),
        "wet_area": np.array(wet_area),
        "average_rain": np.array(average_rain),
    }

    print(dataset, "- maps checked:", len(depth_files), "- impossible values found:", impossible_values)
