def load_region(dataset):
    # Load one flood from Drive and prepare its layers
    folder = PREPARED_DIR + "/" + dataset

    # Water depth and rain for every 30-minute step (missing values become 0)
    depth = np.nan_to_num(np.load(folder + "/depth.npy"), copy=False)
    rain = np.nan_to_num(np.load(folder + "/rain.npy"), copy=False)

    # Ground height, with missing squares set to the average height
    terrain = np.load(folder + "/terrain.npy").astype("float32")
    terrain[np.isnan(terrain)] = np.nanmean(terrain)

    # Local ground height: positive = higher than the area around it, negative = a dip
    local_height = terrain - uniform_filter(terrain, size=TERRAIN_WINDOW)
    local_height = local_height / max(local_height.std(), 1e-6)

    # Rivers and lakes (so the severity only counts land that is normally dry)
    permanent_water = np.load(folder + "/permanent_water.npy").astype(bool)

    # Map details
    with open(folder + "/map_info.json") as file:
        info = json.load(file)

    number_of_maps = depth.shape[0]

    region = {}
    region["depth"] = depth
    region["rain"] = rain
    region["local_height"] = local_height.astype("float32")
    region["dry_land"] = ~permanent_water
    region["info"] = info
    region["number_of_maps"] = number_of_maps
    region["train_end"] = int(TRAIN_SHARE * number_of_maps)
    region["validation_end"] = int((TRAIN_SHARE + VALIDATION_SHARE) * number_of_maps)
    return region
