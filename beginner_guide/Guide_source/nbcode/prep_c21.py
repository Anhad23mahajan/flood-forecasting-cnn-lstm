for dataset, info in DATASETS.items():
    region = info["region"]
    rows, columns = DEPTH_SHAPES[dataset]
    dataset_dir = os.path.join(SAVE_DIR, dataset)

    # Ground height: the average height inside each depth square
    terrain = read_on_depth_grid(dem_path(region), dataset, rows, columns, Resampling.average)
    np.save(os.path.join(dataset_dir, "terrain.npy"), terrain)

    # Land cover is a category, so each depth square takes the most common category inside it
    land_cover = read_on_depth_grid(land_cover_path(region), dataset, rows, columns, Resampling.mode).astype("uint8")
    np.save(os.path.join(dataset_dir, "land_cover.npy"), land_cover)

    # Rivers and lakes that are there even without a flood
    permanent_water = land_cover == 1
    np.save(os.path.join(dataset_dir, "permanent_water.npy"), permanent_water)

    # Rain on the depth grid, one map per 30 minutes.
    # Stored as float16 to halve the file size: rain stays below about 20 mm per hour, where float16 is accurate to about 0.02 mm per hour.
    number_of_maps = len(get_depth_files(dataset))
    rain_files = get_rain_files(region)
    rain = np.zeros((number_of_maps, rows, columns), dtype="float16")
    missing_rain = 0

    for step in range(number_of_maps):
        if step < len(rain_files):
            rain[step] = read_on_depth_grid(rain_files[step], dataset, rows, columns, Resampling.average)
        else:
            missing_rain += 1

        if step % 100 == 0:
            print(dataset, "- rain map", step, "of", number_of_maps)

    np.save(os.path.join(dataset_dir, "rain.npy"), rain)
    print(dataset, "- saved. Rain maps missing at the end (left as 0):", missing_rain)
