# Depth maps: how many, how many hours, how big, and are they evenly spaced?
depth_rows = []

for dataset in DEPTH_FOLDERS:
    files = get_depth_files(dataset)
    seconds = [get_seconds(f) for f in files]

    # Every gap between two maps should be exactly 5 minutes
    evenly_spaced = bool(np.all(np.diff(seconds) == DEPTH_STEP_SECONDS))

    with rasterio.open(files[0]) as source:
        map_size = str(source.height) + " x " + str(source.width)

    depth_rows.append({
        "Dataset": dataset,
        "Number of depth maps": len(files),
        "Hours covered": (seconds[-1] - seconds[0]) / 3600,
        "Map size (rows x columns)": map_size,
        "Every 5 minutes?": evenly_spaced,
    })

print("DEPTH MAPS")
display(pd.DataFrame(depth_rows))

# Rain maps of each region
rain_rows = []

for region in REGIONS:
    files = get_rain_files(region)

    with rasterio.open(files[0]) as source:
        map_size = str(source.height) + " x " + str(source.width)

    rain_rows.append({
        "Region": region,
        "Number of rain maps": len(files),
        "Hours covered": (len(files) - 1) * RAIN_STEP_SECONDS / 3600,
        "Map size (rows x columns)": map_size,
    })

print("RAIN MAPS")
display(pd.DataFrame(rain_rows))
