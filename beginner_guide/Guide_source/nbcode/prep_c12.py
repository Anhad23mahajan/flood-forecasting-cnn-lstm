def get_seconds(file_path):
    # Turn a depth file name like "1800.tif" into the number 1800
    return int(os.path.basename(file_path).replace(".tif", ""))


def get_depth_files(dataset):
    # The 30-minute depth maps of one dataset, in time order
    folder = os.path.join(RAW_DIR, DATASETS[dataset]["folder"])
    files = glob.glob(os.path.join(folder, "*.tif"))

    # Keep only maps on a 30-minute mark
    files = [f for f in files if get_seconds(f) % STEP_SECONDS == 0]
    files.sort(key=get_seconds)
    return files


def get_rain_files(region):
    # The rain maps of one region, in time order (names are date and time, so alphabetical order works)
    folder = os.path.join(RAW_DIR, "Relevant data", "Rainfall", region + " flood")
    files = glob.glob(os.path.join(folder, "*.tif"))
    files.sort()
    return files


def dem_path(region):
    # File with the ground height of a region
    return os.path.join(RAW_DIR, "Relevant data", "DEM", region + "_DEM.tif")


def land_cover_path(region):
    # File with the land cover of a region
    return os.path.join(RAW_DIR, "Relevant data", "Land use and land cover", region + ".tif")


def read_on_depth_grid(file_path, dataset, rows, columns, method):
    # Read the part of a 30 m map under the depth map, shrunk onto the depth map's grid with the given method
    info = DATASETS[dataset]
    scale = info["scale"]

    # The area of the 30 m map that the depth map covers
    window = Window(col_off=info["col_offset"], row_off=info["row_offset"],
                    width=columns * scale, height=rows * scale)

    with rasterio.open(file_path) as source:
        data = source.read(1, window=window, out_shape=(rows, columns), resampling=method)

    data = data.astype("float32")

    # These files use a huge negative number to mean "no data". None should be under the depth map.
    if np.any(data < -1e30):
        raise ValueError("No-data values found under the depth map in " + file_path)

    return data


def land_cover_positions(land_cover):
    # Turn land-cover codes (1, 2, 4, ...) into positions (0, 1, 2, ...) so each gets its own colour
    positions = np.full(land_cover.shape, np.nan)
    for position, code in enumerate(LAND_COVER.keys()):
        positions[land_cover == code] = position
    return positions


# Colours used in the plots
land_cover_colours = ListedColormap([colour for name, colour in LAND_COVER.values()])
land_cover_names = [name for name, colour in LAND_COVER.values()]

# Blue colours that start at light blue instead of white, so shallow water stays visible; dry squares grey
water_colours = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
water_colours.set_bad("lightgrey")
