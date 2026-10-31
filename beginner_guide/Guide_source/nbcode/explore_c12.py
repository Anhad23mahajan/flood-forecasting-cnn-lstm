def get_seconds(file_path):
    # Turn a depth file name like "300.tif" into the number 300
    file_name = os.path.basename(file_path)
    return int(file_name.replace(".tif", ""))


def get_depth_files(dataset):
    # The depth maps of one dataset, in time order
    folder = os.path.join(DATA_DIR, DEPTH_FOLDERS[dataset])
    files = glob.glob(os.path.join(folder, "*.tif"))

    # Each file is named after its time in seconds, so sort by that number
    files.sort(key=get_seconds)
    return files


def get_rain_files(region):
    # The rain maps of one region, in time order
    folder = os.path.join(DATA_DIR, "Relevant data", "Rainfall", region + " flood")
    files = glob.glob(os.path.join(folder, "*.tif"))

    # Rain files are named by date and time (e.g. 20151204-S120000.tif), so alphabetical order is time order
    files.sort()
    return files


def read_map(file_path, max_size=None):
    # Read one map file into a 2D array. With max_size, big maps are shrunk so they load and plot faster
    with rasterio.open(file_path) as source:
        if max_size is None:
            data = source.read(1)
        else:
            # Shrink the map so its longest side is about max_size
            longest_side = max(source.height, source.width)
            shrink = max(1, int(np.ceil(longest_side / max_size)))
            data = source.read(1, out_shape=(source.height // shrink, source.width // shrink))

    data = data.astype("float32")

    # These files mark "no data" with a huge negative number, so turn those into NaN
    data[data < -1e30] = np.nan
    return data
