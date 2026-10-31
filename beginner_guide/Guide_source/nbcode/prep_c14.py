DEPTH_SHAPES = {}
size_rows = []

for dataset, info in DATASETS.items():
    region = info["region"]

    # Size of the depth map
    with rasterio.open(get_depth_files(dataset)[0]) as source:
        rows = source.height
        columns = source.width
    DEPTH_SHAPES[dataset] = (rows, columns)

    # Area of the 30 m grid that the depth map needs
    needed_rows = info["row_offset"] + rows * info["scale"]
    needed_columns = info["col_offset"] + columns * info["scale"]

    layers = {"Ground height": dem_path(region), "Land cover": land_cover_path(region), "Rain": get_rain_files(region)[0]}

    for layer_name, path in layers.items():
        with rasterio.open(path) as source:
            fits = needed_rows <= source.height and needed_columns <= source.width
            layer_size = str(source.height) + " x " + str(source.width)

        size_rows.append({"Dataset": dataset, "Layer": layer_name, "Layer size": layer_size,
                          "Area needed": str(needed_rows) + " x " + str(needed_columns), "Big enough?": fits})

display(pd.DataFrame(size_rows))
