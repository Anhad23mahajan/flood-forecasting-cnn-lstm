coordinate_rows = []

for dataset, info in DATASETS.items():
    region = info["region"]
    rows, columns = DEPTH_SHAPES[dataset]
    scale = info["scale"]

    # A world file (.tfw) describes the 30 m grid in 6 numbers:
    # square width, two rotation terms, square height (negative), and the x and y of the top-left square's centre
    tfw_path = os.path.join(RAW_DIR, "Relevant data", "DEM", region + "_DEM.tfw")
    with open(tfw_path) as file:
        numbers = [float(line) for line in file if line.strip()]
    cell_width, rotation_a, rotation_b, cell_height, centre_x, centre_y = numbers

    # Top-left corner of the 30 m grid (the world file gives the centre of the first square, so move half a square)
    grid_left = centre_x - cell_width / 2
    grid_top = centre_y - cell_height / 2

    # Top-left corner of the depth map inside the 30 m grid
    left = grid_left + info["col_offset"] * cell_width
    top = grid_top + info["row_offset"] * cell_height

    map_info = {
        "dataset": dataset,
        "region": region,
        "resolution_m": info["resolution_m"],
        "rows": rows,
        "columns": columns,
        "number_of_maps": len(get_depth_files(dataset)),
        "step_seconds": STEP_SECONDS,
        "epsg": EPSG_CODES[region],
        "transform": [cell_width * scale, 0.0, left, 0.0, cell_height * scale, top],
        "rain_meaning": "rain[t] is the rain in mm per hour falling between depth map t and depth map t+1",
    }

    with open(os.path.join(SAVE_DIR, dataset, "map_info.json"), "w") as file:
        json.dump(map_info, file, indent=2)

    coordinate_rows.append({"Dataset": dataset, "Square size from world file (m)": abs(cell_width * scale),
                            "Expected square size (m)": info["resolution_m"], "Map rotated?": rotation_a != 0 or rotation_b != 0})

display(pd.DataFrame(coordinate_rows))
