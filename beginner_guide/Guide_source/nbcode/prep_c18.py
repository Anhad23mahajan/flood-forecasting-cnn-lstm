def read_full_map(file_path):
    # Read a whole 30 m map. No-data values become the map's average so they don't disturb the tests
    with rasterio.open(file_path) as source:
        data = source.read(1).astype("float32")
    no_data = data < -1e30
    data[no_data] = data[~no_data].mean()
    return data


def rivers_map(region):
    # 1 where the land-cover map shows a river or lake, 0 everywhere else
    return (read_full_map(land_cover_path(region)) == 1).astype("float32")


def valleys_map(region, scale):
    # How much lower each 30 m square is than the ground around it (river channels and valleys score high)
    ground = read_full_map(dem_path(region))

    # Average ground height in a square around each point, about 10 depth squares wide
    surroundings = uniform_filter(ground, size=10 * scale + 1)
    return surroundings - ground


def shift_scores(dataset, reference):
    # Move a 30 m reference map a little in every direction and measure how well it matches the average depth
    info = DATASETS[dataset]
    scale = info["scale"]
    rows, columns = DEPTH_SHAPES[dataset]

    # Leave out a 3-square border of the depth map, so every move stays inside the reference map
    border = 3
    depth_inner = depth_averages[dataset][border:rows - border, border:columns - border]
    inner_rows, inner_columns = depth_inner.shape

    # Moves of up to 3 depth squares in every direction
    largest_shift = border * scale
    shift_step = max(1, scale // 4)
    shifts = list(range(-largest_shift, largest_shift + 1, shift_step))

    scores = np.full((len(shifts), len(shifts)), np.nan)

    for i, row_shift in enumerate(shifts):
        for j, column_shift in enumerate(shifts):
            top = info["row_offset"] + border * scale + row_shift
            left = info["col_offset"] + border * scale + column_shift
            bottom = top + inner_rows * scale
            right = left + inner_columns * scale

            # Skip moves that would go past the edge of the reference map
            if top < 0 or left < 0 or bottom > reference.shape[0] or right > reference.shape[1]:
                continue

            # Shrink the moved piece onto the depth grid and compare it with the average depth
            piece = reference[top:bottom, left:right]
            piece_on_depth_grid = piece.reshape(inner_rows, scale, inner_columns, scale).mean(axis=(1, 3))
            scores[i, j] = np.corrcoef(piece_on_depth_grid.ravel(), depth_inner.ravel())[0, 1]

    return scores, shifts
