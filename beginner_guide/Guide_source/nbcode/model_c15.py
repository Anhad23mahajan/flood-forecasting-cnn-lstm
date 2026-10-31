def make_test_tiles(region, first_target, last_target, max_count):
    # Cut the maps into a regular grid of 64 x 64 tiles and pick some of them at random
    rows = region["depth"].shape[1]
    columns = region["depth"].shape[2]

    # Every (time, tile) pair
    all_positions = []
    for target_step in range(first_target, last_target):
        for top, left in cut_into_tiles(rows, columns):
            all_positions.append((target_step, top, left))

    # Pick max_count of them at random
    picked = np.random.permutation(len(all_positions))[:max_count]

    positions = []
    inputs = []
    targets = []

    for number in picked:
        target_step, top, left = all_positions[number]
        positions.append((target_step, top, left))
        inputs.append(make_input(region, target_step - 1, top, left, TILE_SIZE, TILE_SIZE))
        targets.append(region["depth"][target_step, top:top + TILE_SIZE, left:left + TILE_SIZE, None])

    return np.array(inputs), np.array(targets, dtype="float32"), positions
