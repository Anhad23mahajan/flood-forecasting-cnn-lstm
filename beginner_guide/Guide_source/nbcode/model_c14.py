def make_training_tiles(region, first_target, last_target, count):
    # Cut random 64 x 64 tiles at random times; the answer is the depth map 30 minutes later
    rows = region["depth"].shape[1]
    columns = region["depth"].shape[2]

    inputs = []
    targets = []
    positions = []

    while len(inputs) < count:
        # Pick a random time and a random place
        target_step = np.random.randint(first_target, last_target)
        top = np.random.randint(0, rows - TILE_SIZE + 1)
        left = np.random.randint(0, columns - TILE_SIZE + 1)

        target = region["depth"][target_step, top:top + TILE_SIZE, left:left + TILE_SIZE]
        last = region["depth"][target_step - 1, top:top + TILE_SIZE, left:left + TILE_SIZE]

        # Skip some calm tiles (with CALM_TILE_SHARE = 1.0 nothing is skipped)
        water_changes = np.abs(target - last).max() > CHANGE_LIMIT_M
        if not water_changes and np.random.rand() > CALM_TILE_SHARE:
            continue

        inputs.append(make_input(region, target_step - 1, top, left, TILE_SIZE, TILE_SIZE))
        targets.append(target[:, :, None])
        positions.append((target_step, top, left))

    return np.array(inputs), np.array(targets, dtype="float32"), positions
