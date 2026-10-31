def make_input(region, last_step, top=0, left=0, height=None, width=None):
    # Build one model input: the 4 maps up to last_step, each with depth, past rain and ground height
    # Without a tile position, the whole map is used

    if height is None:
        height = region["depth"].shape[1]
        width = region["depth"].shape[2]

    maps = []

    for step in range(last_step - INPUT_STEPS + 1, last_step + 1):
        depth = region["depth"][step, top:top + height, left:left + width]

        # rain[step - 1] is the rain that fell before this map, so it has already happened
        rain = region["rain"][step - 1, top:top + height, left:left + width].astype("float32") / RAIN_SCALE

        height_map = region["local_height"][top:top + height, left:left + width]

        # Stack the 3 layers of this map
        maps.append(np.stack([depth, rain, height_map], axis=-1))

    # Shape: (4 maps, height, width, 3 layers)
    return np.stack(maps).astype("float32")
