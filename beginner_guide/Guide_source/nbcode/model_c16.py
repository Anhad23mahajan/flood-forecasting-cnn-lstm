# Make the training and checking tiles for both models
tiles = {}

for pair in PAIRS:
    region = regions[pair["train"]]

    train_inputs, train_targets, train_positions = make_training_tiles(
        region, FIRST_TARGET, region["train_end"], TRAIN_TILES)

    validation_inputs, validation_targets, validation_positions = make_training_tiles(
        region, region["train_end"], region["validation_end"], VALIDATION_TILES)

    tiles[pair["name"]] = {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "train_positions": train_positions,
        "validation_inputs": validation_inputs,
        "validation_targets": validation_targets,
        "validation_positions": validation_positions,
    }

    print(pair["name"], "training tiles:", train_inputs.shape, "checking tiles:", validation_inputs.shape)
