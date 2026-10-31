# Test tiles: last 15% of the training floods, and the whole of the never-seen floods
test_tiles = {}

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]

    parts = [
        (pair["train"], train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, first_target, last_target in parts:
        region = regions[dataset]
        inputs, targets, positions = make_test_tiles(region, first_target, last_target, TEST_TILES)

        # Capsule network's answer, and Model A's forecast (used by the rules)
        lengths = severity_models[pair["name"]].predict(inputs, batch_size=32, verbose=0)
        forecast = forecasters[pair["name"]].predict(inputs, batch_size=32, verbose=0)

        test_tiles[dataset] = {
            "inputs": inputs,
            "targets": targets,
            "positions": positions,
            "labels": tile_labels(region, targets, positions),
            "capsule_lengths": lengths,
            "capsule_classes": np.argmax(lengths, axis=1),
            "forecast": forecast,
        }

        print(REGION_NAMES[dataset], "test tiles:", len(inputs))
