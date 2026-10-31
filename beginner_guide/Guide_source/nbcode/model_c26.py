# Test Model A on whole maps in all four floods
evaluations = {}
examples = {}
whole_map_forecasters = {}

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]

    # Copy the trained weights into models the size of each whole map
    whole_map_forecasters[pair["train"]] = forecaster_for_whole_map(forecasters[pair["name"]], train_region)
    whole_map_forecasters[pair["test"]] = forecaster_for_whole_map(forecasters[pair["name"]], test_region)

    # Training flood: only the last 15% (never used in training)
    evaluations[pair["train"]], examples[pair["train"]] = evaluate_region(
        whole_map_forecasters[pair["train"]], train_region, train_region["validation_end"], train_region["number_of_maps"])

    # Test flood: the whole flood (never seen)
    evaluations[pair["test"]], examples[pair["test"]] = evaluate_region(
        whole_map_forecasters[pair["test"]], test_region, FIRST_TARGET, test_region["number_of_maps"])

    print(pair["name"], "done")
