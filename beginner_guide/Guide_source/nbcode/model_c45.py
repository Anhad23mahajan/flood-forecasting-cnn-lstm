def tile_labels(region, targets, positions):
    # Severity class of each tile, from the depth map 30 minutes later
    labels = []
    for number in range(len(targets)):
        target_step, top, left = positions[number]
        dry_land = region["dry_land"][top:top + TILE_SIZE, left:left + TILE_SIZE]
        labels.append(severity_class(flooded_share(targets[number][:, :, 0], dry_land)))
    return np.array(labels)


# Labels for the training and checking tiles
for pair in PAIRS:
    data = tiles[pair["name"]]
    region = regions[pair["train"]]
    data["train_labels"] = tile_labels(region, data["train_targets"], data["train_positions"])
    data["validation_labels"] = tile_labels(region, data["validation_targets"], data["validation_positions"])

# How many training tiles are in each class
label_counts = pd.DataFrame(index=SEVERITY_NAMES)
for pair in PAIRS:
    label_counts[pair["name"] + " model"] = np.bincount(tiles[pair["name"]]["train_labels"], minlength=4)
display(label_counts)
