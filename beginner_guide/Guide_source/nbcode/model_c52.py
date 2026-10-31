# Grad-CAM of the capsule network, on the 4 most flooded tiles of each never-seen flood
fig, axes = plt.subplots(4, 4, figsize=(18, 18))
row = 0

for pair in PAIRS:
    dataset = pair["test"]
    data = test_tiles[dataset]
    region = regions[dataset]

    # Flooded share of every test tile, and the 4 highest
    shares = []
    for number in range(len(data["targets"])):
        step, top, left = data["positions"][number]
        dry_land = region["dry_land"][top:top + TILE_SIZE, left:left + TILE_SIZE]
        shares.append(flooded_share(data["targets"][number][:, :, 0], dry_land))
    chosen = np.argsort(shares)[-4:]

    for column in range(4):
        tile = chosen[column]
        chosen_class = data["capsule_classes"][tile]
        heatmap = grad_cam(severity_grad_cam_models[pair["name"]], data["inputs"][tile:tile + 1], chosen_class)

        # Top: the water. Bottom: the heat map.
        show_water(axes[row, column], data["inputs"][tile, -1, :, :, 0], 2.0)
        axes[row, column].set_title(REGION_NAMES[dataset] + "\ntrue: " + SEVERITY_NAMES[data["labels"][tile]]
                                    + ", network: " + SEVERITY_NAMES[chosen_class])

        axes[row + 1, column].imshow(data["inputs"][tile, -1, :, :, 0], cmap="Greys", vmin=0, vmax=2)
        axes[row + 1, column].imshow(heatmap, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
        axes[row + 1, column].set_title("Grad-CAM for '" + SEVERITY_NAMES[chosen_class] + "'")

    row = row + 2

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
save_figure(fig, "06_grad_cam_capsule")
plt.show()
