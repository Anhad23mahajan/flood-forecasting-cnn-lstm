for dataset in DEPTH_FOLDERS:
    files = get_depth_files(dataset)

    # 6 moments spread evenly over the flood
    picked = np.linspace(0, len(files) - 1, 6).astype(int)

    maps = []
    for number in picked:
        maps.append(read_map(files[number], max_size=800))

    # Same colour scale for all 6 pictures, so they can be compared
    all_wet_values = np.concatenate([m[m > WET_DEPTH_M] for m in maps])
    deepest_shown = np.percentile(all_wet_values, 98)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(dataset + ": how the flood changes (grey = dry)", fontsize=15)

    for ax, depth, number in zip(axes, maps, picked):
        # Hide squares with less than 1 cm of water
        depth[depth < WET_DEPTH_M] = np.nan

        ax.set_facecolor("lightgrey")
        picture = ax.imshow(depth, cmap="Blues", vmin=0, vmax=deepest_shown)

        hours = get_seconds(files[number]) / 3600
        ax.set_title("after " + str(round(hours)) + " hours")
        ax.axis("off")

    fig.colorbar(picture, ax=axes, label="water depth (m)")
    plt.show()
