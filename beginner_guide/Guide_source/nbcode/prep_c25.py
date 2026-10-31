for dataset in DATASETS:
    dataset_dir = os.path.join(SAVE_DIR, dataset)

    # mmap_mode="r" reads from the file only when needed, so big arrays don't fill the memory
    depth = np.load(os.path.join(dataset_dir, "depth.npy"), mmap_mode="r")
    rain = np.load(os.path.join(dataset_dir, "rain.npy"), mmap_mode="r")
    terrain = np.load(os.path.join(dataset_dir, "terrain.npy"))
    land_cover = np.load(os.path.join(dataset_dir, "land_cover.npy"))
    permanent_water = np.load(os.path.join(dataset_dir, "permanent_water.npy"))

    # Moment when the largest area is under more than 30 cm of water
    flooded_area = [np.mean(depth[step] > FLOOD_DEPTH_M) for step in range(depth.shape[0])]
    wettest = int(np.argmax(flooded_area))
    hours = wettest * STEP_SECONDS / 3600

    # Heaviest rain in the whole flood, so the rain picture uses a fixed scale
    heaviest_rain = max(float(np.max(rain)), 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout="constrained")
    fig.suptitle(dataset + ", " + str(round(hours)) + " hours after the start", fontsize=17)

    # Top left: water depth, with rivers and lakes from the land-cover map outlined in red
    water = np.array(depth[wettest])
    water[water < WET_DEPTH_M] = np.nan
    picture = axes[0, 0].imshow(water, cmap=water_colours, vmin=0, vmax=np.nanpercentile(water, 98))
    axes[0, 0].contour(permanent_water, levels=[0.5], colors="red", linewidths=0.7)
    axes[0, 0].set_title("Water depth", fontsize=13)
    fig.colorbar(picture, ax=axes[0, 0], label="metres")
    depth_legend = [Patch(color="lightgrey", label="Dry (less than 1 cm of water)"),
                    Line2D([], [], color="red", label="Rivers and lakes on the land-cover map")]
    axes[0, 0].legend(handles=depth_legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)

    # Top right: ground height
    picture = axes[0, 1].imshow(terrain, cmap="terrain")
    axes[0, 1].set_title("Ground height", fontsize=13)
    fig.colorbar(picture, ax=axes[0, 1], label="metres above sea level")

    # Bottom left: what covers the ground
    picture = axes[1, 0].imshow(land_cover_positions(land_cover), cmap=land_cover_colours,
                                vmin=-0.5, vmax=len(LAND_COVER) - 0.5, interpolation="nearest")
    axes[1, 0].set_title("What covers the ground", fontsize=13)
    colour_bar = fig.colorbar(picture, ax=axes[1, 0], ticks=range(len(LAND_COVER)))
    colour_bar.ax.set_yticklabels(land_cover_names)

    # Bottom right: rain falling in the next 30 minutes
    rain_now = np.array(rain[wettest], dtype="float32")
    picture = axes[1, 1].imshow(rain_now, cmap="Purples", vmin=0, vmax=heaviest_rain)
    axes[1, 1].set_title("Rain in the next 30 minutes", fontsize=13)
    fig.colorbar(picture, ax=axes[1, 1], label="mm per hour")
    if rain_now.max() == 0:
        axes[1, 1].text(0.5, 0.5, "No rain in this half hour", transform=axes[1, 1].transAxes,
                        ha="center", va="center", fontsize=14, backgroundcolor="white")

    for ax in axes.flat:
        ax.axis("off")

    plt.show()
