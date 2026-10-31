# Name and colour of each land-cover code
LAND_USE = {
    1: ("Water", "#1f78b4"),
    2: ("Trees", "#33a02c"),
    4: ("Flooded vegetation", "#a6cee3"),
    5: ("Crops", "#fdbf6f"),
    7: ("Buildings", "#e31a1c"),
    8: ("Bare ground", "#b15928"),
    9: ("Snow / ice", "#ffffff"),
    10: ("Clouds", "#bdbdbd"),
    11: ("Rangeland", "#b2df8a"),
}

land_use_codes = list(LAND_USE.keys())
land_use_names = [LAND_USE[code][0] for code in land_use_codes]
land_use_colours = ListedColormap([LAND_USE[code][1] for code in land_use_codes])

for region in REGIONS:
    dem_path = os.path.join(DATA_DIR, "Relevant data", "DEM", region + "_DEM.tif")
    land_use_path = os.path.join(DATA_DIR, "Relevant data", "Land use and land cover", region + ".tif")
    start_water_path = os.path.join(DATA_DIR, START_WATER_FILES[region])

    ground_height = read_map(dem_path, max_size=800)
    land_use = read_map(land_use_path, max_size=800)
    start_water = read_map(start_water_path, max_size=800)

    # Turn each land-cover code into a position 0, 1, 2, ... so each gets its own colour
    land_use_position = np.full(land_use.shape, np.nan)
    for position, code in enumerate(land_use_codes):
        land_use_position[land_use == code] = position

    # Hide squares with less than 1 cm of water, so the water stands out
    start_water[start_water < WET_DEPTH_M] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(region, fontsize=16)

    # Left: how high the ground is
    picture = axes[0].imshow(ground_height, cmap="terrain")
    axes[0].set_title("Ground height")
    fig.colorbar(picture, ax=axes[0], label="metres above sea level")

    # Middle: what covers the ground
    picture = axes[1].imshow(land_use_position, cmap=land_use_colours, vmin=-0.5, vmax=len(land_use_codes) - 0.5)
    axes[1].set_title("What covers the ground")
    colour_bar = fig.colorbar(picture, ax=axes[1], ticks=range(len(land_use_codes)))
    colour_bar.ax.set_yticklabels(land_use_names)

    # Right: water already on the ground when the simulation starts
    axes[2].set_facecolor("lightgrey")
    picture = axes[2].imshow(start_water, cmap="Blues")
    axes[2].set_title("Water at the start (grey = dry)")
    fig.colorbar(picture, ax=axes[2], label="water depth (m)")

    for ax in axes:
        ax.axis("off")

    plt.show()
