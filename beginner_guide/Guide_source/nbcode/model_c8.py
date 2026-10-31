def save_figure(fig, name):
    # Save a figure as a PNG on Drive
    fig.savefig(FIGURE_DIR + "/" + name + ".png", dpi=150, bbox_inches="tight")
    print("Saved figure:", name)


def save_table(table, name):
    # Save a table as a CSV on Drive
    table.to_csv(TABLE_DIR + "/" + name + ".csv", index=False)
    print("Saved table:", name)


def show_water(ax, depth_map, top_value):
    # Draw a depth map: dry squares grey, water blue
    water = np.array(depth_map, dtype="float32")
    water[water < WET_DEPTH_M] = np.nan
    return ax.imshow(water, cmap=water_colours, vmin=0, vmax=top_value)


def flooded_share(depth_map, dry_land):
    # Share of the normally dry land that is under more than 30 cm of water
    if dry_land.sum() == 0:
        return 0.0
    return float(np.mean(depth_map[dry_land] > FLOOD_DEPTH_M))


def severity_class(share):
    # Turn a flooded share into a class number: 0 None, 1 Minor, 2 Moderate, 3 Severe
    return int(np.digitize(share, SEVERITY_LIMITS))


def cut_into_tiles(rows, columns):
    # Top-left corner of every full 64 x 64 tile in a map
    corners = []
    for top in range(0, rows - TILE_SIZE + 1, TILE_SIZE):
        for left in range(0, columns - TILE_SIZE + 1, TILE_SIZE):
            corners.append((top, left))
    return corners
