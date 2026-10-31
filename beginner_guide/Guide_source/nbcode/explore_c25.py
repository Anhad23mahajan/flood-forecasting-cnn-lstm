OLD_TILE_SIZE = 128

label_rows = []

for dataset, m in measurements.items():
    # The moment when the most area is under water
    wettest_step = int(np.argmax(m["wet_area"]))
    depth = read_map(get_depth_files(dataset)[wettest_step * DEPTH_MAPS_PER_RAIN_MAP])

    rows, columns = depth.shape

    tiles_called_severe = 0
    total_tiles = 0
    flooded_shares = []

    # Cut the map into 128 x 128 tiles
    for top in range(0, rows - OLD_TILE_SIZE + 1, OLD_TILE_SIZE):
        for left in range(0, columns - OLD_TILE_SIZE + 1, OLD_TILE_SIZE):
            tile = depth[top:top + OLD_TILE_SIZE, left:left + OLD_TILE_SIZE]
            total_tiles += 1

            # The old label: "severe" if any single square is 2 m deep or more
            if np.nanmax(tile) >= 2.0:
                tiles_called_severe += 1

            # Share of the tile really under more than 30 cm of water
            flooded_shares.append(np.nanmean(tile > 0.3))

    if total_tiles == 0:
        label_rows.append({"Dataset": dataset, "Tiles": 0, "Called severe (%)": 0, "Actually flooded (%)": 0})
    else:
        label_rows.append({"Dataset": dataset, "Tiles": total_tiles,
                           "Called severe (%)": 100 * tiles_called_severe / total_tiles,
                           "Actually flooded (%)": 100 * np.mean(flooded_shares)})

label_table = pd.DataFrame(label_rows)

# Bars side by side: what the label says against how much of each tile is really flooded
positions = np.arange(len(label_table))
bar_width = 0.4

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(positions - bar_width / 2, label_table["Called severe (%)"], bar_width, color="tab:red",
       label="Tiles the old label calls 'severe'")
ax.bar(positions + bar_width / 2, label_table["Actually flooded (%)"], bar_width, color="tab:blue",
       label="Average share of a tile under more than 30 cm of water")

# Mark datasets whose map is smaller than one tile
for position, tiles in enumerate(label_table["Tiles"]):
    if tiles == 0:
        ax.text(position, 3, "map smaller\nthan one tile", ha="center")

ax.set_xticks(positions)
ax.set_xticklabels(label_table["Dataset"])
ax.set_ylabel("Percent")
ax.set_ylim(0, 105)
ax.set_title("Is the 'severe' label measuring the flood?\n(when the red bar is far taller than the blue bar, one deep spot decides the label)")
ax.legend()
plt.show()

display(label_table)
