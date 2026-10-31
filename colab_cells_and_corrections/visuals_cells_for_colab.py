# ===== CELL A of 2: bar chart, copy vs trend vs first version vs version 2 (instant, needs v2_table from cell 3) =====

fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")

columns = ["Skill: trend", "Skill: CNN-LSTM (first version)", "Skill: CNN-LSTM version 2"]
labels = ["trend (one number, no network)", "CNN-LSTM, first version", "CNN-LSTM, version 2 (starts from the trend)"]
colours = ["tab:orange", "tab:blue", "tab:green"]

positions = np.arange(len(v2_table))
bar_width = 0.26

for number, (column, label, colour) in enumerate(zip(columns, labels, colours)):
    bars = ax.bar(positions + (number - 1) * bar_width, v2_table[column].astype(float), width=bar_width,
                  color=colour, label=label)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

ax.axhline(0, color="black", linewidth=1)
ax.set_ylim(min(float(v2_table[columns].astype(float).to_numpy().min()) - 0.25, -0.25), 1.15)
ax.set_xticks(positions)
ax.set_xticklabels([region + "\n" + part for region, part in zip(v2_table["Region"], v2_table["Tested on"])])
ax.set_ylabel("skill (0 = same as copying the last map, 1 = perfect)")
ax.set_title("Skill over copy-the-last-map: the simple trend forecast against the two CNN-LSTM versions", fontsize=14)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right")

save_figure(fig, "08_skill_comparison")
plt.show()


# ===== CELL B of 2: error at every step, and error maps (takes a few minutes: it forecasts every whole map again) =====

# Per-step errors of version 2 and of the trend forecast, on the same test parts as before
v2_evaluations = {}
v2_examples = {}
trend_squared_errors = {}

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]
    trend_factor = trend_factors[pair["name"]]

    parts = [
        (pair["train"], train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, first_target, last_target in parts:
        region = regions[dataset]

        whole_map_model = v2_for_whole_map(v2_forecasters[pair["name"]], region, trend_factor)
        v2_evaluations[dataset], v2_examples[dataset] = evaluate_region(whole_map_model, region, first_target, last_target)
        del whole_map_model

        trend_errors = []

        for step in range(first_target, last_target):
            last = region["depth"][step - 1]
            trend = np.maximum(last + trend_factor * (last - region["depth"][step - 2]), 0)
            trend_errors.append(float(np.sum((trend - region["depth"][step]) ** 2, dtype="float64")))

        trend_squared_errors[dataset] = np.array(trend_errors)

    print(pair["name"], "model - done")


# Figure 1: forecast error at every step of the test period
fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
fig.suptitle("Forecast error at every step of the test period (lower is better)", fontsize=15)

for ax, dataset in zip(axes.flat, REGION_NAMES):
    first_version = evaluations[dataset]
    version_2 = v2_evaluations[dataset]
    hours = first_version["hours"]
    cells = first_version["cells"]

    ax.plot(hours, 100 * np.sqrt(first_version["copy_squared_error"] / cells), color="grey", label="copy the last map")
    ax.plot(hours, 100 * np.sqrt(first_version["model_squared_error"] / cells), color="tab:blue",
            label="CNN-LSTM, first version")
    ax.plot(hours, 100 * np.sqrt(trend_squared_errors[dataset] / cells), color="tab:orange", linewidth=2.5,
            label="trend (no network)")
    ax.plot(hours, 100 * np.sqrt(version_2["model_squared_error"] / cells), color="tab:green", linestyle="--",
            linewidth=1.5, label="CNN-LSTM, version 2")

    ax.set_title(REGION_NAMES[dataset])
    ax.set_xlabel("hours since the start of the flood")
    ax.set_ylabel("typical error over the map (cm)")
    ax.grid(alpha=0.3)
    ax.legend()

save_figure(fig, "08_error_over_time_with_trend")
plt.show()


# Figure 2: where the forecasts go wrong, at the moment of fastest change in each region
fig, axes = plt.subplots(4, 4, figsize=(24, 21), layout="constrained")
fig.suptitle("Where the forecasts go wrong, 30 minutes ahead, at the moment of fastest change in each region", fontsize=18)

column_titles = ["Water now", "True change in 30 min", "Error: first version", "Error: version 2"]
column_units = ["water depth (m)", "change in depth (m)", "error (m)", "error (m)"]

for row, dataset in enumerate(REGION_NAMES):
    first_version = examples[dataset]
    version_2 = v2_examples[dataset]

    # Both versions are shown at the same moment (it is chosen from the true maps only)
    assert first_version["step"] == version_2["step"]

    top_depth = max(np.percentile(first_version["true"], 99), 0.1)
    true_change = first_version["true"] - first_version["last"]
    top_change = max(np.percentile(np.abs(true_change), 99.5), 0.01)

    pictures = [
        show_water(axes[row, 0], first_version["last"], top_depth),
        axes[row, 1].imshow(true_change, cmap="RdBu", vmin=-top_change, vmax=top_change),
        axes[row, 2].imshow(first_version["predicted"] - first_version["true"], cmap="PuOr", vmin=-top_change, vmax=top_change),
        axes[row, 3].imshow(version_2["predicted"] - version_2["true"], cmap="PuOr", vmin=-top_change, vmax=top_change),
    ]

    for column, picture in enumerate(pictures):
        axes[row, column].set_title(REGION_NAMES[dataset] + "\n" + column_titles[column], fontsize=13)
        fig.colorbar(picture, ax=axes[row, column], label=column_units[column], shrink=0.8)

    axes[row, 0].set_ylabel(str(first_version["step"] * STEP_HOURS) + " h into the flood", fontsize=13)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.legend(handles=[Patch(color="lightgrey", label="Dry (less than 1 cm of water)")],
           loc="lower center", ncol=1, fontsize=13, bbox_to_anchor=(0.5, -0.02))

save_figure(fig, "08_error_maps")
plt.show()
