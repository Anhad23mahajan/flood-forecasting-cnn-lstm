# ===== Print-ready figures for the report: about 6.3 inches wide, text 7 to 8 pt, saved at 300 dpi =====
# HOW TO USE: paste as ONE new cell at the very bottom of your copy of Flood_model_updated.ipynb and run it LAST.
# Before it, these cells must have been run in the same session: the whole notebook, the baseline cell, the version-2 cells
# (cells 1 to 3) and the visuals cell B (it creates v2_evaluations, v2_examples and trend_squared_errors).
# If the session was closed: Runtime > Run all, then run this cell. It trains nothing (TRAIN_NEW stays False).
# It saves the PNG files into the figures folder on Drive and downloads one zip file.

import zipfile

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

PRINT_FILES = []


def save_print(fig, name):
    """Saves a figure at 300 dpi in the figures folder on Drive."""

    path = os.path.join(FIGURE_DIR, name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    PRINT_FILES.append(path)
    print("Saved:", name + ".png")


# ---------- 1. Skill: trend forecast, Model A, Model B ----------
fig, ax = plt.subplots(figsize=(6.3, 2.8))

columns = ["Skill: trend", "Skill: CNN-LSTM (first version)", "Skill: CNN-LSTM version 2"]
labels = ["Trend forecast (no network)", "Model A", "Model B"]
shades = ["#d9d9d9", "#5a5a5a", "#ffffff"]
hatches = ["", "", "////"]

positions = np.arange(len(v2_table))
bar_width = 0.26

for number, (column, label, shade, hatch) in enumerate(zip(columns, labels, shades, hatches)):
    values = v2_table[column].astype(float).to_numpy()
    bars = ax.bar(positions + (number - 1) * bar_width, values, width=bar_width, color=shade, edgecolor="black",
                  linewidth=0.6, hatch=hatch, label=label)
    ax.bar_label(bars, labels=["%.3f" % value for value in values], fontsize=5.5, padding=1.5, rotation=90)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylim(-2.2, 1.9)
ax.set_xticks(positions)
ax.set_xticklabels([region + "\n" + part for region, part in zip(v2_table["Region"], v2_table["Tested on"])], fontsize=6.5)
ax.set_ylabel("skill over copying the last map\n(0 = same as copying, 1 = perfect)")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper center", ncol=3, frameon=False, columnspacing=1.0, handlelength=1.2)
fig.tight_layout()
save_print(fig, "P1_skill_comparison")
plt.show()


# ---------- 2. Error at every step of the test period ----------
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4))

for ax, dataset in zip(axes.flat, REGION_NAMES):
    first_version = evaluations[dataset]
    version_2 = v2_evaluations[dataset]
    hours = first_version["hours"]
    cells = first_version["cells"]

    ax.plot(hours, 100 * np.sqrt(first_version["copy_squared_error"] / cells), color="#a0a0a0", linewidth=1.0, label="Copy the last map")
    ax.plot(hours, 100 * np.sqrt(first_version["model_squared_error"] / cells), color="black", linestyle="--", linewidth=0.9, label="Model A")
    ax.plot(hours, 100 * np.sqrt(trend_squared_errors[dataset] / cells), color="#6e6e6e", linewidth=2.2, label="Trend forecast")
    ax.plot(hours, 100 * np.sqrt(version_2["model_squared_error"] / cells), color="black", linestyle=":", linewidth=1.1, label="Model B")

    ax.set_title(REGION_NAMES[dataset])
    ax.set_xlabel("hours since the start of the flood")
    ax.set_ylabel("typical error (cm)")
    ax.grid(alpha=0.3)

handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_print(fig, "P2_error_over_time")
plt.show()


# ---------- 3. Error maps, one figure per map scale ----------
def error_map_figure(datasets, name, height):
    fig, axes = plt.subplots(len(datasets), 4, figsize=(6.3, height), layout="constrained")

    column_titles = ["Water now (m)", "True change (m)", "Error of Model A (m)", "Error of Model B (m)"]

    for row, dataset in enumerate(datasets):
        first_version = examples[dataset]
        version_2 = v2_examples[dataset]
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
            if row == 0:
                axes[row, column].set_title(column_titles[column], fontsize=7)

            colour_bar = fig.colorbar(picture, ax=axes[row, column], shrink=0.75, pad=0.02, aspect=18)
            colour_bar.ax.tick_params(labelsize=5.5)

        axes[row, 0].set_ylabel(REGION_NAMES[dataset] + "\n" + str(first_version["step"] * STEP_HOURS) + " h into the flood", fontsize=6.5)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True)

    save_print(fig, name)
    plt.show()


error_map_figure(["Australia_30m", "UK_30m"], "P3a_error_maps_30m", 3.2)
error_map_figure(["Pakistan_480m", "Mozambique_480m"], "P3b_error_maps_480m", 4.6)


# ---------- 4. Water balance check ----------
fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.9))

for ax, dataset in zip(axes, REGION_NAMES):
    region = regions[dataset]

    average_depth_mm = 1000 * region["depth"].mean(axis=(1, 2), dtype="float64")
    rise_mm = np.diff(average_depth_mm)
    rain_now_mm = 0.5 * region["rain"][:len(rise_mm)].mean(axis=(1, 2), dtype="float64")

    ax.scatter(rain_now_mm, rise_mm, s=4, color="black", alpha=0.6, linewidths=0)
    top = max(rain_now_mm.max(), rise_mm.max())
    ax.plot([0, top], [0, top], color="#777777", linestyle="--", linewidth=1.0)

    ax.set_title(REGION_NAMES[dataset], fontsize=7)
    ax.set_xlabel("rain x 0.5 h (mm)", fontsize=7)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("rise of the water (mm)", fontsize=7)
fig.tight_layout()
save_print(fig, "P4_water_balance")
plt.show()


# ---------- 5. Capsule network confusion matrices ----------
fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.2))
short_names = ["None", "Minor", "Mod.", "Severe"]

for ax, dataset in zip(axes, REGION_NAMES):
    data = test_tiles[dataset]

    matrix = np.zeros((4, 4), dtype=int)
    for true_class, answer in zip(data["labels"], data["capsule_classes"]):
        matrix[int(true_class), int(answer)] += 1

    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=matrix.max() * 1.15)

    for i in range(4):
        for j in range(4):
            colour = "white" if matrix[i, j] > 0.55 * matrix.max() else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6, color=colour)

    accuracy = np.mean(data["labels"] == data["capsule_classes"])
    ax.set_title(REGION_NAMES[dataset] + "\n" + str(round(100 * accuracy, 1)) + "% correct", fontsize=6.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(short_names, fontsize=5.5, rotation=45)
    ax.set_yticks(range(4))
    ax.set_yticklabels(short_names if ax is axes[0] else [], fontsize=5.5)
    ax.set_xlabel("network's answer", fontsize=6)

axes[0].set_ylabel("true severity", fontsize=6.5)
fig.tight_layout()
save_print(fig, "P5_capsule_confusion")
plt.show()


# ---------- 6. Grad-CAM of the forecaster (Model A) ----------
fig, axes = plt.subplots(4, 3, figsize=(6.3, 7.6), layout="constrained")

for row, dataset in enumerate(REGION_NAMES):
    region = regions[dataset]
    example = examples[dataset]
    top, left, height, width = grad_cam_window(example)

    window_forecaster = build_forecaster(height, width)
    window_forecaster.set_weights(forecasters[model_of_region[dataset]].get_weights())
    grad_cam_model = keras.Model(window_forecaster.input,
                                 [window_forecaster.get_layer("memory").output, window_forecaster.output])

    model_input = make_input(region, example["step"] - 1, top, left, height, width)[None]
    heatmap = grad_cam(grad_cam_model, model_input, lambda forecast: tf.reduce_sum(forecast))

    rows = slice(top, top + height)
    columns = slice(left, left + width)
    last = example["last"][rows, columns]
    predicted_change = example["predicted"][rows, columns] - last

    top_depth = max(np.percentile(example["true"][rows, columns], 99), 0.1)
    top_change = max(np.percentile(np.abs(predicted_change), 99.5), 0.01)

    picture = show_water(axes[row, 0], last, top_depth)
    colour_bar = fig.colorbar(picture, ax=axes[row, 0], shrink=0.75, pad=0.02, aspect=18)
    colour_bar.ax.tick_params(labelsize=5.5)

    picture = axes[row, 1].imshow(predicted_change, cmap="RdBu", vmin=-top_change, vmax=top_change)
    colour_bar = fig.colorbar(picture, ax=axes[row, 1], shrink=0.75, pad=0.02, aspect=18)
    colour_bar.ax.tick_params(labelsize=5.5)

    axes[row, 2].imshow(region["local_height"][rows, columns], cmap="Greys_r", vmin=-2, vmax=2)
    picture = axes[row, 2].imshow(heatmap, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    colour_bar = fig.colorbar(picture, ax=axes[row, 2], shrink=0.75, pad=0.02, aspect=18)
    colour_bar.ax.tick_params(labelsize=5.5)

    if row == 0:
        axes[row, 0].set_title("Water now (m)", fontsize=7)
        axes[row, 1].set_title("Predicted change (m)", fontsize=7)
        axes[row, 2].set_title("Grad-CAM (bright = important)", fontsize=7)

    axes[row, 0].set_ylabel(REGION_NAMES[dataset], fontsize=7)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

save_print(fig, "P6_grad_cam_forecaster")
plt.show()


# ---------- Download everything as one zip ----------
import tempfile

archive = os.path.join(tempfile.gettempdir(), "print_ready_figures.zip")

with zipfile.ZipFile(archive, "w") as zip_file:
    for path in PRINT_FILES:
        zip_file.write(path, os.path.basename(path))

try:
    from google.colab import files
    files.download(archive)
except Exception:
    print("The files are in this folder on Drive:", FIGURE_DIR)
