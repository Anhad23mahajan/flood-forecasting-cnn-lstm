column_titles = ["Water now", "True water in 30 min", "Predicted water in 30 min",
                 "True change", "Predicted change", "Model error (predicted - true)"]

fig, axes = plt.subplots(4, 6, figsize=(30, 21))

for row, dataset in enumerate(REGION_NAMES):
    example = examples[dataset]

    # Same colour range for the water maps, and for the change maps
    top_depth = max(np.percentile(example["true"], 99), 0.1)
    true_change = example["true"] - example["last"]
    top_change = max(np.percentile(np.abs(true_change), 99.5), 0.01)

    # The predicted columns show the forecast after the water rule
    pictures = [
        show_water(axes[row, 0], example["last"], top_depth),
        show_water(axes[row, 1], example["true"], top_depth),
        show_water(axes[row, 2], example["corrected"], top_depth),
        axes[row, 3].imshow(true_change, cmap="RdBu", vmin=-top_change, vmax=top_change),
        axes[row, 4].imshow(example["corrected"] - example["last"], cmap="RdBu", vmin=-top_change, vmax=top_change),
        axes[row, 5].imshow(example["corrected"] - example["true"], cmap="PuOr", vmin=-top_change, vmax=top_change),
    ]

    for column in range(6):
        axes[row, column].set_title(REGION_NAMES[dataset] + "\n" + column_titles[column])
        fig.colorbar(pictures[column], ax=axes[row, column], shrink=0.8)

    axes[row, 0].set_ylabel(str(example["step"] * STEP_HOURS) + " h into the flood")

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Model A + water rule: forecasts 30 minutes ahead", fontsize=18)
plt.tight_layout()
save_figure(fig, "03_forecast_maps")
plt.show()
