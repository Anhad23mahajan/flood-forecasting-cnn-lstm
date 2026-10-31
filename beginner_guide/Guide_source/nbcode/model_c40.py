# Error maps of Model A and Model B (purple = too much water, orange = too little)
column_titles = ["Water now", "True change in 30 min", "Error: Model A", "Error: Model B"]

fig, axes = plt.subplots(4, 4, figsize=(24, 21))

for row, dataset in enumerate(REGION_NAMES):
    model_a = examples[dataset]
    model_b = v2_examples[dataset]

    top_depth = max(np.percentile(model_a["true"], 99), 0.1)
    true_change = model_a["true"] - model_a["last"]
    top_change = max(np.percentile(np.abs(true_change), 99.5), 0.01)

    pictures = [
        show_water(axes[row, 0], model_a["last"], top_depth),
        axes[row, 1].imshow(true_change, cmap="RdBu", vmin=-top_change, vmax=top_change),
        axes[row, 2].imshow(model_a["predicted"] - model_a["true"], cmap="PuOr", vmin=-top_change, vmax=top_change),
        axes[row, 3].imshow(model_b["predicted"] - model_b["true"], cmap="PuOr", vmin=-top_change, vmax=top_change),
    ]

    for column in range(4):
        axes[row, column].set_title(REGION_NAMES[dataset] + "\n" + column_titles[column])
        fig.colorbar(pictures[column], ax=axes[row, column], shrink=0.8)

    axes[row, 0].set_ylabel(str(model_a["step"] * STEP_HOURS) + " h into the flood")

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Where the forecasts go wrong", fontsize=18)
plt.tight_layout()
save_figure(fig, "04_error_maps")
plt.show()
