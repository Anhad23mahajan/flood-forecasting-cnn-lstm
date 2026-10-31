# Grad-CAM of Model A, one row per flood
fig, axes = plt.subplots(4, 3, figsize=(18, 22))

for row, dataset in enumerate(REGION_NAMES):
    region = regions[dataset]
    example = examples[dataset]
    top, left, height, width = grad_cam_window(example)

    # Model A at the window size, with the trained weights
    window_forecaster = build_forecaster(height, width)
    window_forecaster.set_weights(forecasters[model_of_region[dataset]].get_weights())
    grad_cam_model = keras.Model(window_forecaster.input,
                                 [window_forecaster.get_layer("memory").output, window_forecaster.output])

    model_input = make_input(region, example["step"] - 1, top, left, height, width)[None]
    heatmap = grad_cam(grad_cam_model, model_input)

    # The same window cut out of the maps
    last = example["last"][top:top + height, left:left + width]
    predicted_change = example["predicted"][top:top + height, left:left + width] - last

    top_depth = max(np.percentile(example["true"][top:top + height, left:left + width], 99), 0.1)
    top_change = max(np.percentile(np.abs(predicted_change), 99.5), 0.01)

    picture = show_water(axes[row, 0], last, top_depth)
    axes[row, 0].set_title(REGION_NAMES[dataset] + "\nwater now")
    fig.colorbar(picture, ax=axes[row, 0], shrink=0.8)

    picture = axes[row, 1].imshow(predicted_change, cmap="RdBu", vmin=-top_change, vmax=top_change)
    axes[row, 1].set_title(REGION_NAMES[dataset] + "\npredicted change in 30 min")
    fig.colorbar(picture, ax=axes[row, 1], shrink=0.8)

    # Heat map on top of the ground height
    axes[row, 2].imshow(region["local_height"][top:top + height, left:left + width], cmap="Greys_r", vmin=-2, vmax=2)
    picture = axes[row, 2].imshow(heatmap, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[row, 2].set_title(REGION_NAMES[dataset] + "\nGrad-CAM (bright = important)")
    fig.colorbar(picture, ax=axes[row, 2], shrink=0.8)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
save_figure(fig, "06_grad_cam_forecaster")
plt.show()
