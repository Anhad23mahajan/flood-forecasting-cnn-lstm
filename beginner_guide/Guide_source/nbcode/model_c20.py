# Training curves (only shown when the models were trained in this run)
if len(histories) > 0:
    fig, axes = plt.subplots(1, len(histories), figsize=(14, 4.5), squeeze=False)

    for ax, name in zip(axes[0], histories):
        history = histories[name]
        epochs = np.arange(1, len(history["loss"]) + 1)

        # Square root of the error, in centimetres
        ax.plot(epochs, 100 * np.sqrt(history["loss"]), "o-", label="training tiles")
        ax.plot(epochs, 100 * np.sqrt(history["val_loss"]), "o-", label="checking tiles")

        ax.set_title(name + " model")
        ax.set_xlabel("epoch")
        ax.set_ylabel("typical error (cm)")
        ax.legend()

    save_figure(fig, "02_training_progress")
    plt.show()
