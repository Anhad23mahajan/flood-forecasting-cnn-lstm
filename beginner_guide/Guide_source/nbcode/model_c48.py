# Confusion matrix: true class (rows) vs the network's answer (columns)
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

for ax, dataset in zip(axes, REGION_NAMES):
    data = test_tiles[dataset]
    matrix = confusion_matrix(data["labels"], data["capsule_classes"], labels=range(4))
    ConfusionMatrixDisplay(matrix, display_labels=SEVERITY_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")

    accuracy = np.mean(data["labels"] == data["capsule_classes"])
    ax.set_title(REGION_NAMES[dataset] + "\naccuracy " + str(round(100 * accuracy, 1)) + "%")
    ax.set_xlabel("capsule network's answer")
    ax.set_ylabel("true severity")

save_figure(fig, "05_capsule_confusion")
plt.show()
