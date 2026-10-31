# Error at every step: copy, trend, Model A, Model B
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

for ax, dataset in zip(axes.flat, REGION_NAMES):
    model_a = evaluations[dataset]
    model_b = v2_evaluations[dataset]
    hours = model_a["hours"]
    cells = model_a["cells"]

    ax.plot(hours, 100 * np.sqrt(model_a["copy_squared_error"] / cells), color="grey", label="copy the last map")
    ax.plot(hours, 100 * np.sqrt(model_a["model_squared_error"] / cells), color="tab:blue", label="Model A")
    ax.plot(hours, 100 * np.sqrt(trend_squared_errors[dataset] / cells), color="tab:orange", linewidth=2.5, label="trend forecast")
    ax.plot(hours, 100 * np.sqrt(model_b["model_squared_error"] / cells), color="tab:green", linestyle="--", label="Model B")

    ax.set_title(REGION_NAMES[dataset])
    ax.set_xlabel("hours since the start of the flood")
    ax.set_ylabel("typical error (cm)")
    ax.legend()

fig.suptitle("Forecast error at every step (lower is better)")
plt.tight_layout()
save_figure(fig, "04_error_over_time_with_trend")
plt.show()
