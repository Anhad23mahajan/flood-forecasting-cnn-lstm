# Error at every step of the test period
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

for ax, dataset in zip(axes.flat, REGION_NAMES):
    evaluation = evaluations[dataset]
    hours = evaluation["hours"]
    cells = evaluation["cells"]

    ax.plot(hours, 100 * np.sqrt(evaluation["copy_squared_error"] / cells), color="grey", label="copy the last map")
    ax.plot(hours, 100 * np.sqrt(evaluation["model_squared_error"] / cells), color="tab:blue", label="Model A")
    ax.plot(hours, 100 * np.sqrt(evaluation["rule_squared_error"] / cells), color="tab:green", label="Model A + water rule")

    ax.set_title(REGION_NAMES[dataset])
    ax.set_xlabel("hours since the start of the flood")
    ax.set_ylabel("typical error (cm)")
    ax.legend()

fig.suptitle("Forecast error at every step (lower is better)")
plt.tight_layout()
save_figure(fig, "03_error_over_time")
plt.show()
