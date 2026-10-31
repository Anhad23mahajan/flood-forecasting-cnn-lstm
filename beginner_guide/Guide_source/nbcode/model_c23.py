fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
balance_rows = []

for ax, dataset in zip(axes, REGION_NAMES):
    region = regions[dataset]

    # Average depth of the whole map at every step (mm), and how much it rises each 30 minutes
    average_depth_mm = 1000 * region["depth"].mean(axis=(1, 2), dtype="float64")
    rise_mm = np.diff(average_depth_mm)

    # Rain falling at the same time, times half an hour
    rain_now_mm = 0.5 * region["rain"][:len(rise_mm)].mean(axis=(1, 2), dtype="float64")

    # Rain of the 30 minutes before (the only rain a forecast is allowed to use)
    rain_before_mm = np.concatenate([[0.0], rain_now_mm[:-1]])

    # Each dot is one step; the red line is the law
    ax.scatter(rain_now_mm, rise_mm, s=10, alpha=0.6)
    top = max(rain_now_mm.max(), rise_mm.max())
    ax.plot([0, top], [0, top], "r--", label="rise = rain x 0.5 h")

    ax.set_title(REGION_NAMES[dataset])
    ax.set_xlabel("rain x 0.5 hours (mm)")
    ax.set_ylabel("average rise of the water (mm)")
    ax.legend()

    balance_rows.append({
        "Region": REGION_NAMES[dataset],
        "Total rise / total (rain x 0.5 h)": round(rise_mm.sum() / max(rain_now_mm.sum(), 1e-9), 3),
        "Correlation with rain at the same time": round(np.corrcoef(rain_now_mm, rise_mm)[0, 1], 3),
        "Correlation with rain of the 30 min before": round(np.corrcoef(rain_before_mm, rise_mm)[0, 1], 3),
    })

fig.suptitle("Water balance: average rise of the water vs the rain")
save_figure(fig, "03_water_balance_check")
plt.show()

balance_table = pd.DataFrame(balance_rows)
display(balance_table)
save_table(balance_table, "03_water_balance_check")
