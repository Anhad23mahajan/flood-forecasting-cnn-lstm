for dataset, m in measurements.items():
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(dataset + ": rain and flood over time", fontsize=15)

    # Top: how hard it rained
    axes[0].bar(m["hours"], m["average_rain"], width=0.5, color="grey")
    axes[0].set_ylabel("Rain\n(mm per hour)")

    # Middle: average water depth over the whole map
    axes[1].plot(m["hours"], m["average_depth"] * 100, color="tab:blue")
    axes[1].set_ylabel("Average water\ndepth (cm)")

    # Bottom: how much of the map is under water
    axes[2].plot(m["hours"], m["wet_area"] * 100, color="tab:green")
    axes[2].set_ylabel("Area under\nwater (%)")
    axes[2].set_xlabel("Hours since the simulation started")

    plt.tight_layout()
    plt.show()
