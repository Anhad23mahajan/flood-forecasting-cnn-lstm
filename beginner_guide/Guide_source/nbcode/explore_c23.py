def match_strength(water_rise, rain, shift):
    # How closely the water rise follows the rain shifted by "shift" steps (1 = perfect match, 0 = no link)
    rise_values = []
    rain_values = []

    for step in range(len(water_rise)):
        rain_step = step + shift
        if 0 <= rain_step < len(rain) and not np.isnan(rain[rain_step]):
            rise_values.append(water_rise[step])
            rain_values.append(rain[rain_step])

    return np.corrcoef(rise_values, rain_values)[0, 1]


for dataset, m in measurements.items():
    # Water added to the map in each half hour, in millimetres
    water_rise = np.diff(m["average_depth"]) * 1000
    rain = m["average_rain"]

    # Match with the rain map stamped 30 minutes earlier, at the same time, and 30 minutes later
    strengths = [match_strength(water_rise, rain, -1),
                 match_strength(water_rise, rain, 0),
                 match_strength(water_rise, rain, 1)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(dataset, fontsize=15)

    # Left: which rain map explains the water rise best (the best one is green)
    labels = ["Rain map stamped\n30 min earlier", "Rain map stamped\nat the same time", "Rain map stamped\n30 min later"]
    colours = ["lightgrey", "lightgrey", "lightgrey"]
    colours[int(np.argmax(strengths))] = "tab:green"

    axes[0].bar(labels, strengths, color=colours)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("How closely they match (1 = perfectly)")
    axes[0].set_title("Which rain map explains the water that\nappears in the next 30 minutes?")
    for position, value in enumerate(strengths):
        axes[0].text(position, value + 0.02, str(round(value, 2)), ha="center")

    # Right: rain against the water that appeared
    rain_same_time = rain[:len(water_rise)]
    usable = ~np.isnan(rain_same_time)
    axes[1].scatter(rain_same_time[usable], water_rise[usable], s=10, color="tab:blue", label="Each dot = one half hour")

    # Line for "all the rain stays on the ground": half an hour of rain = 0.5 x the hourly rate
    rain_range = np.linspace(0, np.nanmax(rain_same_time), 50)
    axes[1].plot(rain_range, 0.5 * rain_range, "k--", label="If all the rain stayed on the ground")

    axes[1].set_xlabel("Rain (mm per hour)")
    axes[1].set_ylabel("Water added in the next 30 minutes (mm)")
    axes[1].set_title("Does the water added match the rain that fell?")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
