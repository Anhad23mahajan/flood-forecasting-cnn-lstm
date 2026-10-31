# Bar chart: trend forecast vs Model A vs Model B
fig, ax = plt.subplots(figsize=(13, 5.5))

columns = ["Skill: trend", "Skill: Model A", "Skill: Model B"]
names = ["Trend forecast (no network)", "Model A", "Model B"]
colours = ["tab:orange", "tab:blue", "tab:green"]

positions = np.arange(len(v2_table))

for number in range(3):
    values = v2_table[columns[number]].astype(float)
    bars = ax.bar(positions + (number - 1) * 0.26, values, width=0.26, color=colours[number], label=names[number])
    ax.bar_label(bars, fmt="%.3f", fontsize=8)

labels = []
for region, part in zip(v2_table["Region"], v2_table["Tested on"]):
    labels.append(region + "\n" + part)

ax.axhline(0, color="black")
ax.set_ylim(-1.7, 1.15)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("skill (0 = same as copying, 1 = perfect)")
ax.set_title("Trend forecast vs Model A vs Model B")
ax.legend(loc="lower right")

save_figure(fig, "04_skill_comparison")
plt.show()
