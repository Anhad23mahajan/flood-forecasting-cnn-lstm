# Bar chart of the skill of Model A, with and without the water rule
fig, ax = plt.subplots(figsize=(11, 5))

labels = []
for region, part in zip(result_table["Region"], result_table["Tested on"]):
    labels.append(region + "\n" + part)

positions = np.arange(len(labels))
ax.bar(positions - 0.2, result_table["CNN-LSTM: skill"], width=0.4, color="tab:blue", label="Model A")
ax.bar(positions + 0.2, result_table["CNN-LSTM + water rule: skill"], width=0.4, color="tab:green", label="Model A + water rule")
ax.axhline(0, color="black")

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_title("Skill over copying the last map (above 0 = better than copying)")
ax.set_ylabel("skill")
ax.legend()

save_figure(fig, "03_skill_per_region")
plt.show()
