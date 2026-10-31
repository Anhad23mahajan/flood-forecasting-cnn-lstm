"""Print-size versions (6.3 in wide, fonts 7 to 8 pt) of the laptop-study figures, drawn from the numbers of REPORT.md and the run files.
Run with the system python (numpy + matplotlib)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"report\diagrams"
RUNS = r"C:\Dev\flood-forecast-demo\completion\results\runs"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- Figure: ladder (numbers = Table 6, copied from REPORT.md, hybrid rain, mean of 3 folds) ----------------
rows = [
    ("Linear model, depth history", [0.818, 0.798, 0.790]),
    ("Two-feature lookup table", [0.786, 0.763, 0.737]),
    ("Per-pixel MLP",           [0.846, 0.854, 0.829]),
    ("Original CNN-LSTM",       [0.187, 0.188, 0.061]),
    ("Local CNN",               [0.841, 0.830, 0.807]),
    ("Small ConvLSTM",          [0.866, 0.840, 0.811]),
]
groups = ["E60: Australia to UK, 60 m", "E30: Australia to UK, 30 m", "E480: Pakistan to Mozambique, 480 m"]
shades = ["#d9d9d9", "#9e9e9e", "#6b6b6b"]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
x = np.arange(len(rows))
w = 0.26
for g in range(3):
    vals = [r[1][g] for r in rows]
    bars = ax.bar(x + (g - 1) * w, vals, width=w, color=shades[g], edgecolor="black", linewidth=0.5, label=groups[g])
    ax.bar_label(bars, labels=["%.2f" % (v + 1e-9) for v in vals], fontsize=5.5, padding=1.5, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right")
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("skill on the never-seen flood\n(0 = copying the last frame)")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper center", ncol=3, frameon=False, columnspacing=0.8, handlelength=1.2, fontsize=6.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "print_ladder.png"))
plt.close(fig)

# ---------------- Figure: skill by lead time (from the run files) ----------------
found = {}
for f in sorted(glob.glob(RUNS + r"\*.json")):
    d = json.load(open(f))
    batch_ok = d.get("batch") in (None, 1)
    lr_ok = d.get("lr") is None or abs(d["lr"] - 0.001) < 1e-12
    if not (d.get("regime") == "hybrid" and d.get("variant") == "full" and d.get("epochs") == 30 and batch_ok and lr_ok
            and not d.get("blocked") and not d.get("augment") and d.get("n_train") == 300 and d["seed"] in (1003, 1013, 1023)):
        continue
    if d["arch"] not in ("local", "clstm"):
        continue
    found.setdefault((d["exp"], d["arch"]), {})[d["fold"]] = d["test"]["lead_skill"]
table = {}
for key, folds in found.items():
    assert sorted(folds) == [0, 1, 2], (key, sorted(folds))
    table[key] = np.mean([folds[k] for k in sorted(folds)], axis=0)
json.dump({"%s_%s" % k: v.round(3).tolist() for k, v in table.items()}, open(os.path.join(OUT, "lead_skill_table.json"), "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
leads = [0.5, 1.0, 1.5, 2.0, 2.5]
for ax, exp, title in zip(axes, ["E60", "E30", "E480"], ["E60 (UK, 60 m)", "E30 (UK, 30 m)", "E480 (Mozambique)"]):
    ax.plot(leads, table[(exp, "local")], "o--", color="black", markersize=3, linewidth=1, label="Local CNN")
    ax.plot(leads, table[(exp, "clstm")], "s-", color="#555555", markersize=3, linewidth=1.2, label="Small ConvLSTM")
    ax.set_title(title)
    ax.set_xlabel("hours ahead")
    ax.grid(alpha=0.3)
    ax.set_xticks(leads)
    ax.set_xticklabels(["0.5", "1", "1.5", "2", "2.5"])
axes[0].set_ylabel("skill on the never-seen flood")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "print_lead.png"))
plt.close(fig)
for k in sorted(table):
    print(k, table[k].round(3).tolist())
print("done")
