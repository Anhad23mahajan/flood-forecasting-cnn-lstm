"""Architecture figure of the FINAL CNN-LSTM (small full-resolution ConvLSTM 'clstm') next to what it replaced (v1). Output: completion/results/architecture_v3.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAVY, TEAL, GREY, RED, LIGHT = "#1B4F72", "#5DA9C9", "#8A9BA8", "#B03A2E", "#E8EEF1"

fig, ax = plt.subplots(figsize=(16, 7.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 7.6); ax.axis("off")


def box(x, y, w, h, text, fc=NAVY, tc="white", fs=11, bold=False, ec=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1, color=GREY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2))


ax.text(0.2, 7.25, "Final model: a small full-resolution CNN-LSTM (4,741 parameters)", fontsize=16, fontweight="bold", color=NAVY)
box(0.2, 5.0, 2.7, 1.55, "Input: 5 observed frames\n(30-minute steps, 128x128)\ndepth | terrain | rain\n+ depth change since the\nprevious frame (in cm)", fc=LIGHT, tc=NAVY, fs=10.5)
box(3.4, 5.0, 3.0, 1.55, "ConvLSTM2D (8 channels, 3x3)\nruns over the 5 frames at\nFULL resolution: keeps the\nmap layout, so small\nper-pixel changes stay visible", fs=10.5, bold=False)
box(6.9, 5.0, 2.5, 1.55, "Head: 3x3 conv (16) -> 1x1 conv\n(zero-initialised) ->\n5 change maps", fs=10.5)
box(9.9, 5.0, 2.3, 1.55, "forecast = last observed\ndepth + change\n(persistence-anchored)", fc=TEAL, tc="white", fs=10.5, bold=True)
box(12.7, 5.0, 3.0, 1.55, "5 forecast depth maps\n(30 min ... 2.5 h)\nscored against 'copy the\nlast frame' (skill)", fc=LIGHT, tc=NAVY, fs=10.5)
for x0, x1 in ((2.9, 3.4), (6.4, 6.9), (9.4, 9.9), (12.2, 12.7)):
    arrow(x0, 5.78, x1, 5.78)
arrow(4.9, 4.95, 4.9, 4.3, TEAL)
box(3.4, 3.05, 6.2, 1.2, "Severity head (Australia 30 m only):\nthree stride-2 convs -> capsules (routing-by-agreement),\nor a plain dense layer as the control\n-> none / minor / moderate / severe", fc="white", tc=NAVY, fs=10, ec=TEAL, ls="--")
box(10.1, 3.05, 5.6, 1.2, "Post-hoc rule layer + Grad-CAM (explanations only;\nno gradients flow back from them)", fc="white", tc=NAVY, fs=10, ec=TEAL, ls="--")
arrow(9.6, 3.65, 10.1, 3.65, TEAL)

ax.text(0.2, 2.45, "What it replaced: the original CNN-LSTM (v1)", fontsize=13, fontweight="bold", color=RED)
box(0.2, 0.75, 3.0, 1.35, "Per-frame CNN to 16x16x64,\nthen FLATTENED", fc="white", tc=RED, ec=RED, fs=10)
box(3.7, 0.75, 3.0, 1.35, "ONE 128-number dense layer\n(bottleneck) -> LSTM vector", fc="white", tc=RED, ec=RED, fs=10)
box(7.2, 0.75, 3.6, 1.35, "Decoder: dense -> 16x16 map ->\nupsampling; only the LAST frame's\nfeatures as skip", fc="white", tc=RED, ec=RED, fs=10)
box(11.3, 0.75, 4.4, 1.35, "Could not see per-pixel changes in depth\n(scaled by 5 m: mm changes ~ 1e-4) -> a straight-line\nguess beat it; skill ~ +0.05 .. +0.2", fc=RED, tc="white", fs=10, bold=True)
for x0, x1 in ((3.2, 3.7), (6.7, 7.2), (10.8, 11.3)):
    arrow(x0, 1.42, x1, 1.42, RED)
plt.tight_layout()
plt.savefig("completion/results/architecture_v3.png", dpi=110)
print("saved completion/results/architecture_v3.png")
