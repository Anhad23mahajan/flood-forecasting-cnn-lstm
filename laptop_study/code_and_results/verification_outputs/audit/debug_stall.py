"""Why does the 60 m Australia model not learn while Pakistan (480 m) and Australia 30 m do?  Inspect the actual training windows:
how often is there rain, how big is the rain, and does the window-level rain relate to the window-level change to be predicted?
Usage: python debug_stall.py <australia60|pakistan|australia30> [n_windows]"""
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
import depth_experiment as de
from real_data_loader import RegionData
from windows import valid_positions, start_indices_for

kind = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
region, res = {"australia60": ("Australia", 60), "pakistan": ("Pakistan", 480), "australia30": ("Australia", 30)}[kind]
rd = RegionData(root_dir=de.ROOT, region=region, fidelity_dir=de.FID[res])
rng = np.random.default_rng(42)
pos = valid_positions(rd, de.PATCH, 1.0)
perm = rng.permutation(len(pos))
tr_pos = [pos[i] for i in perm[max(1, int(round(len(pos) * 0.25))):]]
starts = start_indices_for(rd, 80, de.T_IN, de.HORIZON, de.STRIDE)
pool = de.scan_wet_windows(rd, tr_pos, starts)
sel = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
ex = de.load_examples(rd, sel)
rain_last = np.array([x[-1, ..., 2].mean() for x, _, _ in ex])
rain_max_hist = np.array([x[..., 2].mean(axis=(1, 2)).max() for x, _, _ in ex])
change = np.array([(y[-1] - x[-1, ..., 0]).mean() for x, y, _ in ex])          # mean depth change to be predicted (m)
abs_change = np.array([np.abs(y - x[-1, ..., 0][None]).mean() for x, y, _ in ex])
depth_last = np.array([x[-1, ..., 0].mean() for x, _, _ in ex])
print(f"{region} {res} m: {len(ex)} training windows (pool {len(pool)})")
print(f"  patch-mean rain in the last history frame: mean {rain_last.mean():.3f} mm, max {rain_last.max():.2f}; windows with rain > 0.05 mm: {(rain_last > 0.05).mean():.0%}; "
      f"> 1 mm: {(rain_last > 1).mean():.0%}")
print(f"  patch-mean depth (last frame): mean {depth_last.mean():.3f} m;  mean |change| per window {abs_change.mean() * 1000:.2f} mm, mean signed change {change.mean() * 1000:+.2f} mm")
if rain_last.std() > 0:
    print(f"  corr(rain, signed change) = {np.corrcoef(rain_last, change)[0, 1]:+.3f};  corr(rain, |change|) = {np.corrcoef(rain_last, abs_change)[0, 1]:+.3f};  "
          f"corr(depth, signed change) = {np.corrcoef(depth_last, change)[0, 1]:+.3f}")
    hi = rain_last > np.median(rain_last)
    print(f"  mean |change| when rain is above / below its median: {abs_change[hi].mean() * 1000:.2f} / {abs_change[~hi].mean() * 1000:.2f} mm")
else:
    print("  rain is constant in these windows: nothing to learn from it")
big = np.argsort(abs_change)[::-1][: max(1, len(ex) // 10)]
print(f"  the 10% biggest-change windows: rain mean {rain_last[big].mean():.3f} mm vs {rain_last.mean():.3f} overall; their depth mean {depth_last[big].mean():.3f} m")
