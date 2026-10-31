"""Why did the 60 m Australia model stall at train/persist = 0.9817? Hypothesis: the persistence MSE is dominated by a few huge-change windows,
so the pooled loss / skill is insensitive to what happens in the typical window.  Data only: per-window persistence MSE for every wet window,
its concentration (share of the total carried by the top 1 / 5 / 20 % of windows) and WHEN in the event the biggest changes happen.
Usage: python diagnose_persistence_tail.py"""
import os
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
from real_data_loader import RegionData
from windows import valid_positions, start_indices_for

ROOT = "data/extracted/FloodCastBench"
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}
PATCH, T_IN, H, STRIDE = 128, 5, 5, 6
for region, res in (("Australia", 30), ("Australia", 60), ("UK", 30), ("Pakistan", 480), ("Mozambique", 480)):
    rd = RegionData(root_dir=ROOT, region=region, fidelity_dir=FID[res])
    pos = valid_positions(rd, PATCH, 1.0)
    if not pos:
        print(f"{region} {res} m: no full 128-px tile"); continue
    starts = start_indices_for(rd, 80, T_IN, H, STRIDE)
    rows = []
    for st in starts:
        hist, fut = rd.window_frame_indices(int(st), T_IN, H, STRIDE)
        with rasterio.open(rd.depth_files[hist[-1]]) as s:
            last = s.read(1)
        futs = []
        for i in fut:
            with rasterio.open(rd.depth_files[i]) as s:
                futs.append(s.read(1))
        for (r, c) in pos:
            sl = (slice(r, r + PATCH), slice(c, c + PATCH))
            mse = float(np.mean([(f[sl] - last[sl]) ** 2 for f in futs]))
            wet = max(float(last[sl].max()), float(futs[-1][sl].max())) >= 0.05
            if wet:
                rows.append((mse, int(st), r, c))
    a = np.array([r[0] for r in rows])
    order = np.argsort(a)[::-1]
    tot = a.sum()
    share = lambda q: a[order[:max(1, int(len(a) * q))]].sum() / tot
    t_top = np.array([rows[i][1] for i in order[:max(1, int(len(a) * 0.05))]])
    print(f"{region:10s} {res:3d} m: {len(a):5d} wet windows | persistence RMSE mean {np.sqrt(a.mean()) * 1000:6.1f} mm, median window {np.sqrt(np.median(a)) * 1000:6.1f} mm | "
          f"share of total squared change carried by top 1% / 5% / 20% of windows: {share(0.01):.0%} / {share(0.05):.0%} / {share(0.20):.0%} | "
          f"top-5% windows sit at start indices {int(np.percentile(t_top, 10))}..{int(np.percentile(t_top, 90))} of {rd.max_start_index(T_IN, H, STRIDE)}")
