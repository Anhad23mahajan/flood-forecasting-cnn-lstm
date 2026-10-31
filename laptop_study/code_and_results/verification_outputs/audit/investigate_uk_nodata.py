"""Where are the NoData sentinel pixels in the UK DEM / rainfall, and does any window we actually use touch them?"""
import glob
import os
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
from real_data_loader import RegionData  # noqa: E402
from windows import all_positions, valid_positions  # noqa: E402

ROOT = "data/extracted/FloodCastBench"
FID = "High-fidelity flood forecasting/30m"
SENT = -1e30  # anything below this is the float32 NoData sentinel (-3.4028235e38)

for region in ("Australia", "UK"):
    rd = RegionData(root_dir=ROOT, region=region, fidelity_dir=FID)
    dy, dx = rd.layer_offset
    H, W = rd.grid_height, rd.grid_width
    print(f"\n=== {region}: depth grid {H}x{W}, layer offset {rd.layer_offset}")

    with rasterio.open(rd.dem_path) as s:
        dem = s.read(1)
    bad = dem < SENT
    print(f"DEM {dem.shape}: {int(bad.sum())} sentinel pixels ({bad.mean():.2%})")
    if bad.any():
        rr, cc = np.where(bad)
        print(f"   sentinel rows {rr.min()}..{rr.max()}, cols {cc.min()}..{cc.max()}")
        inside = bad[dy:dy + H, dx:dx + W]
        print(f"   inside the depth-aligned crop [{dy}:{dy + H}, {dx}:{dx + W}]: {int(inside.sum())} pixels")

    n_rain_bad_files, worst_frac, where = 0, 0.0, None
    for f in rd.rain_files:
        with rasterio.open(f) as s:
            a = s.read(1)
        b = a < SENT
        if b.any():
            n_rain_bad_files += 1
            frac = b.mean()
            if frac > worst_frac:
                worst_frac = frac
                rr, cc = np.where(b)
                where = (rr.min(), rr.max(), cc.min(), cc.max())
    print(f"rainfall: {n_rain_bad_files} of {len(rd.rain_files)} files contain sentinel pixels; "
          f"worst file {worst_frac:.2%} of pixels; sentinel bbox rows/cols {where}")

    # which patches would the pipeline use, and do their DEM / rainfall windows touch a sentinel?
    positions = valid_positions(rd, 128, 1.0)
    print(f"valid patches used by the pipeline: {len(positions)} of {len(all_positions(rd, 128))} tiles")
    touched_dem = 0
    for (r, c) in positions:
        w = dem[r + dy:r + dy + 128, c + dx:c + dx + 128]
        touched_dem += int((w < SENT).any())
    print(f"   patches whose DEM window contains a sentinel: {touched_dem}")
    # a patch dropped by the flat-DEM filter, and why
    dropped = [p for p in all_positions(rd, 128) if p not in set(positions)]
    print(f"   tiles dropped by the flat-DEM filter: {len(dropped)}")

    # rainfall windows for the patches used, over ALL files
    if region == "UK":
        worst = 0
        for f in rd.rain_files:
            with rasterio.open(f) as s:
                a = s.read(1)
            for (r, c) in positions:
                w = a[r + dy:r + dy + 128, c + dx:c + dx + 128]
                worst = max(worst, int((w < SENT).sum()))
        print(f"   max sentinel pixels inside any used UK rainfall window (all {len(rd.rain_files)} files): {worst}")
