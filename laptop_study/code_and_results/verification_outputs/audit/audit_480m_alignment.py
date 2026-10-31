"""Can the 480 m regions (Pakistan, Mozambique) actually be aligned with their 30 m terrain? Our docs say they "cannot be".
That was only inferred from non-integer size ratios; the search we used for the UK was never tried here.

Hypothesis: the ML-ready depth raster is the 30 m grid block-averaged by 16 (30 m x 16 = 480 m) and then cropped, so
DEM_blockavg16[dy:dy+H, dx:dx+W] (for some sub-block phase and crop offset) should correlate with depth the way the aligned
UK DEM does. Search over every phase and offset; a real alignment shows as one offset far stronger than all the others.
Usage: python audit_480m_alignment.py Mozambique|Pakistan [phase_step]"""
import glob
import os
import re
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
region = sys.argv[1]
phase_step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROOT = "data/extracted/FloodCastBench"
F = 16

files = glob.glob(f"{ROOT}/Low-fidelity flood forecasting/480m/{region}/*.tif")
files.sort(key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
# the wettest of a few frames
best = None
for i in np.linspace(len(files) // 5, len(files) - 1, 8).astype(int):
    with rasterio.open(files[i]) as s:
        a = s.read(1).astype("float32")
    if best is None or a.sum() > best[0]:
        best = (a.sum(), a, i)
depth = best[1]
H, W = depth.shape
with rasterio.open(f"{ROOT}/Relevant data/DEM/{region}_DEM.tif") as s:
    dem = s.read(1).astype("float32")
valid = dem > -1e30
dem0 = np.where(valid, dem, 0.0).astype("float64")
print(f"{region}: depth {H}x{W} (wet fraction {(depth > 0.05).mean():.1%}, frame {best[2]}/{len(files)}), DEM {dem.shape}, "
      f"NoData pixels {(~valid).mean():.1%}")

results = []
n_dem_r, n_dem_c = dem.shape
dvalid = np.isfinite(depth)
for oy in range(0, F, phase_step):
    for ox in range(0, F, phase_step):
        hh, ww = (n_dem_r - oy) // F, (n_dem_c - ox) // F
        blk = dem0[oy:oy + hh * F, ox:ox + ww * F].reshape(hh, F, ww, F)
        cnt = valid[oy:oy + hh * F, ox:ox + ww * F].reshape(hh, F, ww, F).sum(axis=(1, 3))
        d = blk.sum(axis=(1, 3)) / np.maximum(cnt, 1)
        d[cnt < F * F // 2] = np.nan
        for dy in range(0, max(hh - H, -1) + 1):
            for dx in range(0, max(ww - W, -1) + 1):
                sub = d[dy:dy + H, dx:dx + W]
                if sub.shape != depth.shape:
                    continue
                m = np.isfinite(sub) & dvalid
                if m.sum() < 0.5 * m.size:
                    continue
                r = np.corrcoef(depth[m], sub[m])[0, 1]
                results.append((r, oy, ox, dy, dx))
results.sort()
arr = np.array([r[0] for r in results])
print(f"searched {len(results)} (phase, crop) combinations")
print("strongest (most negative) depth-terrain correlations:")
for r, oy, ox, dy, dx in results[:5]:
    print(f"   r={r:+.3f}  phase=({oy},{ox})  crop offset in 480 m cells=({dy},{dx})")
print(f"median r over all combinations: {np.median(arr):+.3f}; 5th-best vs median gap: {np.median(arr) - results[4][0]:+.3f}")
gap = results[0][0] - results[1][0]
print("VERDICT:", "one clear alignment peak (best is far stronger than the runner-up)" if gap < -0.05 and results[0][0] < -0.15
      else "no clear peak - alignment NOT recoverable by this method")
