"""Alignment probe #2: high-pass flood footprint vs high-pass terrain.

The event's MAXIMUM DEPTH map follows the terrain (water sits in low ground). Depth-vs-DEM correlation alone is dominated by the
large-scale trend and barely changes with small shifts (why the first Mozambique search found no peak). Removing that trend from
both maps (subtract a wide Gaussian blur) leaves local structure - channels, terraces, embankments - which lines up sharply only
at the true geometry.

Validated on regions with KNOWN geometry first (Australia offset (0,0); UK offset (5,5)), then applied to the coarse regions.
Usage: python probe_alignment_terrain.py <Region> [n_frames]"""
import glob
import os
import re
import sys

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, map_coordinates

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = "data/extracted/FloodCastBench"
region = sys.argv[1]
NFR = int(sys.argv[2]) if len(sys.argv) > 2 else 24
RES = sys.argv[3] if len(sys.argv) > 3 else ("480m" if region in ("Pakistan", "Mozambique") else "30m")
SC = {"30m": 1, "60m": 2, "480m": 16}[RES]          # base (30 m) pixels per depth cell
fid = {"30m": "High-fidelity flood forecasting/30m", "60m": "High-fidelity flood forecasting/60m",
       "480m": "Low-fidelity flood forecasting/480m"}[RES]
files = sorted(glob.glob(f"{ROOT}/{fid}/{region}/*.tif"), key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
idx = np.linspace(0, len(files) - 1, NFR).astype(int)
mx = None
for i in idx:
    with rasterio.open(files[i]) as s:
        a = s.read(1).astype("float32")
    mx = a if mx is None else np.maximum(mx, a)
H, W = mx.shape
print(f"{region}: depth grid {H}x{W}; max-depth map from {NFR} frames; wet fraction {(mx > 0.05).mean():.1%}; max {mx.max():.2f} m")
with rasterio.open(f"{ROOT}/Relevant data/DEM/{region}_DEM.tif") as s:
    dem = s.read(1).astype("float32")
FH, FW = dem.shape
bad = dem < -1e30
dem[bad] = np.nan
print(f"DEM {FH}x{FW}; ratio rows {FH / H:.3f} cols {FW / W:.3f}; NoData {bad.mean():.1%}")

# work at 1/Q scale to keep it tractable (Q base pixels per working pixel)
Q = 1 if FH * FW < 8e6 else 4
if Q > 1:
    h4, w4 = FH // Q * Q, FW // Q * Q
    dem = np.nanmean(dem[:h4, :w4].reshape(h4 // Q, Q, w4 // Q, Q), axis=(1, 3))
fill = np.nanmean(dem)
dem_f = np.where(np.isfinite(dem), dem, fill)


def hp(a, sigma):
    return a - gaussian_filter(a, sigma)


SIG = 6.0                                # blur width in depth-grid cells
mxh = hp(np.log1p(mx * 20), SIG)
mxh = (mxh - mxh.mean()) / (mxh.std() + 1e-9)


def sample(sy, sx, oy, ox):
    """DEM at the depth-cell centres for cells of sy x sx base pixels with origin (oy, ox) base pixels, 3x3 supersampled."""
    n = 3
    yy = (oy + (np.arange(H * n) + 0.5) / n * sy) / Q - 0.5
    xx = (ox + (np.arange(W * n) + 0.5) / n * sx) / Q - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    v = map_coordinates(dem_f, [Y, X], order=1, mode="nearest")
    return v.reshape(H, n, W, n).mean(axis=(1, 3))


def score(sy, sx, oy, ox):
    d = hp(sample(sy, sx, oy, ox), SIG)
    d = (d - d.mean()) / (d.std() + 1e-9)
    return float((d * mxh).mean())          # Pearson r of the two high-passed maps


rows = []
if SC in (1, 2):   # 30 m (cells = base pixels) or 60 m (cells = 2x2 base pixels): scan every whole-pixel crop offset
    for oy in range(0, 13):
        for ox in range(0, 13):
            rows.append((score(float(SC), float(SC), oy, ox), f"{RES} cells of {SC}x{SC} base px, crop offset ({oy},{ox}) base px"))
else:
    sy0, sx0 = FH / H, FW / W
    for dy in range(-6, 7):                  # H1: full-extent resize, shifted by whole cells
        for dx in range(-6, 7):
            rows.append((score(sy0, sx0, dy * sy0, dx * sx0), f"H1 full-extent resize, shift ({dy:+d},{dx:+d}) cells"))
    for ds in (0.97, 0.985, 1.015, 1.03):    # H1 with a slightly different scale
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                rows.append((score(sy0 * ds, sx0 * ds, dy * sy0, dx * sx0), f"H1 scale x{ds} shift ({dy:+d},{dx:+d})"))
    marg_y, marg_x = FH - 16 * H, FW - 16 * W  # H2: exact 16-base-pixel cells, cropped
    if marg_y >= 0 and marg_x >= 0:
        for oy in range(0, int(marg_y) + 1, 8):
            for ox in range(0, int(marg_x) + 1, 8):
                rows.append((score(16.0, 16.0, oy, ox), f"H2 16-px blocks, crop offset ({oy},{ox}) base px"))
rows.sort()  # ascending: the physical relation is NEGATIVE (water sits in low ground), so the best geometry is the most negative r
arr = np.array([r[0] for r in rows])
print(f"\n{len(rows)} geometries tried. Best five (Pearson r, high-passed max-depth vs high-passed terrain):")
for r, name in rows[:5]:
    print(f"   r = {r:+.3f}   {name}")
mad = np.median(np.abs(arr - np.median(arr))) * 1.4826
print(f"   median {np.median(arr):+.3f}, robust sd {mad:.3f}; best is {(np.median(arr) - rows[0][0]) / (mad + 1e-9):.1f} robust-sd below the median; "
      f"gap to 5th best {rows[0][0] - rows[4][0]:+.3f}")
