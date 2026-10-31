"""Alignment probe using WATER BODIES, a much sharper signal than depth-vs-terrain correlation.

The 'Initial conditions' rasters hold the baseline water (rivers, lakes) on the SAME grid as the depth frames. The land-use
raster (30 m base grid) has a 'Water' class (code 1). If we resample the land-use water mask onto the depth grid with the right
geometry, the two masks must agree sharply; the wrong geometry gives a blur.

Hypotheses for how the coarse grid relates to the 30 m base grid:
  H1  full-extent bilinear resize to the target shape (the paper: "resampled ... using bilinear interpolation"), no crop
  H2  integer x16 block-average of the 30 m grid, cropped by some offset
Validated first on cases whose alignment is KNOWN: Australia (same grid) and UK (crop offset 5,5).
Usage: python probe_alignment_water.py Pakistan"""
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = "data/extracted/FloodCastBench"
region = sys.argv[1]
IC = {"Australia": "High-fidelity flood forecasting/Australia_30m.tif", "UK": "High-fidelity flood forecasting/UK_30m.tif",
      "Pakistan": "Low-fidelity flood forecasting/Pakistan_480m.tif", "Mozambique": "Low-fidelity flood forecasting/Mozambique_480m.tif"}[region]
with rasterio.open(f"{ROOT}/Relevant data/Initial conditions/{IC}") as s:
    ic = s.read(1).astype("float32")
H, W = ic.shape
M = ic > 0.05
print(f"{region}: initial-condition grid {H}x{W}, baseline-water pixels {M.sum()} ({M.mean():.2%}), max depth {ic.max():.2f} m")
with rasterio.open(f"{ROOT}/Relevant data/Land use and land cover/{region}.tif") as s:
    lu = s.read(1)
FH, FW = lu.shape
water = (lu == 1).astype("float32")
print(f"land-use grid {FH}x{FW}, water fraction {water.mean():.2%}; ratio to depth grid: rows {FH / H:.3f}, cols {FW / W:.3f}")
if M.sum() < 20:
    sys.exit("no baseline water on this grid - use another signal")


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def resample(sy, sx, oy, ox, area=True):
    """Value of the water fraction at each depth cell centre for a grid whose cells are sy x sx base pixels, origin (oy, ox)
    base pixels. Cell average via a small supersampling grid (bilinear read of a 2x-blurred map is enough here)."""
    n = 4
    yy = (oy + (np.arange(H * n) + 0.5) / n * sy)
    xx = (ox + (np.arange(W * n) + 0.5) / n * sx)
    Y, X = np.meshgrid(yy - 0.5, xx - 0.5, indexing="ij")
    v = map_coordinates(water, [Y, X], order=1, mode="constant", cval=0.0)
    return v.reshape(H, n, W, n).mean(axis=(1, 3))


rows = []
if region in ("Australia", "UK"):
    # known geometry: cells = base pixels, unknown origin offset
    for oy in range(0, 11):
        for ox in range(0, 11):
            rows.append((corr(M.astype("float32"), resample(1.0, 1.0, oy, ox)), f"offset ({oy},{ox})"))
else:
    sy0, sx0 = FH / H, FW / W
    # H1: full extent, then small shifts and scale changes around it
    for ds in (0.96, 0.98, 1.0, 1.02, 1.04):
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                sy, sx = sy0 * ds, sx0 * ds
                oy, ox = dy * sy, dx * sx
                if ds == 1.0 or (dy % 2 == 0 and dx % 2 == 0):
                    rows.append((corr(M.astype("float32"), resample(sy, sx, oy, ox)), f"H1 scale x{ds:.2f} shift ({dy},{dx}) cells"))
    # H2: exactly 16 base pixels per cell, every crop offset that fits
    for oy in range(0, int(FH - 16 * H) + 1, 4):
        for ox in range(0, int(FW - 16 * W) + 1, 4):
            rows.append((corr(M.astype("float32"), resample(16.0, 16.0, oy, ox)), f"H2 16x16 blocks, crop offset ({oy},{ox}) base px"))
rows.sort(reverse=True)
arr = np.array([r[0] for r in rows])
print(f"\n{len(rows)} geometries tried. Best five (Pearson r between baseline-water mask and land-use water):")
for r, name in rows[:5]:
    print(f"   r = {r:+.3f}   {name}")
print(f"   median r over all tried: {np.median(arr):+.3f};  best minus runner-up-of-a-different-geometry gap: {rows[0][0] - rows[5][0]:+.3f}")
