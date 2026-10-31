"""Refine the crop offset of the 480 m regions (cells = exactly 16 base pixels), with three independent checks.

  1. terrain-structure score (high-passed max-depth vs high-passed DEM; most negative r = best)    [both regions]
  2. baseline-water score (initial-condition water mask vs land-use 'Water')                       [Pakistan only]
  3. NoData ring: DEM NoData pixels must lie OUTSIDE the true crop, as in the UK                    [both regions]
Usage: python refine_480m_alignment.py Pakistan 44 176   |   python refine_480m_alignment.py Mozambique 152 48"""
import glob
import os
import re
import sys

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, map_coordinates

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = "data/extracted/FloodCastBench"
region, cy, cx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sorted(glob.glob(f"{ROOT}/Low-fidelity flood forecasting/480m/{region}/*.tif"), key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
mx = None
for i in np.linspace(0, len(files) - 1, 24).astype(int):
    with rasterio.open(files[i]) as s:
        a = s.read(1).astype("float32")
    mx = a if mx is None else np.maximum(mx, a)
H, W = mx.shape
with rasterio.open(f"{ROOT}/Relevant data/DEM/{region}_DEM.tif") as s:
    dem_raw = s.read(1).astype("float32")
FH, FW = dem_raw.shape
nodata = dem_raw < -1e30
dem_raw[nodata] = np.nan
Q = 1 if FH * FW < 8e6 else 4
dem = dem_raw
if Q > 1:
    h4, w4 = FH // Q * Q, FW // Q * Q
    dem = np.nanmean(dem_raw[:h4, :w4].reshape(h4 // Q, Q, w4 // Q, Q), axis=(1, 3))
dem_f = np.where(np.isfinite(dem), dem, np.nanmean(dem))
hp = lambda a, s: a - gaussian_filter(a, s)
mxh = hp(np.log1p(mx * 20), 6.0); mxh = (mxh - mxh.mean()) / (mxh.std() + 1e-9)


def sample(field, oy, ox, q):
    n = 3
    yy = (oy + (np.arange(H * n) + 0.5) / n * 16.0) / q - 0.5
    xx = (ox + (np.arange(W * n) + 0.5) / n * 16.0) / q - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(field, [Y, X], order=1, mode="nearest").reshape(H, n, W, n).mean(axis=(1, 3))


def terrain_score(oy, ox):
    d = hp(sample(dem_f, oy, ox, Q), 6.0); d = (d - d.mean()) / (d.std() + 1e-9)
    return float((d * mxh).mean())


water_ok = False
if region == "Pakistan":
    with rasterio.open(f"{ROOT}/Relevant data/Initial conditions/Low-fidelity flood forecasting/Pakistan_480m.tif") as s:
        M = (s.read(1) > 0.05).astype("float32")
    with rasterio.open(f"{ROOT}/Relevant data/Land use and land cover/Pakistan.tif") as s:
        lu = s.read(1)
    wq = (lu[:FH // 4 * 4, :FW // 4 * 4] == 1).astype("float32").reshape(FH // 4, 4, FW // 4, 4).mean(axis=(1, 3))
    Mc = M - M.mean()
    water_ok = True

    def water_score(oy, ox):
        v = sample(wq, oy, ox, 4); v = v - v.mean()
        return float((v * Mc).sum() / np.sqrt((v * v).sum() * (Mc * Mc).sum() + 1e-12))

marg_y, marg_x = FH - 16 * H, FW - 16 * W
print(f"{region}: depth {H}x{W}; DEM {FH}x{FW}; margins (rows, cols) = ({marg_y}, {marg_x}) base px; coarse estimate ({cy},{cx})")
res = []
for oy in range(max(0, cy - 20), min(marg_y, cy + 20) + 1, 2):
    for ox in range(max(0, cx - 20), min(marg_x, cx + 20) + 1, 2):
        res.append((oy, ox, terrain_score(oy, ox), water_score(oy, ox) if water_ok else float("nan")))
res.sort(key=lambda r: r[2])
print("terrain-structure score, best five offsets (most negative r):")
for oy, ox, t, w in res[:5]:
    print(f"   ({oy:3d},{ox:3d})  terrain r={t:+.3f}" + (f"   water r={w:+.3f}" if water_ok else ""))
if water_ok:
    resw = sorted(res, key=lambda r: -r[3])
    print("baseline-water score, best five offsets (highest r):")
    for oy, ox, t, w in resw[:5]:
        print(f"   ({oy:3d},{ox:3d})  water r={w:+.3f}   terrain r={t:+.3f}")
    # joint: rank-sum of the two independent scores
    rt = {(r[0], r[1]): i for i, r in enumerate(res)}
    rw = {(r[0], r[1]): i for i, r in enumerate(resw)}
    joint = sorted(rt, key=lambda k: rt[k] + rw[k])[:5]
    print("joint (rank-sum of the two independent signals), best five:", joint)
    best = joint[0]
else:
    best = (res[0][0], res[0][1])
oy, ox = best
print(f"\nadopted crop offset for {region}: (row, col) = ({oy}, {ox}) base px  -> cell (i,j) covers base rows [{oy}+16i, {oy}+16i+16), cols [{ox}+16j, ...)")

# check 3: NoData pixels vs the crop
rr, cc = np.where(nodata)
inside = ((rr >= oy) & (rr < oy + 16 * H) & (cc >= ox) & (cc < ox + 16 * W)).sum() if len(rr) else 0
print(f"DEM NoData pixels: {len(rr)} ({len(rr) / nodata.size:.2%}); inside the adopted crop: {int(inside)}; outside: {len(rr) - int(inside)}")
if len(rr):
    print(f"   NoData bounding box rows {rr.min()}..{rr.max()}, cols {cc.min()}..{cc.max()};  crop rows {oy}..{oy + 16 * H}, cols {ox}..{ox + 16 * W}")
