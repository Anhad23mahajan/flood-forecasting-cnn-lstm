"""Water-surface-elevation (WSE) alignment probe: an independent, PHYSICAL test of the recovered depth<->terrain geometry.

Ponded water has a (nearly) level surface, so WSE = terrain + depth is much SMOOTHER than the terrain itself wherever water stands, and
only when terrain and depth are registered correctly: with a wrong offset the depth's structure no longer cancels the terrain's, and the
local variability of WSE inside the flood is larger.  For a candidate offset (oy, ox) the terrain is point-sampled at the depth cells'
centres (the coarse depth is a bilinear resampling of the 30 m result), WSE = terrain + depth of the frame with the largest wet area is
formed, and
      E(oy, ox) = mean over wet cells (depth > 0.3 m) of | WSE - gaussian_blur(WSE, sigma) |          (lower = better registered).
The probe is VALIDATED first on regions whose geometry is already known (Australia 30 m -> (0,0), Mozambique 480 m -> (150,50)); it is
only trusted for Pakistan if it recovers those.  Reports the arg-min offset and how far below the median of all candidates it lies.

Run: python completion/c_wse_probe.py <Region> <res>       Region in Australia|Mozambique|Pakistan|UK   res 30|60|480
"""
import glob
import json
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = "data/extracted/FloodCastBench"
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}
GEOM = {("Australia", 30): (1, (0, 0)), ("UK", 30): (1, (5, 5)), ("Australia", 60): (2, (0, 0)), ("UK", 60): (2, (5, 5)),
        ("Pakistan", 480): (16, (42, 178)), ("Mozambique", 480): (16, (150, 50))}
region, res = sys.argv[1], int(sys.argv[2])
scale, (ay, ax) = GEOM[(region, res)]

files = sorted(glob.glob(f"{ROOT}/{FID[res]}/{region}/*.tif"), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
best_frame, best_area = None, -1
for i in np.linspace(0, len(files) - 1, 40).astype(int):
    with rasterio.open(files[i]) as s:
        a = s.read(1)
    area = float((a > 0.3).mean())
    if area > best_area:
        best_area, best_frame, best_i = area, a, i
depth = best_frame
H, W = depth.shape
wet = depth > 0.3
print(f"{region} {res} m: depth grid {H}x{W}, frame {best_i} with wet fraction {best_area:.2f}", flush=True)

with rasterio.open(f"{ROOT}/Relevant data/DEM/{region}_DEM.tif") as s:
    dem = s.read(1)
FH, FW = dem.shape
dem = np.where(dem < -1e30, np.nan, dem)
half = scale // 2
SIG = 3.0 if scale > 1 else 8.0
R = 14 if scale > 1 else 12                                   # search radius (cells of the base grid) around the adopted offset
oys = range(max(0, ay - R), min(FH - scale * H, ay + R) + 1)
oxs = range(max(0, ax - R), min(FW - scale * W, ax + R) + 1)
if region == "Pakistan":                                      # wider search: the evidence for this region was the weakest
    oys = range(max(0, ay - 40), min(FH - scale * H, ay + 40) + 1)
    oxs = range(max(0, ax - 40), min(FW - scale * W, ax + 40) + 1)
E = np.full((len(oys), len(oxs)), np.nan)
E_dem = np.full_like(E, np.nan)
for a_, oy in enumerate(oys):
    for b_, ox in enumerate(oxs):
        d = dem[oy + half: oy + half + scale * H: scale, ox + half: ox + half + scale * W: scale][:H, :W]
        if d.shape != (H, W) or np.isnan(d[wet]).any():
            continue
        wse = np.where(wet, d + depth, d + 0.0)
        hp = np.abs(wse - gaussian_filter(np.nan_to_num(wse), SIG))
        E[a_, b_] = hp[wet].mean()
        hp2 = np.abs(d - gaussian_filter(np.nan_to_num(d), SIG))
        E_dem[a_, b_] = hp2[wet].mean()
    if a_ % 10 == 0:
        print(f"  row {a_}/{len(oys)}", flush=True)
ia, ib = np.unravel_index(np.nanargmin(E), E.shape)
best = (list(oys)[ia], list(oxs)[ib])
med, mad = np.nanmedian(E), np.nanmedian(np.abs(E - np.nanmedian(E))) * 1.4826
z = (med - np.nanmin(E)) / max(mad, 1e-12)
adopted = E[list(oys).index(ay), list(oxs).index(ax)] if ay in oys and ax in oxs else np.nan
# how sharp: the number of candidates within 1 robust-sd of the minimum
near = int(np.sum(E <= np.nanmin(E) + 0.25 * mad))
out = dict(region=region, res=res, frame=int(best_i), wet_fraction=best_area, adopted=[ay, ax], argmin=[int(best[0]), int(best[1])],
           E_min=float(np.nanmin(E)), E_adopted=float(adopted), E_median=float(med), robust_sd=float(mad), z_below_median=float(z),
           n_candidates=int(np.isfinite(E).sum()), n_near_min=near, offset_of_argmin_from_adopted=[int(best[0] - ay), int(best[1] - ax)])
print(json.dumps(out, indent=1))
os.makedirs("completion/results", exist_ok=True)
json.dump(out, open(f"completion/results/wse_probe_{region}_{res}.json", "w"), indent=1)
np.save(f"completion/results/wse_probe_{region}_{res}_map.npy", E)
