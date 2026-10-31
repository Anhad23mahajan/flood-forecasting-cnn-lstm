"""Which rainfall file drives which depth change?  (physical timing test, independent of any model)

For every 30-minute interval j of a region's simulation (depth frames at t = 1800*j) we compute
    dV_j = mean depth at t=1800*(j+1)  -  mean depth at t=1800*j        (domain-mean water gain during interval j)
    R_i  = domain-mean rainfall of rainfall file i
and correlate dV_j with R_(j+k) for k = -6 .. +6.  The simulator's forcing was applied over some interval; water gained in interval j
should correlate best with the rainfall file that was applied during j.  k = 0 means "file j covers the same 30 minutes as depth
interval j" (the loader's assumption: rainfall file index = floor(t / 1800)).  k = -1 would mean the file covers the PREVIOUS
interval (rain leads), k = +1 the NEXT.  Dry-domain tails (no rain, no change) are trimmed by using only intervals where rain
or |dV| is non-negligible.

Run: python completion/a_rain_lag.py <Region> <res>       e.g.  Australia 30 | UK 30 | Pakistan 480 | Mozambique 480
Writes completion/results/rain_lag_<Region>_<res>.json
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = "data/extracted/FloodCastBench"
FID = {30: "High-fidelity flood forecasting/30m", 60: "High-fidelity flood forecasting/60m", 480: "Low-fidelity flood forecasting/480m"}
GEOM = {("Australia", 30): (1, (0, 0)), ("UK", 30): (1, (5, 5)), ("Australia", 60): (2, (0, 0)), ("UK", 60): (2, (5, 5)),
        ("Pakistan", 480): (16, (42, 178)), ("Mozambique", 480): (16, (150, 50))}
region, res = sys.argv[1], int(sys.argv[2])
scale, (oy, ox) = GEOM[(region, res)]

dfiles = glob.glob(f"{ROOT}/{FID[res]}/{region}/*.tif")
dsec = sorted((int(os.path.splitext(os.path.basename(p))[0]), p) for p in dfiles)
assert dsec[0][0] == 0 and all(b[0] - a[0] == 300 for a, b in zip(dsec, dsec[1:]))
rfiles = sorted(((datetime.strptime("".join(re.match(r"^(\d{8})-S(\d{6})$", os.path.splitext(os.path.basename(p))[0]).groups()), "%Y%m%d%H%M%S"), p)
                 for p in glob.glob(f"{ROOT}/Relevant data/Rainfall/{region} flood/*.tif")))
print(region, res, "depth frames", len(dsec), "rain files", len(rfiles), "first rain", rfiles[0][0], flush=True)

# domain-mean depth at t = 1800 * j
mean_depth, wet_area = [], []
for sec, p in dsec:
    if sec % 1800:
        continue
    with rasterio.open(p) as s:
        a = s.read(1)
    mean_depth.append(float(a.mean()))
    wet_area.append(float((a > 0.05).mean()))
mean_depth, wet_area = np.array(mean_depth), np.array(wet_area)
H, W = a.shape

# domain-mean rainfall of each file over the depth-covered area (block mean via decimated read of that window)
rain = []
for _, p in rfiles:
    with rasterio.open(p) as s:
        win = ((oy, oy + scale * H), (ox, ox + scale * W))
        b = s.read(1, window=win, out_shape=(max(1, H // 4), max(1, W // 4)), resampling=Resampling.average, boundless=True, fill_value=0)
    b = b.astype(np.float64)
    b[b < -1e30] = np.nan
    rain.append(float(np.nanmean(b)))
rain = np.array(rain)
print("rain mean per file: max %.3f mm, total %.1f mm" % (rain.max(), rain.sum()), flush=True)

n = len(mean_depth) - 1                       # intervals
dV = np.diff(mean_depth)                      # dV[j]: gain during interval j
dW = np.diff(wet_area)
act = (np.abs(dV) > 1e-6) | (rain[:n] > 1e-4)  # keep intervals with activity
lags = list(range(-6, 7))
out = {"region": region, "res": res, "n_intervals": int(n), "n_active": int(act.sum()), "lags": lags, "corr_dV": [], "corr_dWet": [],
       "rain_total_mm": float(rain.sum())}
for k in lags:
    js = [j for j in range(n) if act[j] and 0 <= j + k < len(rain)]
    r = np.array([rain[j + k] for j in js])
    out["corr_dV"].append(float(np.corrcoef(r, dV[js])[0, 1]) if r.std() > 0 else None)
    out["corr_dWet"].append(float(np.corrcoef(r, dW[js])[0, 1]) if r.std() > 0 else None)
# mass balance at k = 0: water gained (mm of domain-mean depth) vs rainfall of the same-index file (mm)
js0 = [j for j in range(n) if act[j]]
x0, y0 = rain[js0], dV[js0] * 1000.0
slope, icpt = np.polyfit(x0, y0, 1)
out["k0_slope_mmdepth_per_mmrain"] = float(slope)
out["k0_intercept_mm"] = float(icpt)
out["k0_total_gain_mm"] = float(dV.sum() * 1000.0)
out["k0_total_rain_mm"] = float(rain[:n].sum())
print("mass balance k=0: depth gain (mm) = %.3f * rain (mm) + %.3f ; total gain %.1f mm vs total rain %.1f mm" % (slope, icpt, out["k0_total_gain_mm"], out["k0_total_rain_mm"]))
best = int(np.nanargmax(np.array([c if c is not None else -9 for c in out["corr_dV"]])))
out["best_lag_dV"] = lags[best]
bw = int(np.nanargmax(np.array([c if c is not None else -9 for c in out["corr_dWet"]])))
out["best_lag_dWet"] = lags[bw]
print("lag k :", " ".join(f"{k:+d}" for k in lags))
print("corr dV  :", " ".join(f"{c:+.2f}" if c is not None else "  nan" for c in out["corr_dV"]), "  best k =", lags[best])
print("corr dWet:", " ".join(f"{c:+.2f}" if c is not None else "  nan" for c in out["corr_dWet"]), "  best k =", lags[bw])
os.makedirs("completion/results", exist_ok=True)
json.dump(out, open(f"completion/results/rain_lag_{region}_{res}.json", "w"), indent=1)
