"""Read EVERY file of the whole dataset (all 14,868) and check it. The earlier audits only touched what the pipeline used (~50%).

For every region and resolution:  frame count, numeric naming, even time spacing, identical shape, dtype, and per-file
NaN / NoData-sentinel / negative counts and min / max, aggregated. Also DEM, land use, initial conditions, rainfall, world files.
Writes a compact table; anomalies are listed by file.
Usage: python audit_full_dataset.py            (about 10-20 minutes)"""
import glob
import os
import re
import sys
import time

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = "data/extracted/FloodCastBench"
SENT = -1e30
problems = []


def stats_for(files, label, expect_step=None, kind="depth"):
    t0 = time.perf_counter()
    shapes, dtypes = set(), set()
    nan_files = sent_files = neg_files = 0
    lo, hi = np.inf, -np.inf
    for f in files:
        with rasterio.open(f) as s:
            a = s.read(1)
            shapes.add((s.height, s.width)); dtypes.add(str(a.dtype))
        a = a.astype("float64", copy=False) if a.dtype.kind != "f" else a
        nn = int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0
        ss = int((a < SENT).sum())
        valid = a[(a >= SENT) & np.isfinite(a)] if (nn or ss) else a
        if valid.size:
            lo, hi = min(lo, float(valid.min())), max(hi, float(valid.max()))
        nan_files += nn > 0; sent_files += ss > 0; neg_files += bool(valid.size and valid.min() < 0)
        if nn: problems.append(f"{label}: NaN in {os.path.basename(f)} ({nn} px)")
        if kind == "depth" and valid.size and valid.min() < 0: problems.append(f"{label}: negative depth in {os.path.basename(f)}")
    print(f"{label:52s} {len(files):5d} files  shapes {sorted(shapes)}  dtype {sorted(dtypes)}  range [{lo:.3g}, {hi:.3g}]  "
          f"files with NaN {nan_files}, NoData {sent_files}, negative {neg_files}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if len(shapes) != 1:
        problems.append(f"{label}: files have different shapes {shapes}")


def numeric_sorted(files):
    return sorted(files, key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))


print("=" * 120)
print("DEPTH FRAMES")
for fid, regs in (("High-fidelity flood forecasting/30m", ("Australia", "UK")), ("High-fidelity flood forecasting/60m", ("Australia", "UK")),
                  ("Low-fidelity flood forecasting/480m", ("Pakistan", "Mozambique"))):
    for r in regs:
        fs = glob.glob(f"{ROOT}/{fid}/{r}/*.tif")
        names = [os.path.splitext(os.path.basename(f))[0] for f in fs]
        if not all(n.isdigit() for n in names):
            problems.append(f"{fid}/{r}: non-numeric depth filenames")
        secs = sorted(int(n) for n in names)
        gaps = set(np.diff(secs).tolist())
        label = f"{fid.split('/')[-1]} / {r}"
        print(f"{label}: first {secs[0]} s, last {secs[-1]} s ({secs[-1] / 86400:.2f} days), step set {sorted(gaps)}")
        if len(gaps) != 1:
            problems.append(f"{label}: depth frames not evenly spaced: gaps {sorted(gaps)}")
        stats_for(numeric_sorted(fs), label)

print("\nRAINFALL")
for r in ("Australia", "UK", "Pakistan", "Mozambique"):
    fs = sorted(glob.glob(f"{ROOT}/Relevant data/Rainfall/{r} flood/*.tif"))
    first, last = os.path.basename(fs[0]), os.path.basename(fs[-1])
    print(f"{r}: first file {first}, last file {last}")
    stats_for(fs, f"rainfall / {r}", kind="rain")

print("\nSTATIC LAYERS")
for kind, pat in (("DEM", "Relevant data/DEM/*.tif"), ("land use", "Relevant data/Land use and land cover/*.tif"),
                  ("initial conditions", "Relevant data/Initial conditions/*/*.tif")):
    for f in sorted(glob.glob(f"{ROOT}/{pat}")):
        stats_for([f], f"{kind} / {os.path.basename(f)}", kind="static")

print("\nWORLD FILES")
for f in sorted(glob.glob(f"{ROOT}/Relevant data/*/*.tfw")):
    v = [float(x) for x in open(f).read().split()]
    print(f"{os.path.relpath(f, ROOT + '/Relevant data'):48s} pixel {v[0]:.3f} x {v[3]:.3f} m, origin ({v[4]:.2f}, {v[5]:.2f})")

print("\n" + "=" * 120)
print("ANOMALIES:" if problems else "NO ANOMALIES FOUND (beyond the documented NoData sentinel in DEM/rainfall padding).")
for p in problems:
    print("  -", p)
