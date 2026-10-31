"""Data-level audit. Independent of the model. Each check re-derives something the pipeline claims, straight from the
raw files with plain rasterio/numpy (not through RegionData / windows.py), and compares.

Run from the project root:   python verification_outputs/audit/audit_data_checks.py
Needs the local dataset at data/extracted/FloodCastBench (Australia + UK).
"""
import os
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
from real_data_loader import RegionData, depth_to_severity  # noqa: E402
from windows import valid_positions, scan_windows, start_indices_for  # noqa: E402

ROOT = "data/extracted/FloodCastBench"
FID = "High-fidelity flood forecasting/30m"
PATCH, T_IN, HORIZON, STRIDE = 128, 5, 5, 6
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


rd = {r: RegionData(root_dir=ROOT, region=r, fidelity_dir=FID) for r in ("Australia", "UK")}

# ------------------------------------------------------------------ A1 raster sanity (NoData / NaN / negatives / units)
print("\nA1. raw raster sanity (a NoData or NaN value would silently corrupt normalization and losses)")
for r, d in rd.items():
    n = len(d.depth_files)
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    worst_min, worst_max, nan_total, nodata_tags = np.inf, -np.inf, 0, set()
    for i in idxs:
        with rasterio.open(d.depth_files[i]) as s:
            a = s.read(1)
            nodata_tags.add(s.nodata)
            dt = a.dtype
        nan_total += int(np.isnan(a).sum())
        worst_min, worst_max = min(worst_min, float(np.nanmin(a))), max(worst_max, float(np.nanmax(a)))
    check(f"{r} depth: no NaN, non-negative, plausible metres", nan_total == 0 and worst_min >= 0 and worst_max < 50,
          f"(dtype {dt}, min {worst_min:.3f}, max {worst_max:.2f} m, nodata tag {nodata_tags}, {len(idxs)} frames)")
    # The UK DEM / rainfall carry a NoData padding ring OUTSIDE the depth-aligned crop (found by this audit), so the
    # test is: clean inside the crop the pipeline reads, and count what lies outside it.
    dy, dx = d.layer_offset
    crop = (slice(dy, dy + d.grid_height), slice(dx, dx + d.grid_width))
    with rasterio.open(d.dem_path) as s:
        dem_all = s.read(1).astype("float64"); dem_nodata = s.nodata
    ring = int((dem_all < -1e30).sum() - (dem_all[crop] < -1e30).sum())
    dem = dem_all[crop]
    check(f"{r} DEM inside the aligned crop: no NaN, no NoData sentinel", not np.isnan(dem).any() and dem.min() > -1000 and dem.max() < 9000,
          f"(nodata tag {dem_nodata}; min {dem.min():.1f}, max {dem.max():.1f} m; sentinel pixels outside the crop: {ring})")
    rmin, rmax, rnan, rring = np.inf, -np.inf, 0, 0
    for i in sorted({0, len(d.rain_files) // 3, len(d.rain_files) // 2, len(d.rain_files) - 1}):
        with rasterio.open(d.rain_files[i]) as s:
            a_all = s.read(1)
        a = a_all[crop]
        rring += int((a_all < -1e30).sum() - (a < -1e30).sum())
        rnan += int(np.isnan(a).sum()); rmin = min(rmin, float(np.nanmin(a))); rmax = max(rmax, float(np.nanmax(a)))
    check(f"{r} rainfall inside the aligned crop: no NaN, non-negative", rnan == 0 and rmin >= 0,
          f"(min {rmin:.3f}, max {rmax:.2f} mm per file; sentinel pixels outside the crop over 4 files: {rring})")
    with rasterio.open(d.landuse_path) as s:
        lu = np.unique(s.read(1))
    print(f"        {r} land-use codes present: {lu.tolist()}")

# ------------------------------------------------------------------ A2 train/validation disjointness (notebook's own procedure)
print("\nA2. spatial split: are training and validation patches disjoint? (leakage check, replicating the notebook's procedure)")
au = rd["Australia"]
pos = valid_positions(au, PATCH, 1.0)
for seed in (42, 7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pos))
    nval = max(1, int(round(len(pos) * 0.25)))
    val_p = {pos[i] for i in perm[:nval]}
    tr_p = {pos[i] for i in perm[nval:]}
    # patches are non-overlapping tiles on a 128-px grid, so disjoint origins == disjoint pixels
    origins_ok = all(r % PATCH == 0 and c % PATCH == 0 for r, c in pos)
    check(f"seed {seed}: {len(tr_p)} train / {len(val_p)} val patches share no tile", not (tr_p & val_p) and origins_ok)

# ------------------------------------------------------------------ A3 window arithmetic: no future frame may leak into the input
print("\nA3. window indexing: history strictly before forecast, correct spacing, correct rainfall file")
ok_all = True
for start in (0, 1, 500, 1234, au.max_start_index(T_IN, HORIZON, STRIDE)):
    hist, fut = au.window_frame_indices(start, T_IN, HORIZON, STRIDE)
    gaps = {b - a for a, b in zip(hist + fut, (hist + fut)[1:])}
    rain_idx = [au.rainfall_index_for_depth_second(au.depth_seconds[i]) for i in hist]
    ok = (max(hist) < min(fut) and gaps == {STRIDE} and len(hist) == T_IN and len(fut) == HORIZON
          and all(b - a == 1 for a, b in zip(rain_idx, rain_idx[1:])))
    ok_all &= ok
check("history < future, uniform 30-min spacing, one new rainfall file per step (5 windows incl. first/last)", ok_all)
try:
    au.window_frame_indices(au.max_start_index(T_IN, HORIZON, STRIDE) + 1, T_IN, HORIZON, STRIDE)
    check("a window that runs past the end is rejected", False)
except ValueError:
    check("a window that runs past the end is rejected", True)

# ------------------------------------------------------------------ A4 loader output == direct raw reads; labels recomputed independently
print("\nA4. loader vs direct reads (inputs, targets, rainfall, DEM, severity label, persistence baseline)")
rng = np.random.default_rng(123)
THR = (0.3, 1.0, 2.0)


def sev(v):
    return sum(v >= t for t in THR)


def raw(path, r0, c0, h=PATCH, w=PATCH):
    with rasterio.open(path) as s:
        return s.read(1, window=((r0, r0 + h), (c0, c0 + w))).astype("float32")


bad = 0
for name, d in rd.items():
    positions = valid_positions(d, PATCH, 1.0)
    for _ in range(12):
        r0, c0 = positions[rng.integers(len(positions))]
        start = int(rng.integers(0, d.max_start_index(T_IN, HORIZON, STRIDE) + 1))
        x, y, label, mx = d.load_patch_sequence(r0, c0, PATCH, start, T_IN, HORIZON, STRIDE)
        hist, fut = d.window_frame_indices(start, T_IN, HORIZON, STRIDE)
        dy, dx = d.layer_offset
        ok = True
        for k, i in enumerate(hist):
            ok &= np.array_equal(x[k, ..., 0], raw(d.depth_files[i], r0, c0))
            ri = int(d.depth_seconds[i] // d.rain_step_seconds)
            ok &= np.array_equal(x[k, ..., 2], raw(d.rain_files[ri], r0 + dy, c0 + dx))
        ok &= np.array_equal(x[0, ..., 1], raw(d.dem_path, r0 + dy, c0 + dx))
        for k, i in enumerate(fut):
            ok &= np.array_equal(y[k], raw(d.depth_files[i], r0, c0))
        fut_max = max(float(raw(d.depth_files[i], r0, c0).max()) for i in fut)
        ok &= (label == sev(fut_max)) and np.isclose(mx, fut_max)
        # persistence-baseline severity = thresholds on the last observed frame
        base_direct = sev(float(raw(d.depth_files[hist[-1]], r0, c0).max()))
        base_scan = scan_windows(d, [(r0, c0)], [start], PATCH, T_IN, HORIZON, STRIDE)[0]
        ok &= (base_scan[3] == label) and (base_scan[4] == base_direct)
        bad += (not ok)
check("24 random windows (12 Australia + 12 UK): every array and both labels match an independent read", bad == 0,
      f"({bad} mismatches)")

# ------------------------------------------------------------------ A5 UK offset: aligned vs unaligned depth-terrain correlation
print("\nA5. UK offset (5,5): is the depth raster really a crop of the terrain grid?")
uk = rd["UK"]
n = len(uk.depth_files)
best = None
for i in range(n // 4, n, max(1, n // 12)):
    a = raw(uk.depth_files[i], 0, 0, uk.grid_height, uk.grid_width)
    if best is None or a.sum() > best[0]:
        best = (a.sum(), a)
depth = best[1]
with rasterio.open(uk.dem_path) as s:
    dem_full = s.read(1).astype("float64")
H, W = depth.shape
r_al = np.corrcoef(depth.ravel(), dem_full[5:5 + H, 5:5 + W].ravel())[0, 1]
r_un = np.corrcoef(depth.ravel(), dem_full[0:H, 0:W].ravel())[0, 1]
scan = {(dy, dx): np.corrcoef(depth.ravel(), dem_full[dy:dy + H, dx:dx + W].ravel())[0, 1]
        for dy in range(11) for dx in range(11)}
best_off = min(scan, key=scan.get)
check("aligned (5,5) is the strongest depth-terrain relationship among all 121 candidate offsets",
      best_off == (5, 5), f"(r aligned {r_al:+.3f} | r unaligned (0,0) {r_un:+.3f} | best offset {best_off})")

# ------------------------------------------------------------------ A6 scan_windows vs an independent census
print("\nA6. window scan vs an independent census on 3 time points x every valid Australian patch")
starts = start_indices_for(au, 3, T_IN, HORIZON, STRIDE)
mine = scan_windows(au, pos, starts, PATCH, T_IN, HORIZON, STRIDE)
mism = 0
for (r0, c0, st, lab, blab) in mine:
    hist, fut = au.window_frame_indices(int(st), T_IN, HORIZON, STRIDE)
    lab2 = sev(max(float(raw(au.depth_files[i], r0, c0).max()) for i in fut))
    blab2 = sev(float(raw(au.depth_files[hist[-1]], r0, c0).max()))
    mism += (lab != lab2) or (blab != blab2)
frac = [sum(1 for w in mine if w[3] == k) / len(mine) for k in range(4)]
check(f"{len(mine)} scanned windows: every severity and baseline label matches", mism == 0,
      f"(class mix here: none {frac[0]:.0%} minor {frac[1]:.0%} moderate {frac[2]:.0%} severe {frac[3]:.0%})")

# ------------------------------------------------------------------ A8 the 60 m and 480 m regions: loader == independent block-averaged reads
print("\nA8. coarse regions (60 m, 480 m): does the loader's DEM / rainfall equal an independent block average of the raw 30 m files?")
COARSE = [("Pakistan", "Low-fidelity flood forecasting/480m", 16, (42, 178)), ("Mozambique", "Low-fidelity flood forecasting/480m", 16, (150, 50)),
          ("Australia", "High-fidelity flood forecasting/60m", 2, (0, 0)), ("UK", "High-fidelity flood forecasting/60m", 2, (5, 5))]
rng8 = np.random.default_rng(8)
for name, fid8, sc, (oy, ox) in COARSE:
    d8 = RegionData(root_dir=ROOT, region=name, fidelity_dir=fid8)
    size = PATCH if min(d8.grid_height, d8.grid_width) >= PATCH else min(d8.grid_height, d8.grid_width)
    ok8, n8 = True, 0
    for _ in range(6):
        r0 = int(rng8.integers(0, d8.grid_height - size + 1)); c0 = int(rng8.integers(0, d8.grid_width - size + 1))
        st = int(rng8.integers(0, d8.max_start_index(T_IN, HORIZON, STRIDE) + 1))
        x, y, _, _ = d8.load_patch_sequence(r0, c0, size, st, T_IN, HORIZON, STRIDE)
        hist, fut = d8.window_frame_indices(st, T_IN, HORIZON, STRIDE)

        def blockmean(path):
            with rasterio.open(path) as s:
                a = s.read(1, window=((oy + sc * r0, oy + sc * (r0 + size)), (ox + sc * c0, ox + sc * (c0 + size)))).astype("float64")
            return a.reshape(size, sc, size, sc).mean(axis=(1, 3))
        ok8 &= np.allclose(x[0, ..., 1], blockmean(d8.dem_path), rtol=1e-5, atol=1e-3)
        for k, i in enumerate(hist):
            ri = int(d8.depth_seconds[i] // d8.rain_step_seconds)
            ok8 &= np.allclose(x[k, ..., 2], blockmean(d8.rain_files[ri]), rtol=1e-5, atol=1e-4)
            ok8 &= np.array_equal(x[k, ..., 0], raw(d8.depth_files[i], r0, c0, size, size))
        for k, i in enumerate(fut):
            ok8 &= np.array_equal(y[k], raw(d8.depth_files[i], r0, c0, size, size))
        n8 += 1
    check(f"{name} {d8.pixel_size_m:.0f} m (scale {sc}, offset ({oy},{ox})): {n8} random windows equal independent block-averaged reads", ok8)

# ------------------------------------------------------------------ A7 NoData guard
print("\nA7. NoData guard: the UK layers have a NoData padding ring; a window that reaches it must be refused")
try:
    uk.load_patch_sequence(-5, -5, 20, 0, T_IN, HORIZON, STRIDE)   # DEM/rain window = ring + a little crop
    check("window reaching the NoData ring is refused", False)
except ValueError as e:
    check("window reaching the NoData ring is refused", "NoData" in str(e))
check("the patches the project actually uses are unchanged by the guard",
      len(valid_positions(au, PATCH, 1.0)) == 62 and len(valid_positions(uk, PATCH, 1.0)) == 2,
      f"(Australia {len(valid_positions(au, PATCH, 1.0))} / UK {len(valid_positions(uk, PATCH, 1.0))} valid patches)")

print("\n" + ("ALL DATA CHECKS PASSED" if not fails else "FAILED: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
