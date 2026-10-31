"""
Window scanning and sampling for FloodCastBench regions.

A "window" is one (spatial patch, start time). Severity is computed EXACTLY the way
RegionData.load_patch_sequence computes it - max depth in the patch over the future
frames from RegionData.window_frame_indices - and both use that same method, so a scan
can never disagree with what the loader later returns.

Each scanned window is a tuple:
    (row, col, start_idx, severity, baseline_severity)
where baseline_severity is what you'd get by just applying the severity thresholds to the
max depth of the LAST OBSERVED frame (a "persistence" classifier that needs no model).
Measured on real Australia data, that trivial classifier is 99.9% accurate at a 5-minute
step and 96.7% at a 30-minute stride - so any severity model has to be judged against it.
"""
import numpy as np
import rasterio

from real_data_loader import SEVERITY_CLASSES, RegionData, depth_to_severity, has_nodata


def all_positions(rd: RegionData, patch_size: int):
    """Non-overlapping patch origins that fit entirely inside the depth grid."""
    return [(r, c) for r in range(0, rd.grid_height - patch_size + 1, patch_size)
                   for c in range(0, rd.grid_width - patch_size + 1, patch_size)]


def valid_positions(rd: RegionData, patch_size: int, dem_nodata_std: float = 1.0):
    """Drop patches whose DEM is (near-)flat - those lie outside the simulation domain - and patches whose
    DEM window touches the NoData sentinel (its std is enormous, so the flat-DEM test alone would keep it)."""
    keep = []
    for (r, c) in all_positions(rd, patch_size):
        dem = rd.load_dem_patch(r, c, patch_size)
        if not has_nodata(dem) and dem.std() > dem_nodata_std:
            keep.append((r, c))
    return keep


def start_indices_for(rd: RegionData, count: int, t_in: int, horizon: int, stride: int):
    """`count` start indices spread evenly over every start at which a whole window fits."""
    last = rd.max_start_index(t_in, horizon, stride)
    return np.unique(np.linspace(0, last, count).astype(int))


def scan_windows(rd: RegionData, positions, start_indices, patch_size: int,
                 t_in: int, horizon: int, stride: int):
    out = []
    for start_idx in start_indices:
        history_idx, future_idx = rd.window_frame_indices(int(start_idx), t_in, horizon, stride)
        with rasterio.open(rd.depth_files[history_idx[-1]]) as src:
            last_observed = src.read(1)
        future_max = None
        for i in future_idx:
            with rasterio.open(rd.depth_files[i]) as src:
                frame = src.read(1)
            future_max = frame if future_max is None else np.maximum(future_max, frame)
        for (row, col) in positions:
            sl = (slice(row, row + patch_size), slice(col, col + patch_size))
            out.append((row, col, int(start_idx),
                        depth_to_severity(float(future_max[sl].max())),
                        depth_to_severity(float(last_observed[sl].max()))))
    return out


def class_fractions(windows):
    total = len(windows)
    return {c: sum(1 for w in windows if w[3] == i) / total for i, c in enumerate(SEVERITY_CLASSES)}


def balanced_sample(pool, per_class: int, rng):
    """Up to `per_class` randomly-chosen windows of each severity class."""
    chosen = []
    for cls_idx, cls in enumerate(SEVERITY_CLASSES):
        members = [w for w in pool if w[3] == cls_idx]
        take = min(per_class, len(members))
        if take < per_class:
            print(f"  WARNING: only {len(members)} '{cls}' windows available (wanted {per_class})")
        pick = rng.choice(len(members), size=take, replace=False)
        chosen.extend(members[i] for i in pick)
    return chosen
