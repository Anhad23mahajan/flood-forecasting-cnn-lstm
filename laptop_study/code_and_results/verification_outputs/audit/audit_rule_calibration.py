"""Are the symbolic rule thresholds on the right scale for the way the model is now run?

symbolic_rule_layer.py's thresholds (escalate above +0.0030 m/hr, de-escalate below -0.0001 m/hr) were derived from real
Australian ground truth at the ORIGINAL native 5-minute step. The model now runs at a 30-minute stride over a 2.5 h
horizon, and predict_severity.py computes depth-rate as (patch-mean of last forecast frame - patch-mean of last observed
frame) / hours. Data only, no model: compute that same quantity from GROUND TRUTH under both configurations and compare
it with the thresholds, by true severity class."""
import os
import sys

import numpy as np
import rasterio

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ".")
from real_data_loader import RegionData, SEVERITY_CLASSES, depth_to_severity  # noqa: E402
from windows import valid_positions  # noqa: E402

rd = RegionData(root_dir="data/extracted/FloodCastBench", region="Australia", fidelity_dir="High-fidelity flood forecasting/30m")
PATCH, T_IN, HORIZON = 128, 5, 5
positions = valid_positions(rd, PATCH, 1.0)
ESC, REC = 0.0030, -0.0001


def read(i):
    with rasterio.open(rd.depth_files[i]) as s:
        return s.read(1)


for stride in (1, 6):
    hours = HORIZON * stride * rd.depth_step_seconds / 3600.0
    starts = np.unique(np.linspace(0, rd.max_start_index(T_IN, HORIZON, stride), 40).astype(int))
    rows = {k: [] for k in range(4)}
    for st in starts:
        hist, fut = rd.window_frame_indices(int(st), T_IN, HORIZON, stride)
        last_obs = read(hist[-1])
        fut_frames = [read(i) for i in fut]
        fmax = np.max(fut_frames, axis=0)
        for (r, c) in positions:
            sl = (slice(r, r + PATCH), slice(c, c + PATCH))
            rate = (float(fut_frames[-1][sl].mean()) - float(last_obs[sl].mean())) / hours
            rows[depth_to_severity(float(fmax[sl].max()))].append(rate)
    print(f"\n=== stride {stride} ({stride * 5} min steps), horizon {hours:.2f} h, {len(starts)} start times x {len(positions)} patches")
    print(f"{'true class':>10s} {'n':>6s} {'mean rate':>11s} {'median':>9s} {'p90':>9s} {'> +0.0030':>10s} {'< -0.0001':>10s}")
    for k in range(4):
        a = np.array(rows[k])
        print(f"{SEVERITY_CLASSES[k]:>10s} {len(a):6d} {a.mean():+11.4f} {np.median(a):+9.4f} {np.percentile(a, 90):+9.4f} "
              f"{(a > ESC).mean():10.1%} {(a < REC).mean():10.1%}")
print("\nThresholds in symbolic_rule_layer.py: escalate > +0.0030 m/hr (with > 1000 m from water); de-escalate < -0.0001 m/hr.")
print("Derivation table in that file (patch-mean m/hr): none 0.0000, minor 0.0002, moderate 0.0028, severe 0.0035.")
