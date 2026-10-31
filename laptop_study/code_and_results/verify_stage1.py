import sys
sys.path.insert(0, ".")
from real_data_loader import RegionData, SEVERITY_CLASSES

rd = RegionData(
    root_dir="data/extracted/FloodCastBench",
    region="Australia",
    fidelity_dir="High-fidelity flood forecasting/30m",
)

patch_size = 64
t_in, horizon = 5, 5
start_idx = 2016  # ~t=604800s, well into the flood per the earlier scan

candidates = {
    "dry":      (600, 1000),
    "moderate": (600, 0),
    "wet":      (0, 600),
}

print(f"{'label':10} {'row':>5} {'col':>5}  {'hist_depth_max':>15} {'future_depth_max':>17} "
      f"{'DEM_range':>18} {'rainfall_range':>18}  severity")
for name, (row, col) in candidates.items():
    input_seq, depth_targets, severity, max_future = rd.load_patch_sequence(
        row, col, patch_size, start_idx, t_in, horizon
    )
    depth_hist = input_seq[..., 0]
    dem = input_seq[..., 1]
    rainfall = input_seq[..., 2]
    print(f"{name:10} {row:5d} {col:5d}  {depth_hist.max():15.3f} {max_future:17.3f}  "
          f"[{dem.min():6.2f},{dem.max():6.2f}]     [{rainfall.min():6.3f},{rainfall.max():6.3f}]      "
          f"{SEVERITY_CLASSES[severity]}")

    assert input_seq.shape == (t_in, patch_size, patch_size, 3), input_seq.shape
    assert depth_targets.shape == (horizon, patch_size, patch_size), depth_targets.shape

print("\nShape checks passed: input_sequence (t_in, 64, 64, 3), depth_targets (horizon, 64, 64)")
