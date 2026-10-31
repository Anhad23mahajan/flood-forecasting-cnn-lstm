"""
Stage 2: train the depth-forecasting branch (model.py) on real FloodCastBench data
for one region (Australia, 30m high-fidelity), verifying the real pipeline works
mechanically - not a final large-scale training run (that's a later, GPU-scale step).
"""
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, ".")
from real_data_loader import RegionData
import model as flood_model
from train import CompiledTrainer

PATCH_SIZE = flood_model.PATCH_SIZE  # 128, matches model.py
T_IN, HORIZON = 5, 5
DEM_NODATA_STD_THRESHOLD = 1.0  # skip patches where DEM is suspiciously flat (no-data)


def build_examples(rd: RegionData, n_spatial: int, time_starts: list[int], seed: int = 0):
    """Enumerate non-overlapping spatial patch positions, filter out no-data (flat DEM)
    ones, pair with a handful of time windows, and return a shuffled example list."""
    import rasterio

    positions = []
    for row in range(0, rd.grid_height - PATCH_SIZE + 1, PATCH_SIZE):
        for col in range(0, rd.grid_width - PATCH_SIZE + 1, PATCH_SIZE):
            positions.append((row, col))

    with rasterio.open(rd.dem_path) as src:
        valid_positions = []
        for row, col in positions:
            dem_patch = src.read(1, window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE)))
            if dem_patch.std() > DEM_NODATA_STD_THRESHOLD:
                valid_positions.append((row, col))

    rng = np.random.default_rng(seed)
    rng.shuffle(valid_positions)
    chosen_positions = valid_positions[:n_spatial]

    examples = [(r, c, t) for (r, c) in chosen_positions for t in time_starts]
    rng.shuffle(examples)
    print(f"positions: {len(positions)} total, {len(valid_positions)} pass DEM no-data filter, "
          f"{len(chosen_positions)} sampled -> {len(examples)} (position, time) examples")
    return examples


def main():
    rd = RegionData(
        root_dir="data/extracted/FloodCastBench",
        region="Australia",
        fidelity_dir="High-fidelity flood forecasting/30m",
    )

    # time starts spanning dry -> flooded, so the model sees real variation, not all-zero targets
    n = len(rd.depth_files)
    time_starts = [int(n * f) for f in [0.35, 0.5, 0.65, 0.75]]
    time_starts = [t for t in time_starts if t + T_IN + HORIZON < n]

    examples = build_examples(rd, n_spatial=8, time_starts=time_starts)
    split = int(len(examples) * 0.8)
    train_examples, holdout_examples = examples[:split], examples[split:]
    print(f"train examples: {len(train_examples)}   holdout examples: {len(holdout_examples)}")

    def load(ex):
        row, col, start_idx = ex
        input_seq, depth_targets, severity, max_future = rd.load_patch_sequence(
            row, col, PATCH_SIZE, start_idx, T_IN, HORIZON
        )
        return input_seq, depth_targets

    print("loading train examples...")
    t0 = time.perf_counter()
    train_data = [load(ex) for ex in train_examples]
    print(f"  loaded {len(train_data)} in {time.perf_counter()-t0:.1f}s")
    print("loading holdout examples...")
    holdout_data = [load(ex) for ex in holdout_examples]

    encoder, decoder = flood_model.build_models(dropout=0.3)
    optimizer = flood_model.OPTIMIZERS["adam"](learning_rate=1e-3)
    trainer = CompiledTrainer(encoder, decoder, optimizer, horizon=HORIZON)

    epochs = 6
    print(f"\ntraining {epochs} epochs on {len(train_data)} real examples...")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        losses = [trainer.run(x, dt, training=True) for x, dt in train_data]
        print(f"  epoch {epoch}/{epochs}  train_loss(MSE)={np.mean(losses):.5f}  ({time.perf_counter()-t0:.1f}s)")

    print("\nevaluating on holdout...")
    holdout_mses = [trainer.run(x, dt, training=False) for x, dt in holdout_data]
    holdout_rmse_normalized = float(np.sqrt(np.mean(holdout_mses)))
    print(f"holdout MSE (mean over examples): {np.mean(holdout_mses):.5f}")
    print(f"holdout RMSE (in depth units, meters): {holdout_rmse_normalized:.4f} m")

    # save one holdout prediction for visual sanity check
    x_sample, depth_targets_sample = holdout_data[0]
    x_batch = x_sample[np.newaxis, ...].astype("float32")
    state_h, state_c, _cnn_features = encoder(tf.constant(x_batch), training=False)
    step_input = tf.ones((1, 1))
    preds = []
    for _ in range(HORIZON):
        depth_out, state_h, state_c = decoder([step_input, state_h, state_c], training=False)
        preds.append(depth_out.numpy()[0, ..., 0])
    preds = np.stack(preds)
    np.savez("stage2_holdout_sample.npz", pred=preds, truth=depth_targets_sample, input_hist=x_sample)
    print("saved stage2_holdout_sample.npz for visualization")


if __name__ == "__main__":
    main()
