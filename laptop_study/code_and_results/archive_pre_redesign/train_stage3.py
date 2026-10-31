"""
Stage 3 real-data pass: train the FULL model (shared encoder -> depth-forecasting
branch + capsule severity branch) on real Australia data, combined loss, verify
both loss components move and check severity classification accuracy on holdout -
not a final large-scale run, same smoke-test scope as Stage 2's real-data pass.
"""
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, ".")
from real_data_loader import RegionData, SEVERITY_CLASSES
import model as flood_model

PATCH_SIZE = flood_model.PATCH_SIZE
T_IN, HORIZON = 5, 5
DEM_NODATA_STD_THRESHOLD = 1.0


def build_examples(rd: RegionData, n_spatial: int, time_starts: list[int], seed: int = 0):
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
    return examples


def main():
    rd = RegionData(
        root_dir="data/extracted/FloodCastBench",
        region="Australia",
        fidelity_dir="High-fidelity flood forecasting/30m",
    )
    n = len(rd.depth_files)
    # wider spread than Stage 2, specifically to get severity class diversity
    time_starts = [int(n * f) for f in [0.15, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]]
    time_starts = [t for t in time_starts if t + T_IN + HORIZON < n]

    examples = build_examples(rd, n_spatial=8, time_starts=time_starts)

    print("loading examples and checking severity label distribution...")
    loaded = []
    for row, col, start_idx in examples:
        input_seq, depth_targets, severity, max_future = rd.load_patch_sequence(
            row, col, PATCH_SIZE, start_idx, T_IN, HORIZON
        )
        loaded.append((input_seq, depth_targets, severity))

    labels = [s for _, _, s in loaded]
    counts = {c: labels.count(i) for i, c in enumerate(SEVERITY_CLASSES)}
    print("severity label distribution:", counts, f"(n={len(labels)})")

    split = int(len(loaded) * 0.8)
    rng = np.random.default_rng(1)
    rng.shuffle(loaded)
    train_data, holdout_data = loaded[:split], loaded[split:]
    print(f"train: {len(train_data)}  holdout: {len(holdout_data)}")

    encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
    decoder = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
    capsule_head = flood_model.build_capsule_head()
    optimizer = flood_model.OPTIMIZERS["adam"](learning_rate=1e-3)
    all_vars = encoder.trainable_variables + decoder.trainable_variables + capsule_head.trainable_variables
    step_input = tf.ones((1, 1))

    def run_step(input_seq, depth_targets, severity_label, training):
        x_batch = tf.constant(input_seq[np.newaxis, ...], dtype=tf.float32)
        depth_targets_t = tf.constant(depth_targets, dtype=tf.float32)
        severity_label_t = tf.constant([severity_label])

        with tf.GradientTape() as tape:
            state_h, state_c, cnn_features = encoder(x_batch, training=training)
            depth_loss_total = 0.0
            sh, sc = state_h, state_c
            for t in range(HORIZON):
                depth_out, sh, sc = decoder([step_input, sh, sc], training=training)
                depth_loss_total += tf.reduce_mean(tf.square(depth_out[0, ..., 0] - depth_targets_t[t]))
            depth_loss = depth_loss_total / HORIZON

            severity_probs = capsule_head(cnn_features[:, -1], training=training)
            severity_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(severity_label_t, severity_probs)
            )
            combined_loss = depth_loss + severity_loss

        if training:
            grads = tape.gradient(combined_loss, all_vars)
            optimizer.apply_gradients((g, v) for g, v in zip(grads, all_vars) if g is not None)

        pred_class = int(tf.argmax(severity_probs[0]))
        return float(depth_loss), float(severity_loss), pred_class

    epochs = 8
    print(f"\ntraining {epochs} epochs on {len(train_data)} real examples (combined depth+severity loss)...")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        depth_losses, severity_losses = [], []
        for input_seq, depth_targets, severity in train_data:
            dl, sl, _ = run_step(input_seq, depth_targets, severity, training=True)
            depth_losses.append(dl)
            severity_losses.append(sl)
        print(f"  epoch {epoch}/{epochs}  depth_loss={np.mean(depth_losses):.5f}  "
              f"severity_loss={np.mean(severity_losses):.5f}  ({time.perf_counter()-t0:.1f}s)")

    print("\nevaluating on holdout...")
    correct = 0
    depth_losses_holdout = []
    for input_seq, depth_targets, severity in holdout_data:
        dl, sl, pred_class = run_step(input_seq, depth_targets, severity, training=False)
        depth_losses_holdout.append(dl)
        correct += int(pred_class == severity)
        print(f"  true={SEVERITY_CLASSES[severity]:10s} pred={SEVERITY_CLASSES[pred_class]:10s} "
              f"depth_loss={dl:.4f}")

    acc = correct / len(holdout_data) if holdout_data else float("nan")
    print(f"\nholdout depth RMSE: {np.sqrt(np.mean(depth_losses_holdout)):.4f} m")
    print(f"holdout severity accuracy: {correct}/{len(holdout_data)} = {acc:.2%}")


if __name__ == "__main__":
    main()
