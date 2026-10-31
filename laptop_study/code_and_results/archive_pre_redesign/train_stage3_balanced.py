"""
Stage 3 real-data pass, take 2: the first attempt (train_stage3.py) sampled time
windows that happened to land overwhelmingly in the flood's sustained peak, so
48/56 examples were "severe" and the model just learned to always predict the
majority class (12/12 holdout predictions were "severe" - not real
discrimination). This version explicitly balances the severity classes instead
of just spreading samples across time/space and hoping.
"""
import sys
import time

import numpy as np
import rasterio
import tensorflow as tf

sys.path.insert(0, ".")
from real_data_loader import RegionData, SEVERITY_CLASSES, depth_to_severity
import model as flood_model

PATCH_SIZE = flood_model.PATCH_SIZE
T_IN, HORIZON = 5, 5
PER_CLASS_TARGET = 8


def find_balanced_examples(rd: RegionData, per_class_target: int):
    """Scan a broad grid of (time_fraction, position) combos, bucket by the
    severity that window would produce, and keep up to per_class_target of each -
    avoids the majority-class collapse the naive time-spread sampling caused."""
    n = len(rd.depth_files)
    positions = [(0, 600), (600, 0), (600, 1000), (300, 300), (300, 900),
                 (900, 300), (0, 0), (900, 900), (128, 500), (500, 128)]
    time_fracs = np.linspace(0.02, 0.97, 40)

    buckets = {c: [] for c in SEVERITY_CLASSES}
    for tf_frac in time_fracs:
        start_idx = int(n * tf_frac)
        if start_idx + T_IN + HORIZON >= n:
            continue
        check_idx = start_idx + T_IN + HORIZON - 1
        with rasterio.open(rd.depth_files[check_idx]) as src:
            for (row, col) in positions:
                cls = SEVERITY_CLASSES[len(buckets) - 1]  # placeholder, overwritten below
                if all(len(v) >= per_class_target for v in buckets.values()):
                    break
                patch = src.read(1, window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE)),
                                  boundless=True, fill_value=0)
                sev_idx = depth_to_severity(float(patch.max()))
                cls = SEVERITY_CLASSES[sev_idx]
                if len(buckets[cls]) < per_class_target:
                    buckets[cls].append((row, col, start_idx))
        if all(len(v) >= per_class_target for v in buckets.values()):
            break

    for c in SEVERITY_CLASSES:
        print(f"  {c}: found {len(buckets[c])}")
    return buckets


def main():
    rd = RegionData(
        root_dir="data/extracted/FloodCastBench",
        region="Australia",
        fidelity_dir="High-fidelity flood forecasting/30m",
    )

    print("scanning for class-balanced examples...")
    buckets = find_balanced_examples(rd, PER_CLASS_TARGET)

    all_examples = [(r, c, t, cls) for cls, lst in buckets.items() for (r, c, t) in lst]
    rng = np.random.default_rng(2)
    rng.shuffle(all_examples)

    print(f"\nloading {len(all_examples)} examples...")
    loaded = []
    for row, col, start_idx, expected_cls in all_examples:
        input_seq, depth_targets, severity, max_future = rd.load_patch_sequence(
            row, col, PATCH_SIZE, start_idx, T_IN, HORIZON
        )
        assert SEVERITY_CLASSES[severity] == expected_cls, (
            f"Mismatch: scan said {expected_cls}, full load says {SEVERITY_CLASSES[severity]} "
            f"(row={row},col={col},start_idx={start_idx}) - horizon window must differ from the scan's check point"
        )
        loaded.append((input_seq, depth_targets, severity))

    labels = [s for _, _, s in loaded]
    counts = {c: labels.count(i) for i, c in enumerate(SEVERITY_CLASSES)}
    print("final severity distribution:", counts, f"(n={len(labels)})")

    split = int(len(loaded) * 0.8)
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

    epochs = 12
    print(f"\ntraining {epochs} epochs on {len(train_data)} class-balanced real examples...")
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
    pred_counts = {c: 0 for c in SEVERITY_CLASSES}
    for input_seq, depth_targets, severity in holdout_data:
        dl, sl, pred_class = run_step(input_seq, depth_targets, severity, training=False)
        depth_losses_holdout.append(dl)
        correct += int(pred_class == severity)
        pred_counts[SEVERITY_CLASSES[pred_class]] += 1
        print(f"  true={SEVERITY_CLASSES[severity]:10s} pred={SEVERITY_CLASSES[pred_class]:10s} depth_loss={dl:.4f}")

    acc = correct / len(holdout_data) if holdout_data else float("nan")
    print(f"\nholdout depth RMSE: {np.sqrt(np.mean(depth_losses_holdout)):.4f} m")
    print(f"holdout severity accuracy: {correct}/{len(holdout_data)} = {acc:.2%}")
    print(f"prediction distribution: {pred_counts}  (majority-class collapse would show one class dominating)")


if __name__ == "__main__":
    main()
