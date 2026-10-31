"""
Stage 5/5b verification: run predict_severity_with_reasoning() on real held-out
patches spanning all 4 severity classes. Checks: justification text is sensible,
severity only changes when a rule actually fires (i.e. reasoned == raw whenever
the justification says "no rule triggered"), and the features driving each
decision are real, physically-plausible numbers - not that the underlying
model's raw predictions are accurate (Stage 3/4 already established honestly
that they aren't yet, at this training scale).
"""
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, ".")
from real_data_loader import RegionData, SEVERITY_CLASSES
from train_stage3_balanced import find_balanced_examples, PATCH_SIZE, T_IN, HORIZON
import model as flood_model
from predict_severity import predict_severity_with_reasoning

rd = RegionData(
    root_dir="data/extracted/FloodCastBench",
    region="Australia",
    fidelity_dir="High-fidelity flood forecasting/30m",
)

print("gathering class-balanced examples...")
buckets = find_balanced_examples(rd, per_class_target=6)
all_examples = [(r, c, t) for cls, lst in buckets.items() for (r, c, t) in lst]
rng = np.random.default_rng(3)
rng.shuffle(all_examples)
split = int(len(all_examples) * 0.75)
train_examples, holdout_examples = all_examples[:split], all_examples[split:]
print(f"train: {len(train_examples)}  holdout: {len(holdout_examples)}")

encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
decoder = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
capsule_head = flood_model.build_capsule_head()
optimizer = flood_model.OPTIMIZERS["adam"](learning_rate=1e-3)
all_vars = encoder.trainable_variables + decoder.trainable_variables + capsule_head.trainable_variables
step_input = tf.ones((1, 1))

print("training briefly so the capsule head produces non-degenerate output...")
train_loaded = []
for row, col, start_idx in train_examples:
    seq, depth_targets, severity, _ = rd.load_patch_sequence(row, col, PATCH_SIZE, start_idx, T_IN, HORIZON)
    train_loaded.append((seq, depth_targets, severity))

for epoch in range(1, 7):
    t0 = time.perf_counter()
    for input_seq, depth_targets, severity in train_loaded:
        x_batch = tf.constant(input_seq[np.newaxis, ...], dtype=tf.float32)
        depth_targets_t = tf.constant(depth_targets, dtype=tf.float32)
        severity_t = tf.constant([severity])
        with tf.GradientTape() as tape:
            state_h, state_c, cnn_features = encoder(x_batch, training=True)
            depth_loss_total = 0.0
            sh, sc = state_h, state_c
            for t in range(HORIZON):
                depth_out, sh, sc = decoder([step_input, sh, sc], training=True)
                depth_loss_total += tf.reduce_mean(tf.square(depth_out[0, ..., 0] - depth_targets_t[t]))
            severity_probs = capsule_head(cnn_features[:, -1], training=True)
            severity_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(severity_t, severity_probs)
            )
            loss = depth_loss_total / HORIZON + severity_loss
        grads = tape.gradient(loss, all_vars)
        optimizer.apply_gradients((g, v) for g, v in zip(grads, all_vars) if g is not None)
    print(f"  epoch {epoch}/6 done ({time.perf_counter()-t0:.1f}s)")

print("\nrunning predict_severity_with_reasoning on holdout examples...\n")
for row, col, start_idx in holdout_examples:
    result = predict_severity_with_reasoning(
        encoder, decoder, capsule_head, rd, row, col, PATCH_SIZE, start_idx, T_IN, HORIZON
    )
    f = result["features"]
    print(f"true={result['true_severity']:10s} raw_capsule={result['raw_severity']:10s} "
          f"reasoned={result['reasoned_severity']:10s}")
    print(f"  features: depth_rate={f['depth_rate_m_per_hr']:.4f} m/hr  rainfall={f['rainfall_mm']:.2f} mm  "
          f"slope={f['slope_pct']:.2f}%  drainage_proximity={f['drainage_proximity_m']:.0f} m")
    print(f"  justification: {result['justification']}")

    # explicit consistency check the plan calls for
    no_rule_fired = "no rule triggered" in result["justification"]
    unchanged = result["raw_severity"] == result["reasoned_severity"]
    if no_rule_fired != unchanged:
        print(f"  *** INCONSISTENCY: justification says {'no rule fired' if no_rule_fired else 'a rule fired'} "
              f"but severity {'did not change' if unchanged else 'changed'} ***")
    print()
