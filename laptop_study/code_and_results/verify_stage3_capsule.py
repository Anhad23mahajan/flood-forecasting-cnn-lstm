"""
Stage 3 verification: does the severity (Capsule Network) branch actually share
the CNN encoder with the depth-forecasting branch, in the sense that matters -
gradients from the severity loss ALONE (not mixed with the depth loss) reach
the shared encoder's own conv weights (cnn_encoder_1/2/3)?

This is the direct, mechanical proof that porting the capsule layer to
TensorFlow/Keras actually solved the original TF/PyTorch framework-split
problem, rather than just moving the same disconnect into one process.
"""
import numpy as np
import tensorflow as tf

import model as flood_model

encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
capsule_head = flood_model.build_capsule_head()

x = np.random.rand(5, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE, flood_model.N_CHANNELS).astype("float32")
x_batch = x[np.newaxis, ...]
fake_severity_label = tf.constant([2])  # "moderate"

conv_layer_names = ["cnn_encoder_1", "cnn_encoder_2", "cnn_encoder_3"]
encoder_conv_vars = []
for name in conv_layer_names:
    layer = encoder.get_layer(name)  # TimeDistributed-wrapped variables don't carry the
    encoder_conv_vars.extend(layer.trainable_variables)  # layer name in v.name, so fetch by layer instead
print(f"shared encoder conv variables found: {len(encoder_conv_vars)} (from layers {conv_layer_names})")
assert len(encoder_conv_vars) == 6, f"Expected 6 vars (kernel+bias x 3 conv layers), got {len(encoder_conv_vars)}"

with tf.GradientTape() as tape:
    state_h, state_c, cnn_features = encoder(x_batch, training=True)
    last_step_features = cnn_features[:, -1]  # (batch, 16, 16, 64) - most recent observed step
    severity_probs = capsule_head(last_step_features, training=True)
    severity_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(fake_severity_label, severity_probs)
    )

# Gradient of the SEVERITY loss ALONE (depth branch/decoder not even called here)
grads = tape.gradient(severity_loss, encoder_conv_vars)
n_none = sum(1 for g in grads if g is None)
print(f"\nseverity_loss = {float(severity_loss):.4f}")
print(f"encoder conv-layer gradients from severity loss alone: {len(grads)-n_none}/{len(grads)} non-None")
var_idx = 0
for name in conv_layer_names:
    for v in encoder.get_layer(name).trainable_variables:
        g = grads[var_idx]
        status = "NONE (BROKEN)" if g is None else f"norm={float(tf.norm(g)):.6f}"
        print(f"  {name}/{v.name:20s} shape={str(v.shape):18s} {status}")
        var_idx += 1

if n_none > 0:
    raise SystemExit(f"\nFAILED: {n_none} shared-encoder conv variables got no gradient from the severity loss.")

print("\nPASSED: severity-branch gradients genuinely reach the shared CNN encoder's own conv weights.")
print("This confirms the encoder is NOT just structurally shared but actually jointly trained -")
print("the exact thing that was impossible with the old PyTorch capsule + TensorFlow encoder split.")
