"""
Stage 3, second check: a real combined training step - depth loss (decoder branch)
+ severity loss (capsule branch) backpropagating together through the ONE shared
encoder in a single tf.GradientTape, the way actual training will run. Confirms
the two branches don't interfere with each other mechanically (no shape clashes,
no duplicate-variable issues) and that every trainable variable across all three
sub-models (encoder, decoder, capsule_head) receives a gradient.
"""
import numpy as np
import tensorflow as tf

import model as flood_model

encoder = flood_model.build_encoder(hidden_dim=128, dropout=0.3)
decoder = flood_model.build_decoder_step(hidden_dim=128, dropout=0.3)
capsule_head = flood_model.build_capsule_head()

horizon = 3
x = np.random.rand(5, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE, flood_model.N_CHANNELS).astype("float32")
x_batch = tf.constant(x[np.newaxis, ...])
rng = np.random.default_rng(0)
fake_depth_targets = tf.constant(rng.random((horizon, flood_model.PATCH_SIZE, flood_model.PATCH_SIZE)).astype("float32"))
fake_severity_label = tf.constant([1])

all_vars = encoder.trainable_variables + decoder.trainable_variables + capsule_head.trainable_variables

with tf.GradientTape() as tape:
    enc_out = encoder(x_batch, training=True)
    cnn_features = enc_out[2]

    frames = flood_model.decode_horizon(decoder, enc_out, x_batch, horizon, training=True)
    depth_loss = tf.reduce_mean(tf.square(frames[0] - fake_depth_targets))

    severity_probs = capsule_head(cnn_features[:, -1], training=True)
    severity_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(fake_severity_label, severity_probs)
    )

    combined_loss = depth_loss + severity_loss

grads = tape.gradient(combined_loss, all_vars)
n_none = sum(1 for g in grads if g is None)
print(f"depth_loss={float(depth_loss):.4f}  severity_loss={float(severity_loss):.4f}  combined={float(combined_loss):.4f}")
print(f"total trainable variables (encoder+decoder+capsule_head): {len(all_vars)}")
print(f"non-None gradients: {len(all_vars)-n_none}/{len(all_vars)}")

if n_none > 0:
    print("\nVariables with NO gradient:")
    for v, g in zip(all_vars, grads):
        if g is None:
            print(" ", v.name, v.shape)
    raise SystemExit(f"\nFAILED: {n_none} variables got no gradient in the combined training step.")

print("\nPASSED: one combined training step correctly updates the shared encoder, the depth")
print("decoder, and the capsule severity head together - this is the real Stage 3 training step.")
