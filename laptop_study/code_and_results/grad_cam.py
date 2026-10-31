"""
Stage 4: Grad-CAM (Selvaraju et al. 2017) for the depth-forecasting branch,
adapted from classification to dense regression - there's no single "class
score" to explain, so the target is the SUM of predicted depth over the output
patch at a chosen forecast step, a scalar standing in for "how much flood did
the model predict, in total." Explains cnn_encoder_3 (the encoder's last, most
semantically-rich conv layer - standard Grad-CAM practice), at the input
sequence's LAST timestep, matching how the capsule branch also reads the
encoder's most recent observed state.

Purely post-hoc: no retraining, gradients here are for the heatmap only and are
never applied to any weights.
"""
import numpy as np
import tensorflow as tf

import model as flood_model


def grad_cam(encoder, decoder, input_sequence: np.ndarray, horizon_step: int = 0):
    """input_sequence: (T_in, H, W, 3). Returns (heatmap, net_change_m): a (H, W) heatmap in
    [0,1], upsampled from the encoder's 16x16 feature-map resolution to the full input
    resolution, and the model's predicted net change in depth (metres, summed over the patch)
    at `horizon_step`. The sign of net_change_m says which direction the map explains:
    positive = evidence for the predicted rise, negative = evidence for the predicted fall."""
    x_batch = tf.constant(input_sequence[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        enc_out = encoder(x_batch, training=False)
        cnn_features = enc_out[2]
        tape.watch(cnn_features)  # the same tensor the decoder's skip connection reads, so both paths count

        frames = flood_model.decode_horizon(decoder, enc_out, x_batch, horizon_step + 1, training=False)
        # The decoder is persistence-anchored, so the gradient of the predicted depth w.r.t. the
        # encoder features reflects what drives the model's predicted CHANGE from the last
        # observed frame (the anchor itself carries no gradient to the features).
        #
        # Direction matters. Grad-CAM's ReLU keeps only evidence that INCREASES the target, so
        # if the model predicts net drying the plain map is entirely negative and ReLU blanks
        # it (measured on a real validation window: net change -0.11 m, 0% of cells positive).
        # So the target is signed by the predicted net change - decided outside the gradient
        # path - and the map shows evidence for the direction the model actually predicts.
        predicted_depth = tf.reduce_sum(frames[0, horizon_step])
        net_change = predicted_depth - tf.reduce_sum(flood_model.anchor_from_input(x_batch))
        direction = tf.stop_gradient(tf.where(net_change >= 0, 1.0, -1.0))
        target = direction * predicted_depth

    grads = tape.gradient(target, cnn_features)  # (1, T_in, 16, 16, 64)
    last_step_grads = grads[:, -1]                # (1, 16, 16, 64) - attribute to the most recent step
    last_step_features = cnn_features[:, -1]

    channel_weights = tf.reduce_mean(last_step_grads, axis=(0, 1, 2))  # (64,) global-avg-pool, per Grad-CAM
    weighted = tf.reduce_sum(last_step_features[0] * channel_weights, axis=-1)  # (16, 16)
    heatmap = tf.nn.relu(weighted)

    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-9)  # normalize to [0,1]
    heatmap_full_res = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis], (flood_model.PATCH_SIZE, flood_model.PATCH_SIZE), method="bilinear"
    )[0, ..., 0]

    return heatmap_full_res.numpy(), float(net_change)
