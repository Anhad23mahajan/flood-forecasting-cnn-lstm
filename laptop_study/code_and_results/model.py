"""
CNN -> LSTM encoder-decoder for spatiotemporal flood depth forecasting.

Input: (batch, T_in, H, W, C=3) - depth[t], static DEM, rainfall[t] per historical timestep (rainfall = intensity in mm/h of the interval containing t: partly future for the last frame, the 'hybrid' input).
Output: (batch, T_out, H, W) depth regression only.

This is the depth-forecasting branch only, per the current (Capsule Network + Symbolic
Rule Layer) architecture - the per-pixel dry/flooded classifier this module used to carry
predates that pivot and has been removed. Severity classification now lives entirely in
the separate Capsule Network branch (build_capsule_head, added in Stage 3), which reads
the same shared CNN-encoder features but is trained/evaluated independently of this module.

Architecture follows the dossier's Section 4.5 illustrative design (time-distributed CNN
encoder -> LSTM temporal encoder-decoder -> CNN decoder), itself following the
time-distributed-CNN-then-LSTM pattern from Malik et al. 2024 and the CNN/LSTM-fused-at-
bottleneck pattern from Wang et al. 2026 (see research dossier, Section 2.B).

CO mapping:
  CO1 - dense_bottleneck: Dense layer projecting CNN features to the LSTM's input vector
  CO2 - severity classification head: now the Capsule Network (see Stage 3), not this module
  CO3 - cnn_encoder_*/cnn_decoder_*: convolutional spatial feature extraction/reconstruction
  CO4 - dropout/optimizer/learning_rate exposed as build_models() parameters
  CO5 - encoder_lstm/decoder_lstm: the temporal encoder-decoder generating the forecast sequence
        (round 2: in the trained ORIGINAL model (v1) the depth history was not exploited - see PROJECT_LOG 'Completion, round 2'; the final ConvLSTM is model_v2.py)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from capsule_layers import PrimaryCaps, RoutingCaps, capsule_lengths

PATCH_SIZE = 128  # H = W = 128, per dossier Section 4.5
N_CHANNELS = 3    # depth, DEM, rainfall
NUM_SEVERITY_CLASSES = 4  # none / minor / moderate / severe, see real_data_loader.py

# Fixed (non-learned) scales so raw physical inputs feed the CNN sanely. Without these,
# DEM values (tens to hundreds of metres) swamp rainfall (0-9 mm/h) and depth (0-18 m).
DEPTH_SCALE_M = 5.0     # input depth is divided by this; the decoder multiplies back to metres
DEM_SCALE_M = 50.0      # DEM is made patch-relative (minus its own mean) then divided by this
RAIN_SCALE_MM = 5.0     # in mm/h (the rainfall files hold intensities)
DELTA_SCALE_M = 0.05    # the decoder's unit of "change from the last observed frame" is 5 cm


class InputNormalizer(layers.Layer):
    """Fixed per-channel scaling applied INSIDE the model, so every caller (training,
    predict_severity, Grad-CAM) keeps passing raw physical units and getting metres back.

    The DEM channel is patch-relative (mean removed) rather than absolute elevation: absolute
    elevation is a region-specific fingerprint that can't transfer to a new region, while
    relative relief (slopes, valleys) can - which matters for the cross-regional test."""

    def call(self, x):  # x: (batch, T, H, W, 3) raw units
        depth = x[..., 0:1] / DEPTH_SCALE_M
        dem = x[..., 1:2]
        dem_rel = (dem - tf.reduce_mean(dem, axis=(2, 3), keepdims=True)) / DEM_SCALE_M
        rain = x[..., 2:3] / RAIN_SCALE_MM
        return tf.concat([depth, dem_rel, rain], axis=-1)

OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "rmsprop": keras.optimizers.RMSprop,
    "sgd": keras.optimizers.SGD,
}


def build_encoder(hidden_dim: int = 128, dropout: float = 0.3) -> keras.Model:
    """Shared CNN encoder + LSTM temporal encoder. Outputs both the LSTM's final
    states (for the depth-forecasting decoder) AND the raw per-timestep CNN
    feature maps, BEFORE they're flattened into the LSTM - this second output is
    what the Capsule Network (Stage 3) consumes, so both branches genuinely share
    one encoder and both receive real gradients through it, rather than the
    Capsule Network operating on a separately re-computed copy."""
    inputs = keras.Input(shape=(None, PATCH_SIZE, PATCH_SIZE, N_CHANNELS), name="input_sequence")

    x = InputNormalizer(name="input_normalizer")(inputs)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_1")(x)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_2")(x)
    cnn_features = layers.TimeDistributed(layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_3")(x)
    # 128x128 -> 16x16x64 after three stride-2 blocks; cnn_features: (batch, T, 16, 16, 64)

    x = layers.TimeDistributed(layers.Flatten())(cnn_features)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    x = layers.TimeDistributed(layers.Dense(hidden_dim, activation="relu"), name="dense_bottleneck")(x)  # CO1

    _outputs, state_h, state_c = layers.LSTM(
        hidden_dim, return_sequences=True, return_state=True, dropout=dropout, name="encoder_lstm"
    )(x)

    return keras.Model(inputs, [state_h, state_c, cnn_features], name="encoder")


def build_decoder_step(hidden_dim: int = 128, dropout: float = 0.3) -> keras.Model:
    """One decode step. The LSTM takes a constant learned "step" input and relies entirely
    on its (state_h, state_c) carried from the previous step to produce a different output
    each time - standard for this kind of generative decoder, and much cheaper than
    re-encoding the full previous frame through a dense layer just to feed it back in.

    PERSISTENCE-ANCHORED (residual): the output is the most recent OBSERVED depth frame plus
    a learned change, not a depth field rebuilt from scratch. The first version rebuilt the
    whole 128x128 field from a 128-number LSTM state with no link to the observed frame, so
    it could never approach a near-exact copy - and a copy of the last frame is already
    accurate to ~2 mm at a 5-minute step, ~13 mm at 30 minutes (measured on real data).
    That, not only "too little data", is why the early predictions were blurry. The final
    conv is zero-initialized, so training starts exactly at the persistence forecast and
    can only improve on it."""
    step_input = keras.Input(shape=(1,), name="step_input")  # constant scalar, e.g. tf.ones((1,1))
    state_h_in = keras.Input(shape=(hidden_dim,), name="state_h")
    state_c_in = keras.Input(shape=(hidden_dim,), name="state_c")
    anchor_in = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1), name="anchor_depth")  # last observed depth, metres
    skip_in = keras.Input(shape=(16, 16, 64), name="skip_features")  # encoder's last-step spatial features

    step_emb = layers.Dense(hidden_dim, name="step_embedding")(step_input)
    step_emb_seq = layers.Reshape((1, hidden_dim))(step_emb)
    lstm_out, state_h, state_c = layers.LSTM(hidden_dim, return_state=True, name="decoder_lstm")(
        step_emb_seq, initial_state=[state_h_in, state_c_in]
    )
    lstm_out = layers.Dropout(dropout)(lstm_out)

    # project back to a 16x16x64 feature map, then upsample with transposed convs (mirrors the encoder)
    x = layers.Dense(16 * 16 * 64, activation="relu", name="decoder_bottleneck")(lstm_out)
    x = layers.Reshape((16, 16, 64))(x)
    # Skip connection: the LSTM state alone is a 128-number summary, so the decoder can express a
    # global drift but has no way to say WHERE anything changes. Concatenating the encoder's own
    # last-step spatial features gives every 16x16 cell access to the local observed structure.
    # (Measured: without this, held-out skill plateaued at exactly +2% from epoch 2 onward, and
    # was the same +2% on a region never seen in training - a constant drift, not learning.)
    x = layers.Concatenate(name="context_plus_skip")([x, skip_in])
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="cnn_decoder_1")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", name="cnn_decoder_2")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", name="cnn_decoder_3")(x)
    # 16x16 -> 32x32 -> 64x64 -> 128x128

    delta = layers.Conv2D(1, 1, activation="linear", kernel_initializer="zeros", name="depth_delta")(x)
    delta = layers.Rescaling(DELTA_SCALE_M, name="delta_to_meters")(delta)
    depth_out = layers.Add(name="anchor_plus_delta")([anchor_in, delta])               # (batch,128,128,1) metres

    return keras.Model(
        [step_input, state_h_in, state_c_in, anchor_in, skip_in],
        [depth_out, state_h, state_c],
        name="decoder_step",
    )


def anchor_from_input(x):
    """Most recent observed depth frame from a raw (batch, T, H, W, 3) input -> (batch, H, W, 1)."""
    return x[:, -1, :, :, 0:1]


def decode_horizon(decoder, enc_out, x, horizon: int, training: bool = False):
    """Runs the residual decoder for `horizon` steps. Returns (batch, horizon, H, W) depth in metres.

    enc_out: the encoder's outputs [state_h, state_c, cnn_features]; x: the SAME raw input batch
    the encoder was run on (its last depth frame is the persistence anchor). Every step is
    anchored to that frame and sees the encoder's last-step spatial features; the LSTM state
    evolves the change."""
    state_h, state_c, cnn_features = enc_out
    anchor_depth = anchor_from_input(x)
    skip = cnn_features[:, -1]
    step_input = tf.ones((tf.shape(state_h)[0], 1))
    frames = []
    sh, sc = state_h, state_c
    for _ in range(horizon):
        depth_out, sh, sc = decoder([step_input, sh, sc, anchor_depth, skip], training=training)
        frames.append(depth_out[..., 0])
    return tf.stack(frames, axis=1)


def build_capsule_head(num_primary_capsules: int = 8, primary_capsule_dim: int = 8,
                        severity_capsule_dim: int = 16, n_routing: int = 3,
                        return_lengths: bool = False) -> keras.Model:
    """Severity-classification branch (Stage 3): PrimaryCaps -> routing-by-agreement
    -> one capsule per severity class. Consumes the LAST timestep of the SAME
    cnn_features the encoder outputs - i.e. the shared encoder's own most-recent
    observed spatial representation, not a separately-computed copy - so this
    branch's gradients flow back into the real shared encoder weights.

    return_lengths=False (default): output softmax(lengths), for use with cross-entropy.
    return_lengths=True: output the raw capsule lengths in [0, 1) for use with
    capsule_layers.margin_loss (the loss the paper designed lengths for). argmax is the
    prediction either way."""
    cnn_feat_input = keras.Input(shape=(16, 16, 64), name="cnn_features_last_step")
    primary = PrimaryCaps(num_primary_capsules, primary_capsule_dim, kernel_size=3, strides=2)(cnn_feat_input)
    severity_caps = RoutingCaps(NUM_SEVERITY_CLASSES, severity_capsule_dim, n_routing=n_routing)(primary)
    lengths = layers.Lambda(capsule_lengths, name="capsule_lengths")(severity_caps)
    if return_lengths:
        return keras.Model(cnn_feat_input, lengths, name="capsule_severity_head")
    probs = layers.Softmax(name="severity_probs")(lengths)
    return keras.Model(cnn_feat_input, probs, name="capsule_severity_head")


def build_models(hidden_dim: int = 128, dropout: float = 0.3):
    return build_encoder(hidden_dim, dropout), build_decoder_step(hidden_dim, dropout)


def forecast(encoder, decoder, input_sequence: np.ndarray, horizon: int, hidden_dim: int = 128):
    """Real forward pass: encode T_in history, then decode `horizon` future frames."""
    x = tf.constant(input_sequence[np.newaxis, ...].astype("float32"))  # (1, T_in, H, W, C)
    enc_out = encoder(x, training=False)

    shapes = {"input": tuple(x.shape), "encoder_state_h": tuple(enc_out[0].shape)}
    frames = decode_horizon(decoder, enc_out, x, horizon, training=False)
    shapes["decoded_frames"] = tuple(frames.shape)
    return frames.numpy()[0], shapes


def dummy_training_step(encoder, decoder, input_sequence: np.ndarray, horizon: int,
                         optimizer_name: str = "adam", learning_rate: float = 1e-3, hidden_dim: int = 128):
    """One real gradient step against random fake targets - proves the loss/optimizer/dropout
    wiring (CO4) is mechanically correct. Not real training; the targets are random."""
    optimizer = OPTIMIZERS[optimizer_name](learning_rate=learning_rate)
    x = tf.constant(input_sequence[np.newaxis, ...].astype("float32"))
    rng = np.random.default_rng(0)
    fake_depth_targets = tf.constant(rng.random((horizon, PATCH_SIZE, PATCH_SIZE)).astype("float32"))

    trainable_vars = encoder.trainable_variables + decoder.trainable_variables

    with tf.GradientTape() as tape:
        enc_out = encoder(x, training=True)
        frames = decode_horizon(decoder, enc_out, x, horizon, training=True)
        loss = tf.reduce_mean(tf.square(frames[0] - fake_depth_targets))

    grads = tape.gradient(loss, trainable_vars)
    n_none = sum(1 for g in grads if g is None)
    optimizer.apply_gradients((g, v) for g, v in zip(grads, trainable_vars) if g is not None)

    return float(loss.numpy()), n_none, len(trainable_vars)
