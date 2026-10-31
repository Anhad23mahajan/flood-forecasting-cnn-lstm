"""
CNN -> ConvLSTM encoder-decoder (architecture v2) for spatiotemporal flood-depth forecasting.

WHY v2 EXISTS.  A per-pixel LINEAR model that sees only the depth history (last frame, last-minus-previous, last-minus-first) scores
skill +0.33 .. +0.36 against persistence on every experiment, about DOUBLE the v1 CNN-LSTM in model.py (+0.05 .. +0.23 on the same tasks,
+0.18 on the never-seen UK at 60 m).  The depth history is very informative, and v1 could not use it, for structural reasons:
  * v1 flattens each frame's 16x16x64 features into ONE dense layer (128 numbers) before the LSTM: no per-pixel temporal difference can
    survive that bottleneck (and depth was scaled by 5 m, so millimetre-scale changes were ~1e-4 in network units);
  * its decoder's only spatial information is the LAST frame's features (skip), never a difference between frames.
v2 keeps what makes it a CNN-LSTM but removes both obstacles:
  * the recurrence is a ConvLSTM2D over the 16x16x64 CNN feature maps (spatial structure preserved; the temporal state is a MAP, not a vector);
  * multi-scale TEMPORAL skips: the 64x64 and 32x32 features of ALL input steps are stacked along channels and convolved, so the decoder can
    form frame-to-frame differences at fine resolution;
  * an explicit frame-to-frame depth-difference channel (in units of 1 cm) is added to the input;
  * the decoder is a ConvLSTM that continues the encoder state for `horizon` steps; a shared upsampling head (with the skips) turns each
    step's state into a change map; output = last observed depth + change (persistence-anchored, zero-initialised final conv, exactly as v1).
The capsule severity branch still reads the last step's 16x16x64 CNN features (`cnn_features`), so the two-branch design is unchanged.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PATCH_SIZE = 128
DEPTH_SCALE_M = 5.0
DEM_SCALE_M = 50.0
RAIN_SCALE_MM = 5.0
DIFF_SCALE_M = 0.01     # frame-to-frame depth difference in units of 1 cm
DELTA_SCALE_M = 0.05

OPTIMIZERS = {"adam": keras.optimizers.Adam}


class InputNormalizerV2(layers.Layer):
    """(B, T, H, W, 3) raw [depth m, DEM m, rain mm/h] -> (B, T, H, W, 4): depth/5, patch-relative DEM/50, rain/5, depth difference/1 cm."""

    def call(self, x):
        depth, dem, rain = x[..., 0:1], x[..., 1:2], x[..., 2:3]
        dem_rel = (dem - tf.reduce_mean(dem, axis=(2, 3), keepdims=True)) / DEM_SCALE_M
        prev = tf.concat([depth[:, :1], depth[:, :-1]], axis=1)
        return tf.concat([depth / DEPTH_SCALE_M, dem_rel, rain / RAIN_SCALE_MM, (depth - prev) / DIFF_SCALE_M], axis=-1)


class StackTime(layers.Layer):
    """(B, T, H, W, C) -> (B, H, W, T*C): all time steps side by side along channels (a fixed T)."""

    def call(self, x):
        s = tf.shape(x)
        x = tf.transpose(x, [0, 2, 3, 1, 4])
        return tf.reshape(x, [s[0], s[2], s[3], x.shape[3] * x.shape[4]])


class LinFeatures(layers.Layer):
    """Full-resolution local features for the (optional) local-CNN path, built from the NORMALISED input (B, T, H, W, 4):
    last observed depth, patch-relative DEM, last one-step depth change, total change over the history, wet indicator and the
    rainfall channel of every input step.  -> (B, H, W, T + 5)."""

    def __init__(self, last_obs, **kw):
        super().__init__(**kw)
        self.last_obs = last_obs

    def call(self, x):
        d_last = x[:, self.last_obs, ..., 0:1]
        dem = x[:, self.last_obs, ..., 1:2]
        diff1 = x[:, self.last_obs, ..., 3:4]
        trend = tf.reduce_sum(x[..., 3], axis=1)[..., None]
        wet = tf.cast(d_last * DEPTH_SCALE_M > 0.01, x.dtype)
        rain = tf.transpose(x[..., 2], [0, 2, 3, 1])                  # (B, H, W, T)
        return tf.concat([d_last, dem, diff1, trend, wet, rain], axis=-1)

    def get_config(self):
        return {**super().get_config(), "last_obs": self.last_obs}


class ZerosSeq(layers.Layer):
    """(B, h, w, c) -> zeros (B, horizon, h, w, 1): the constant 'step input' of the decoder ConvLSTM (its state carries the information)."""

    def __init__(self, horizon, **kw):
        super().__init__(**kw)
        self.horizon = horizon

    def call(self, x):
        s = tf.shape(x)
        return tf.zeros([s[0], self.horizon, x.shape[1], x.shape[2], 1], x.dtype)

    def get_config(self):
        return {**super().get_config(), "horizon": self.horizon}


class TileTime(layers.Layer):
    """(B, h, w, c) -> (B, horizon, h, w, c): the same static skip map for every forecast step."""

    def __init__(self, horizon, **kw):
        super().__init__(**kw)
        self.horizon = horizon

    def call(self, x):
        return tf.repeat(x[:, None], self.horizon, axis=1)

    def get_config(self):
        return {**super().get_config(), "horizon": self.horizon}


def build_encoder(T, hidden_dim=64, dropout=0.3, local_path=False, last_obs=4):
    """Outputs [state_h (16,16,hidden), state_c, cnn_features (T,16,16,64), skip64 (64,64,32), skip32 (32,32,32)]."""
    inputs = keras.Input(shape=(T, PATCH_SIZE, PATCH_SIZE, 3), name="input_sequence")
    x = InputNormalizerV2(name="input_normalizer")(inputs)
    c1 = layers.TimeDistributed(layers.Conv2D(16, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_1")(x)      # 64x64
    c2 = layers.TimeDistributed(layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_2")(c1)     # 32x32
    c3 = layers.TimeDistributed(layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"), name="cnn_encoder_3")(c2)     # 16x16
    _, h, c = layers.ConvLSTM2D(hidden_dim, 3, padding="same", return_sequences=True, return_state=True, dropout=dropout,
                                name="encoder_convlstm")(c3)
    s1 = layers.Conv2D(32, 3, padding="same", activation="relu", name="temporal_skip_64")(StackTime(name="stack64")(c1))
    s2 = layers.Conv2D(32, 3, padding="same", activation="relu", name="temporal_skip_32")(StackTime(name="stack32")(c2))
    outs = [h, c, c3, s1, s2]
    if local_path:
        outs.append(LinFeatures(last_obs, name="local_features")(x))
    return keras.Model(inputs, outs, name="encoder_v2")


def build_decoder(horizon, hidden_dim=64, dropout=0.3, local_channels=0):
    h_in = keras.Input(shape=(16, 16, hidden_dim), name="state_h")
    c_in = keras.Input(shape=(16, 16, hidden_dim), name="state_c")
    skip16 = keras.Input(shape=(16, 16, 64), name="skip16")
    skip32 = keras.Input(shape=(32, 32, 32), name="skip32")
    skip64 = keras.Input(shape=(64, 64, 32), name="skip64")
    lin_in = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, local_channels), name="local_features") if local_channels else None
    anchor = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1), name="anchor_depth")

    steps = layers.ConvLSTM2D(hidden_dim, 3, padding="same", return_sequences=True, name="decoder_convlstm")(
        ZerosSeq(horizon, name="step_input")(h_in), initial_state=[h_in, c_in])                      # (B, H, 16, 16, hidden)
    x = layers.Concatenate(name="context_plus_skip16")([steps, TileTime(horizon, name="tile16")(skip16)])
    x = layers.Dropout(dropout)(x)
    x = layers.TimeDistributed(layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"), name="cnn_decoder_1")(x)   # 32
    x = layers.Concatenate(name="skip32_cat")([x, TileTime(horizon, name="tile32")(skip32)])
    x = layers.TimeDistributed(layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"), name="cnn_decoder_2")(x)   # 64
    x = layers.Concatenate(name="skip64_cat")([x, TileTime(horizon, name="tile64")(skip64)])
    x = layers.TimeDistributed(layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu"), name="cnn_decoder_3")(x)   # 128
    delta = layers.TimeDistributed(layers.Conv2D(1, 1, kernel_initializer="zeros"), name="depth_delta")(x)
    delta = layers.Rescaling(DELTA_SCALE_M, name="delta_to_meters")(delta)
    terms = [TileTime(horizon, name="tile_anchor")(anchor), delta]
    if lin_in is not None:                                        # local CNN path: full-resolution, no recurrence, one map per forecast step
        z = layers.Conv2D(16, 5, padding="same", activation="relu", name="local_conv")(lin_in)
        z = layers.Conv2D(horizon, 1, kernel_initializer="zeros", name="local_delta")(z)                  # (B, 128, 128, H)
        z = layers.Rescaling(DELTA_SCALE_M, name="local_to_meters")(z)
        terms.append(layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2])[..., None], name="local_time_first")(z))   # (B, H, 128, 128, 1)
    out = layers.Add(name="anchor_plus_delta")(terms)                                                        # (B, H, 128, 128, 1)
    ins = [h_in, c_in, skip16, skip32, skip64] + ([lin_in] if lin_in is not None else []) + [anchor]
    return keras.Model(ins, out, name="decoder_v2")


def decode_horizon(decoder, enc_out, x, horizon, training=False):
    """(B, horizon, 128, 128) depth in metres. The anchor is the last observed depth frame (for the forced regime the last input
    steps repeat it, so x[:, -1] is the same frame)."""
    h, c, c3, s1, s2 = enc_out[:5]
    anchor = x[:, -1, :, :, 0:1]
    ins = [h, c, c3[:, -1], s2, s1] + ([enc_out[5]] if len(enc_out) > 5 else []) + [anchor]
    y = decoder(ins, training=training)
    return y[..., 0]


# ------------------------------------------------------------------------------------------------- local-only control (no recurrence)
def build_local(T, horizon, last_obs=4, kernel=5):
    """CONTROL MODEL: only the full-resolution local CNN path (LinFeatures -> Conv5x5(16, relu) -> Conv1x1(horizon), zero-initialised),
    added to the persistence anchor.  No ConvLSTM, no encoder, no skips.  Comparing v2lin with this model shows what the ConvLSTM adds."""
    inputs = keras.Input(shape=(T, PATCH_SIZE, PATCH_SIZE, 3), name="input_sequence")
    lin = LinFeatures(last_obs, name="local_features")(InputNormalizerV2(name="input_normalizer")(inputs))
    encoder = keras.Model(inputs, [lin], name="encoder_local")
    lin_in = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, T + 5), name="local_features_in")
    anchor = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1), name="anchor_depth")
    z = layers.Conv2D(16, kernel, padding="same", activation="relu", name="local_conv")(lin_in)      # kernel 1 = a per-pixel network (no spatial context)
    z = layers.Conv2D(horizon, 1, kernel_initializer="zeros", name="local_delta")(z)
    z = layers.Rescaling(DELTA_SCALE_M, name="local_to_meters")(z)
    z = layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2])[..., None], name="local_time_first")(z)
    out = layers.Add(name="anchor_plus_delta")([layers.Lambda(lambda a: tf.repeat(a[:, None], horizon, axis=1), name="tile_anchor")(anchor), z])
    return encoder, keras.Model([lin_in, anchor], out, name="decoder_local")


def decode_horizon_local(decoder, enc_out, x, horizon, training=False):
    lin = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out            # a one-output Keras model returns the tensor itself
    return decoder([lin, x[:, -1, :, :, 0:1]], training=training)[..., 0]


class SliceTime(layers.Layer):
    """(B, T, H, W, C) -> the first `n` time steps."""

    def __init__(self, n, **kw):
        super().__init__(**kw)
        self.n = n

    def call(self, x):
        return x[:, :self.n]

    def get_config(self):
        return {**super().get_config(), "n": self.n}


class FutureRainChannels(layers.Layer):
    """(B, T, H, W, 4) normalised input -> (B, H, W, T - start): the rain channel of the steps AFTER `start`, as side channels."""

    def __init__(self, start, **kw):
        super().__init__(**kw)
        self.start = start

    def call(self, x):
        return tf.transpose(x[:, self.start:, ..., 2], [0, 2, 3, 1])

    def get_config(self):
        return {**super().get_config(), "start": self.start}


def build_clstm(T, horizon, filters=8, kernel=3, future_steps=0):
    """SMALL FULL-RESOLUTION CNN-LSTM: a ConvLSTM2D (`filters` channels, kernel x kernel) runs over the observed input steps at the native 128x128
    resolution; a 3x3 conv + 1x1 conv turns its final hidden map into `horizon` change maps (zero-initialised), added to the persistence
    anchor.  Recurrent like a CNN-LSTM, but with few parameters and no low-resolution bottleneck, so per-pixel temporal change is visible.
    future_steps > 0 (the FORCED regime, T = 5 + future_steps): the ConvLSTM sees only the first T - future_steps steps; the rainfall of the future steps
    is concatenated to its hidden map as `future_steps` side channels (code review F4: five extra time steps handicap the recurrence)."""
    inputs = keras.Input(shape=(T, PATCH_SIZE, PATCH_SIZE, 3), name="input_sequence")
    x = InputNormalizerV2(name="input_normalizer")(inputs)
    hist = SliceTime(T - future_steps, name="observed_steps")(x) if future_steps else x
    h = layers.ConvLSTM2D(filters, kernel, padding="same", return_sequences=False, name="encoder_convlstm")(hist)          # (B, 128, 128, filters)
    if future_steps:
        h = layers.Concatenate(name="hidden_plus_future_rain")([h, FutureRainChannels(T - future_steps, name="future_rain")(x)])
    encoder = keras.Model(inputs, [h], name="encoder_clstm")
    f_in = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, filters + future_steps), name="hidden_map")
    anchor = keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1), name="anchor_depth")
    z = layers.Conv2D(16, 3, padding="same", activation="relu", name="head_conv")(f_in)
    z = layers.Conv2D(horizon, 1, kernel_initializer="zeros", name="head_delta")(z)
    z = layers.Rescaling(DELTA_SCALE_M, name="delta_to_meters")(z)
    z = layers.Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2])[..., None], name="time_first")(z)
    out = layers.Add(name="anchor_plus_delta")([layers.Lambda(lambda a: tf.repeat(a[:, None], horizon, axis=1), name="tile_anchor")(anchor), z])
    return encoder, keras.Model([f_in, anchor], out, name="decoder_clstm")
