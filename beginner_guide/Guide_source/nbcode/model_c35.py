# Model B counts its correction in centimetres
CHANGE_UNIT = 0.01


def build_forecaster_v2(trend_factor, height=TILE_SIZE, width=TILE_SIZE):
    # Model B: same layers as Model A, but it starts from the trend forecast

    inputs = keras.Input(shape=(INPUT_STEPS, height, width, CHANNELS), name="past_maps")

    # Same CNN + ConvLSTM + decoder as Model A
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_1")(inputs)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_2")(x)
    memory = layers.ConvLSTM2D(32, 3, padding="same", name="memory")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="decoder")(memory)
    correction = layers.Conv2D(1, 1, kernel_initializer="zeros", name="correction")(x)

    # Trend forecast = last map + beta x last change (never below 0)
    last_depth = inputs[:, -1, :, :, 0:1]
    previous_depth = inputs[:, -2, :, :, 0:1]
    last_change = layers.Subtract()([last_depth, previous_depth])
    trend = layers.ReLU()(layers.Add()([last_depth, layers.Rescaling(float(trend_factor))(last_change)]))

    # Next map = trend + correction in centimetres (never below 0)
    next_depth = layers.ReLU(name="next_depth")(layers.Add()([trend, layers.Rescaling(CHANGE_UNIT)(correction)]))

    return keras.Model(inputs, next_depth, name="cnn_lstm_forecaster_v2")


def loss_in_cm(true, predicted):
    # Mean squared error in centimetres (in metres the numbers are tiny)
    return tf.reduce_mean(tf.square((true - predicted) / CHANGE_UNIT))


def v2_for_whole_map(trained_forecaster, region, trend_factor):
    # Build Model B at the size of a whole map and copy in the trained weights
    whole_map_forecaster = build_forecaster_v2(trend_factor, region["depth"].shape[1], region["depth"].shape[2])
    whole_map_forecaster.set_weights(trained_forecaster.get_weights())
    return whole_map_forecaster


# Check: before training, Model B must give exactly the trend forecast
check_model = build_forecaster_v2(trend_factors["30m"])
sample = tiles["30m"]["validation_inputs"][:50]

last_maps = sample[:, -1, :, :, 0:1]
previous_maps = sample[:, -2, :, :, 0:1]
expected = np.maximum(last_maps + trend_factors["30m"] * (last_maps - previous_maps), 0)
got = check_model.predict(sample, verbose=0)

print("Parameters:", check_model.count_params())
print("Biggest difference from the trend forecast:", float(np.abs(got - expected).max()))
