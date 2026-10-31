def build_forecaster(height=TILE_SIZE, width=TILE_SIZE):
    # Model A: CNN + ConvLSTM that predicts the depth map 30 minutes ahead

    # Input: 4 maps, each height x width, with 3 layers
    inputs = keras.Input(shape=(INPUT_STEPS, height, width, CHANNELS), name="past_maps")

    # CNN: the same 2 convolution layers are used on each of the 4 maps
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_1")(inputs)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_2")(x)

    # ConvLSTM: reads the 4 maps in order and keeps a memory map
    memory = layers.ConvLSTM2D(32, 3, padding="same", name="memory")(x)

    # Decoder: memory -> predicted change in depth (the last layer starts at zero)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="decoder")(memory)
    change = layers.Conv2D(1, 1, kernel_initializer="zeros", name="depth_change")(x)

    # Next map = last depth map + change, and never below 0
    last_depth = inputs[:, -1, :, :, 0:1]
    next_depth = layers.ReLU(name="next_depth")(layers.Add()([last_depth, change]))

    return keras.Model(inputs, next_depth, name="cnn_lstm_forecaster")


def forecaster_for_whole_map(trained_forecaster, region):
    # Build the same model at the size of a whole map and copy in the trained weights
    height = region["depth"].shape[1]
    width = region["depth"].shape[2]
    whole_map_forecaster = build_forecaster(height, width)
    whole_map_forecaster.set_weights(trained_forecaster.get_weights())
    return whole_map_forecaster


# Show the layers and the number of parameters
build_forecaster().summary()
