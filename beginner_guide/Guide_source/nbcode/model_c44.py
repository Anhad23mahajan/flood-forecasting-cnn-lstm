# Small capsules: 4 types of 8 numbers, at each of 8 x 8 positions
PRIMARY_TYPES = 4
PRIMARY_SIZE = 8
PRIMARY_GRID = TILE_SIZE // 8


def build_severity_model(forecaster):
    # Capsule network on top of Model A's frozen CNN + ConvLSTM

    # Model A's CNN + ConvLSTM part, frozen
    encoder = keras.Model(forecaster.inputs, forecaster.get_layer("memory").output, name="cnn_lstm_encoder")
    encoder.trainable = False

    inputs = keras.Input(shape=(INPUT_STEPS, TILE_SIZE, TILE_SIZE, CHANNELS), name="past_tiles")
    memory = encoder(inputs)

    # Shrink 64 x 64 -> 32 x 32 -> 16 x 16
    x = layers.Conv2D(32, 5, strides=2, padding="same", activation="relu")(memory)
    x = layers.Conv2D(32, 5, strides=2, padding="same", activation="relu")(x)

    # Small capsules at 8 x 8 positions
    x = layers.Conv2D(PRIMARY_TYPES * PRIMARY_SIZE, 3, strides=2, padding="same")(x)
    x = layers.Reshape((PRIMARY_GRID * PRIMARY_GRID * PRIMARY_TYPES, PRIMARY_SIZE))(x)
    primary_capsules = Squash(name="primary_capsules")(x)

    # One capsule per severity class, and its length
    class_capsules = ClassCapsules(len(SEVERITY_NAMES), 16, name="class_capsules")(primary_capsules)
    lengths = CapsuleLength(name="class_lengths")(class_capsules)

    # The model, and a second version that also gives the memory (for Grad-CAM)
    model = keras.Model(inputs, lengths, name="capsule_severity")
    grad_cam_model = keras.Model(inputs, [memory, lengths])
    return model, grad_cam_model
