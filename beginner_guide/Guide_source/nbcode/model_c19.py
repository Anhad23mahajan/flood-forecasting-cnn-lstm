# Load (or train) Model A for each map scale
forecasters = {}
histories = {}

for pair in PAIRS:
    data = tiles[pair["name"]]
    weights_path = MODEL_DIR + "/forecaster_" + pair["name"] + ".weights.h5"

    forecaster = build_forecaster()
    forecaster.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")

    if TRAIN_NEW or not os.path.exists(weights_path):
        print("Training Model A", pair["name"])

        # Stop when the checking error has not improved for 4 epochs, and keep the best weights
        stop_early = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        history = forecaster.fit(
            data["train_inputs"], data["train_targets"],
            validation_data=(data["validation_inputs"], data["validation_targets"]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop_early], verbose=2)

        histories[pair["name"]] = history.history
        forecaster.save_weights(weights_path)
    else:
        print("Loading Model A", pair["name"])
        forecaster.load_weights(weights_path)

    forecasters[pair["name"]] = forecaster
