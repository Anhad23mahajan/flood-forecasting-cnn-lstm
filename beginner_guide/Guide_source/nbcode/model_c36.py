# Load (or train) Model B for each map scale
v2_forecasters = {}

for pair in PAIRS:
    data = tiles[pair["name"]]
    weights_path = MODEL_DIR + "/forecaster_v2_" + pair["name"] + ".weights.h5"

    forecaster = build_forecaster_v2(trend_factors[pair["name"]])
    forecaster.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss=loss_in_cm)

    if TRAIN_V2 or not os.path.exists(weights_path):
        print("Training Model B", pair["name"])

        # Error before training = error of the trend forecast
        start_loss = float(np.ravel(forecaster.evaluate(
            data["validation_inputs"], data["validation_targets"], batch_size=32, verbose=0))[0])
        start_weights = forecaster.get_weights()

        keras.utils.set_random_seed(42)
        stop_early = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        history = forecaster.fit(
            data["train_inputs"], data["train_targets"],
            validation_data=(data["validation_inputs"], data["validation_targets"]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop_early], verbose=2)

        # If training did not help, keep the plain trend forecast
        best_loss = min(history.history["val_loss"])
        if best_loss >= start_loss:
            forecaster.set_weights(start_weights)

        forecaster.save_weights(weights_path)
    else:
        print("Loading Model B", pair["name"])
        forecaster.load_weights(weights_path)

    v2_forecasters[pair["name"]] = forecaster
