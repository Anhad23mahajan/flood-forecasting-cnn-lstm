# Load (or train) the capsule network for each map scale
severity_models = {}
severity_grad_cam_models = {}

for pair in PAIRS:
    data = tiles[pair["name"]]
    weights_path = MODEL_DIR + "/capsule_" + pair["name"] + ".weights.h5"

    model, grad_cam_model = build_severity_model(forecasters[pair["name"]])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss=margin_loss,
                  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])

    if TRAIN_NEW or not os.path.exists(weights_path):
        print("Training the capsule network", pair["name"])

        # Rare classes count more, so the network can't just answer "None"
        counts = np.bincount(data["train_labels"], minlength=4)
        class_weights = len(data["train_labels"]) / (4 * np.maximum(counts, 1))
        sample_weights = class_weights[data["train_labels"]]

        stop_early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(
            data["train_inputs"], keras.utils.to_categorical(data["train_labels"], 4),
            sample_weight=sample_weights,
            validation_data=(data["validation_inputs"], keras.utils.to_categorical(data["validation_labels"], 4)),
            epochs=30, batch_size=32, callbacks=[stop_early], verbose=2)

        model.save_weights(weights_path)
    else:
        print("Loading the capsule network", pair["name"])
        model.load_weights(weights_path)

    severity_models[pair["name"]] = model
    severity_grad_cam_models[pair["name"]] = grad_cam_model
