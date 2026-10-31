# ===== CELL 1 of 3: build the version-2 forecaster and check that it starts exactly as the trend forecast =====
# Run this in the SAME Colab session, in your COPY of the notebook, below the baseline cell.
# It needs the baseline cell to have been run (it uses fit_trend_factor).

# The network's correction is counted in centimetres, so the numbers it works with are not tiny
CHANGE_UNIT = 0.01


def build_forecaster_v2(trend_factor, height=TILE_SIZE, width=TILE_SIZE):
    """Same CNN + ConvLSTM as before, but it starts from the trend forecast instead of from 'copy the last map'.
    The network only learns a correction to the trend. Its last layer starts at zero,
    so the untrained model IS the trend forecast and training can only improve on it."""

    inputs = keras.Input(shape=(INPUT_STEPS, height, width, CHANNELS), name="past_maps")

    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_1")(inputs)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, padding="same", activation="relu"), name="cnn_2")(x)
    memory = layers.ConvLSTM2D(32, 3, padding="same", name="memory")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="decoder")(memory)
    correction = layers.Conv2D(1, 1, kernel_initializer="zeros", name="correction")(x)

    # Trend forecast: the last map plus the change of the last 30 minutes, kept going (never below 0)
    last_depth = inputs[:, -1, :, :, 0:1]
    previous_depth = inputs[:, -2, :, :, 0:1]
    last_change = layers.Subtract()([last_depth, previous_depth])
    trend = layers.ReLU()(layers.Add()([last_depth, layers.Rescaling(float(trend_factor))(last_change)]))

    # Next map = trend forecast + the network's correction (in centimetres), never below 0
    next_depth = layers.ReLU(name="next_depth")(layers.Add()([trend, layers.Rescaling(CHANGE_UNIT)(correction)]))

    return keras.Model(inputs, next_depth, name="cnn_lstm_forecaster_v2")


def loss_in_cm(true, predicted):
    """Mean squared error counted in centimetres (a plain mean squared error in metres is a very small number)."""

    return tf.reduce_mean(tf.square((true - predicted) / CHANGE_UNIT))


# The trend number of each model, from the training part of its training flood
trend_factors = {}

for pair in PAIRS:
    train_region = regions[pair["train"]]
    trend_factors[pair["name"]] = fit_trend_factor(train_region, FIRST_TARGET, train_region["train_end"])

print("Trend numbers:", {name: round(value, 3) for name, value in trend_factors.items()})

# Safety check: the untrained version-2 model must give exactly the trend forecast
check_model = build_forecaster_v2(trend_factors["30m"])
sample = tiles["30m"]["validation_inputs"][:50]

last_maps = sample[:, -1, :, :, 0:1]
previous_maps = sample[:, -2, :, :, 0:1]
expected = np.maximum(last_maps + trend_factors["30m"] * (last_maps - previous_maps), 0)
got = check_model.predict(sample, verbose=0)

largest_difference = float(np.abs(got - expected).max())
print("Parameters:", check_model.count_params(), "(the first version has 62,833)")
print("Largest difference between the untrained model and the trend forecast:", largest_difference)

assert largest_difference < 1e-5, "STOP: the untrained version-2 model is not the trend forecast. Send this message to Anhad."
print("OK: the check passed")


# ===== CELL 2 of 3: train the two version-2 forecasters (saved under NEW names, the first version is not touched) =====

# True: train again and overwrite the version-2 weights. False: load them if they are already saved.
TRAIN_V2 = False

keras.utils.set_random_seed(42)

v2_forecasters = {}

for pair in PAIRS:
    data = tiles[pair["name"]]
    weights_path = os.path.join(MODEL_DIR, "forecaster_v2_" + pair["name"] + ".weights.h5")

    forecaster = build_forecaster_v2(trend_factors[pair["name"]])
    forecaster.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss=loss_in_cm)

    if TRAIN_V2 or not os.path.exists(weights_path):
        print("Training the version-2", pair["name"], "forecaster on", REGION_NAMES[pair["train"]])

        # Error on the validation tiles before any training = the error of the trend forecast itself
        start_loss = float(np.ravel(forecaster.evaluate(
            data["validation_inputs"], data["validation_targets"], batch_size=32, verbose=0))[0])
        start_weights = forecaster.get_weights()

        stop_early = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        history = forecaster.fit(
            data["train_inputs"], data["train_targets"],
            validation_data=(data["validation_inputs"], data["validation_targets"]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[stop_early], verbose=2)

        best_loss = min(history.history["val_loss"])

        # Training must not make things worse than the trend forecast it started from
        if best_loss >= start_loss:
            forecaster.set_weights(start_weights)
            print("Training did not beat the trend forecast on the validation tiles, so the trend forecast is kept.")

        print("Validation error (cm), trend forecast:", round(float(np.sqrt(start_loss)), 4),
              "| after training:", round(float(np.sqrt(min(best_loss, start_loss))), 4))

        forecaster.save_weights(weights_path)
    else:
        print("Loading the saved version-2", pair["name"], "forecaster")
        forecaster.load_weights(weights_path)

    v2_forecasters[pair["name"]] = forecaster


# ===== CELL 3 of 3: score version 2 on whole maps, exactly like the first version, and put it next to the trend forecast =====

def v2_for_whole_map(trained_forecaster, region, trend_factor):
    """Builds the version-2 forecaster at the size of a whole map and copies in the weights trained on tiles."""

    whole_map_forecaster = build_forecaster_v2(trend_factor, region["depth"].shape[1], region["depth"].shape[2])
    whole_map_forecaster.set_weights(trained_forecaster.get_weights())

    return whole_map_forecaster


v2_rows = []

for pair in PAIRS:
    train_region = regions[pair["train"]]
    test_region = regions[pair["test"]]

    parts = [
        (pair["train"], "unseen time (last 15%)", train_region["validation_end"], train_region["number_of_maps"]),
        (pair["test"], "never-seen region", FIRST_TARGET, test_region["number_of_maps"]),
    ]

    for dataset, part, first_target, last_target in parts:
        region = regions[dataset]
        whole_map_model = v2_for_whole_map(v2_forecasters[pair["name"]], region, trend_factors[pair["name"]])

        evaluation, example = evaluate_region(whole_map_model, region, first_target, last_target)
        copy_error = evaluation["copy_squared_error"].sum()

        v1_row = result_table[result_table["Region"] == REGION_NAMES[dataset]].iloc[0]
        trend_row = simple_table[simple_table["Region"] == REGION_NAMES[dataset]].iloc[0]

        v2_rows.append({
            "Region": REGION_NAMES[dataset],
            "Tested on": part,
            "Skill: trend": trend_row["Skill: trend"],
            "Skill: CNN-LSTM (first version)": v1_row["CNN-LSTM: skill"],
            "Skill: CNN-LSTM version 2": round(1 - evaluation["model_squared_error"].sum() / copy_error, 3),
            "Skill: version 2 + rule": round(1 - evaluation["rule_squared_error"].sum() / copy_error, 3),
        })

        del whole_map_model

v2_table = pd.DataFrame(v2_rows)
display(v2_table)
save_table(v2_table, "08_version2_results")
