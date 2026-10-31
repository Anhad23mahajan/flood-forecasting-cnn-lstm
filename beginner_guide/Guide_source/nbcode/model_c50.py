def grad_cam(grad_cam_model, model_input, class_number=None):
    # Heat map (0 to 1) over the ConvLSTM memory
    # class_number = None explains the total predicted water, otherwise the length of that class capsule

    with tf.GradientTape() as tape:
        memory, output = grad_cam_model(model_input, training=False)
        if class_number is None:
            explained = tf.reduce_sum(output)
        else:
            explained = output[0, class_number]

    # How much the explained number changes when the memory changes
    gradients = tape.gradient(explained, memory)

    # One weight per memory map: its average gradient
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))

    # Weighted sum of the memory maps, keeping the positive part
    heatmap = tf.nn.relu(tf.reduce_sum(memory * channel_weights[:, None, None, :], axis=-1))[0].numpy()

    # Scale to 0 - 1
    return heatmap / (heatmap.max() + 1e-8)


# Free memory from the whole-map tests
whole_map_forecasters = {}
gc.collect()

# Largest window size for Grad-CAM
GRAD_CAM_WINDOW = 256

# Which model (30m or 480m) belongs to each flood
model_of_region = {}
for pair in PAIRS:
    model_of_region[pair["train"]] = pair["name"]
    model_of_region[pair["test"]] = pair["name"]


def grad_cam_window(example):
    # Window centred on the busiest area of true change, kept inside the map
    rows, columns = example["true"].shape
    height = min(GRAD_CAM_WINDOW, rows)
    width = min(GRAD_CAM_WINDOW, columns)

    # Smooth the change map so the centre is a busy area, not a single square
    change = np.abs(example["true"] - example["last"])
    busy = uniform_filter(change, size=31)
    centre_row, centre_column = np.unravel_index(np.argmax(busy), busy.shape)

    top = int(np.clip(centre_row - height // 2, 0, rows - height))
    left = int(np.clip(centre_column - width // 2, 0, columns - width))
    return top, left, height, width
