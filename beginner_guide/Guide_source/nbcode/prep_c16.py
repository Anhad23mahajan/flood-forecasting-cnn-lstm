# Average depth of each square over the whole flood, used in the alignment check
depth_averages = {}

for dataset in DATASETS:
    files = get_depth_files(dataset)
    rows, columns = DEPTH_SHAPES[dataset]

    # Check that no 30-minute map is missing
    seconds = [get_seconds(f) for f in files]
    if seconds != list(range(0, seconds[-1] + 1, STEP_SECONDS)):
        print(dataset, "- WARNING: some 30-minute maps are missing")

    # One map per 30 minutes
    depth = np.zeros((len(files), rows, columns), dtype="float32")
    for step, file_path in enumerate(files):
        with rasterio.open(file_path) as source:
            depth[step] = source.read(1)

    # Save to this dataset's folder
    dataset_dir = os.path.join(SAVE_DIR, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    np.save(os.path.join(dataset_dir, "depth.npy"), depth)

    depth_averages[dataset] = depth.mean(axis=0)
    print(dataset, "- saved", len(files), "maps of", rows, "x", columns)

    # Free the memory before the next dataset
    del depth
