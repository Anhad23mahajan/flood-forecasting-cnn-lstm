def same_array(path_a, path_b):
    # True if two saved arrays are exactly the same (compared one map at a time, to save memory)
    a = np.load(path_a, mmap_mode="r")
    b = np.load(path_b, mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.ndim < 3:
        return bool(np.array_equal(a, b, equal_nan=True))
    for step in range(a.shape[0]):
        if not np.array_equal(a[step], b[step], equal_nan=True):
            return False
    return True


compare_rows = []

for dataset in DATASETS:
    for file_name in ["depth.npy", "rain.npy", "terrain.npy", "land_cover.npy", "permanent_water.npy"]:
        ours = os.path.join(SAVE_DIR, dataset, file_name)
        on_drive = os.path.join(DRIVE_PREPARED_DIR, dataset, file_name)

        if not os.path.exists(on_drive):
            result = "not on Drive"
        elif same_array(ours, on_drive):
            result = "identical"
        else:
            result = "DIFFERENT"

        compare_rows.append({"Dataset": dataset, "File": file_name, "Same as the copy on Drive?": result})

display(pd.DataFrame(compare_rows))
