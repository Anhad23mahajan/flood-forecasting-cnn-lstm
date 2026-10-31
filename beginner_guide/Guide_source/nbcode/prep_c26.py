# Every saved file and its size
file_rows = []

for dataset in DATASETS:
    dataset_dir = os.path.join(SAVE_DIR, dataset)
    for file_name in sorted(os.listdir(dataset_dir)):
        size_mb = os.path.getsize(os.path.join(dataset_dir, file_name)) / 1e6
        file_rows.append({"Dataset": dataset, "File": file_name, "Size (MB)": round(size_mb, 1)})

file_table = pd.DataFrame(file_rows)
display(file_table)
print("Total saved:", round(file_table["Size (MB)"].sum() / 1000, 2), "GB")
