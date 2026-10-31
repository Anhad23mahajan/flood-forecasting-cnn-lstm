api = HfApi()
all_files = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)

files_needed = []

for file_name in all_files:
    # Rain, ground height, land cover and map-position files
    if file_name.startswith("Relevant data/Rainfall/"):
        files_needed.append(file_name)
    elif file_name.startswith("Relevant data/DEM/"):
        files_needed.append(file_name)
    elif file_name.startswith("Relevant data/Land use and land cover/"):
        files_needed.append(file_name)
    elif file_name.startswith("Relevant data/Georeferenced files/"):
        files_needed.append(file_name)

    # Depth maps: keep only the ones exactly on a 30-minute mark
    elif file_name.startswith("High-fidelity") or file_name.startswith("Low-fidelity"):
        seconds = int(os.path.basename(file_name).replace(".tif", ""))
        if seconds % STEP_SECONDS == 0:
            files_needed.append(file_name)

print("Files in the dataset:", len(all_files))
print("Files we need:       ", len(files_needed))
