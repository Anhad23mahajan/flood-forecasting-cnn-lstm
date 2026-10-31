# List of every file in the dataset on Hugging Face
api = HfApi()
all_files = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)

# Find any file that did not arrive on Colab
missing_files = []
for file_name in all_files:
    if not os.path.exists(os.path.join(DATA_DIR, file_name)):
        missing_files.append(file_name)

print("Files in the dataset:", len(all_files))
print("Files missing:       ", len(missing_files))

if len(missing_files) > 0:
    print("Missing examples:", missing_files[:5])
    print("Run the download cell again to fetch them")
