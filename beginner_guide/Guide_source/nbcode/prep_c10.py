# Check that every needed file arrived
missing_files = []
for file_name in files_needed:
    if not os.path.exists(os.path.join(RAW_DIR, file_name)):
        missing_files.append(file_name)

print("Files missing:", len(missing_files))
if len(missing_files) > 0:
    print("Examples:", missing_files[:5])
    print("Run the download cell again")
