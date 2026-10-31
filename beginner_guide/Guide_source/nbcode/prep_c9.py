start_time = time.time()

# Try up to 5 times, because large downloads sometimes stop halfway.
# This cell can sit quietly for a minute or two before the progress bar appears.
for attempt in range(1, 6):
    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=RAW_DIR, token=HF_TOKEN,
                          allow_patterns=files_needed, max_workers=8)
        print("Download complete")
        break
    except Exception as error:
        print("Attempt", attempt, "failed:", error)
        print("Waiting 60 seconds before trying again")
        time.sleep(60)

print("Time taken:", round((time.time() - start_time) / 60, 1), "minutes")
