start_time = time.time()

# Try up to 5 times, because large downloads sometimes stop halfway
for attempt in range(1, 6):
    try:
        # Files that are already downloaded are skipped, so a retry carries on where it stopped
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DATA_DIR,
                          token=HF_TOKEN, max_workers=8)
        print("Download complete")
        break
    except Exception as error:
        print("Attempt", attempt, "failed:", error)
        print("Waiting 60 seconds before trying again")
        time.sleep(60)

print("Time taken:", round((time.time() - start_time) / 60, 1), "minutes")
