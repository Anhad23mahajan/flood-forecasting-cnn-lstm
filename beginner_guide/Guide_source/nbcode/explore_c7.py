# Check that Colab has enough free disk space
total, used, free = shutil.disk_usage("/content")

print("Free disk space:", round(free / 1e9, 1), "GB")
print("Space needed:    about 30 GB")
