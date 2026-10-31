# List everything this notebook saved on Drive
for folder in [MODEL_DIR, FIGURE_DIR, TABLE_DIR]:
    print(folder)
    for file_name in sorted(os.listdir(folder)):
        print("   ", file_name)
