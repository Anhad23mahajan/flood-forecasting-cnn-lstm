# Connect Google Drive
drive.mount("/content/drive")

# Folders we read from and save to
PROJECT_DIR = "/content/drive/MyDrive/Flood_project"
PREPARED_DIR = PROJECT_DIR + "/prepared_data"
MODEL_DIR = PROJECT_DIR + "/models"
FIGURE_DIR = PROJECT_DIR + "/model_outputs/figures"
TABLE_DIR = PROJECT_DIR + "/model_outputs/tables"

# Make the output folders if they are missing
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
