# Name of the dataset on Hugging Face
REPO_ID = "mahajananhad/floodcastbench"

# Where the raw files are downloaded on Colab
RAW_DIR = "/content/floodcastbench"

# Where this notebook saves the prepared data (the Colab disk, not Google Drive)
SAVE_DIR = "/content/prepared_data"

# The copy on Google Drive that the model notebook uses (only read, never written)
DRIVE_PREPARED_DIR = "/content/drive/MyDrive/Flood_project/prepared_data"

# Time between two prepared maps: 1800 seconds = 30 minutes, the same spacing as the rain maps
STEP_SECONDS = 1800

# Water above 1 cm counts as wet, above 30 cm counts as flooded
WET_DEPTH_M = 0.01
FLOOD_DEPTH_M = 0.30

DATASETS = {
    "Australia_30m": {"region": "Australia", "folder": "High-fidelity flood forecasting/30m/Australia",
                      "resolution_m": 30, "scale": 1, "row_offset": 0, "col_offset": 0},
    "UK_30m": {"region": "UK", "folder": "High-fidelity flood forecasting/30m/UK",
               "resolution_m": 30, "scale": 1, "row_offset": 5, "col_offset": 5},
    "Australia_60m": {"region": "Australia", "folder": "High-fidelity flood forecasting/60m/Australia",
                      "resolution_m": 60, "scale": 2, "row_offset": 0, "col_offset": 0},
    "UK_60m": {"region": "UK", "folder": "High-fidelity flood forecasting/60m/UK",
               "resolution_m": 60, "scale": 2, "row_offset": 5, "col_offset": 5},
    "Pakistan_480m": {"region": "Pakistan", "folder": "Low-fidelity flood forecasting/480m/Pakistan",
                      "resolution_m": 480, "scale": 16, "row_offset": 42, "col_offset": 178},
    "Mozambique_480m": {"region": "Mozambique", "folder": "Low-fidelity flood forecasting/480m/Mozambique",
                        "resolution_m": 480, "scale": 16, "row_offset": 150, "col_offset": 50},
}

# Map projection code of each region (UTM zones listed in the dataset's Spatial_References.txt)
EPSG_CODES = {"Australia": 32756, "UK": 32630, "Pakistan": 32642, "Mozambique": 32736}

# Name and colour of each land-cover code
LAND_COVER = {
    1: ("Water", "#1f78b4"),
    2: ("Trees", "#33a02c"),
    4: ("Flooded vegetation", "#a6cee3"),
    5: ("Crops", "#fdbf6f"),
    7: ("Buildings", "#e31a1c"),
    8: ("Bare ground", "#b15928"),
    9: ("Snow / ice", "#ffffff"),
    10: ("Clouds", "#bdbdbd"),
    11: ("Rangeland", "#b2df8a"),
}
