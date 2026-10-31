# Name of the dataset on Hugging Face
REPO_ID = "mahajananhad/floodcastbench"

# Folder on Colab where the dataset is saved (it is wiped when the session ends)
DATA_DIR = "/content/floodcastbench"

# The four flood regions
REGIONS = ["Australia", "UK", "Pakistan", "Mozambique"]

# Folder holding the water-depth maps of each region at each map scale
DEPTH_FOLDERS = {
    "Australia 30m": "High-fidelity flood forecasting/30m/Australia",
    "UK 30m": "High-fidelity flood forecasting/30m/UK",
    "Australia 60m": "High-fidelity flood forecasting/60m/Australia",
    "UK 60m": "High-fidelity flood forecasting/60m/UK",
    "Pakistan 480m": "Low-fidelity flood forecasting/480m/Pakistan",
    "Mozambique 480m": "Low-fidelity flood forecasting/480m/Mozambique",
}

# Which region each dataset belongs to
REGION_OF = {
    "Australia 30m": "Australia",
    "UK 30m": "UK",
    "Australia 60m": "Australia",
    "UK 60m": "UK",
    "Pakistan 480m": "Pakistan",
    "Mozambique 480m": "Mozambique",
}

# Water depth at the moment each simulation starts
START_WATER_FILES = {
    "Australia": "Relevant data/Initial conditions/High-fidelity flood forecasting/Australia_30m.tif",
    "UK": "Relevant data/Initial conditions/High-fidelity flood forecasting/UK_30m.tif",
    "Pakistan": "Relevant data/Initial conditions/Low-fidelity flood forecasting/Pakistan_480m.tif",
    "Mozambique": "Relevant data/Initial conditions/Low-fidelity flood forecasting/Mozambique_480m.tif",
}

# Time between two depth maps: 300 seconds = 5 minutes
DEPTH_STEP_SECONDS = 300

# Time between two rain maps: 1800 seconds = 30 minutes
RAIN_STEP_SECONDS = 1800

# One rain map covers the same time as 6 depth maps
DEPTH_MAPS_PER_RAIN_MAP = 6

# Water above 1 cm counts as wet
WET_DEPTH_M = 0.01
