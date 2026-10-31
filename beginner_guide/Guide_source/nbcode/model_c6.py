# Two models: each one learns on one flood and is tested on another flood it has never seen
PAIRS = [
    {"name": "30m", "train": "Australia_30m", "test": "UK_30m"},
    {"name": "480m", "train": "Pakistan_480m", "test": "Mozambique_480m"},
]

# Nice names for the four floods
REGION_NAMES = {
    "Australia_30m": "Australia (30 m)",
    "UK_30m": "UK (30 m)",
    "Pakistan_480m": "Pakistan (480 m)",
    "Mozambique_480m": "Mozambique (480 m)",
}

# Time between two maps, in hours
STEP_HOURS = 0.5

# Water above 1 cm counts as wet, above 30 cm counts as flooded
WET_DEPTH_M = 0.01
FLOOD_DEPTH_M = 0.30

# Severity classes: share of normally dry land under more than 30 cm of water
SEVERITY_LIMITS = [0.05, 0.15, 0.30]
SEVERITY_NAMES = ["None", "Minor", "Moderate", "Severe"]

# First 70% of the flood for training, next 15% for checking, last 15% for testing
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15

# The model sees 4 maps and each map has 3 layers (depth, rain, ground height)
INPUT_STEPS = 4
CHANNELS = 3

# Size and number of the small square pieces (tiles) we cut from the maps
TILE_SIZE = 64
TRAIN_TILES = 4000
VALIDATION_TILES = 600
TEST_TILES = 1000

# A tile "changes" if its water moves by more than 1 cm. 1.0 means we keep every tile.
CHANGE_LIMIT_M = 0.01
CALM_TILE_SHARE = 1.0

# Rain is divided by 10 so it is on a similar scale to the other inputs
RAIN_SCALE = 10.0

# Ground height is compared with the average of the 15 x 15 squares around it
TERRAIN_WINDOW = 15

# Training settings
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# False = load the saved models from Drive. True = train them again.
TRAIN_NEW = False
TRAIN_V2 = False

# Fix the random numbers so every run gives the same result
keras.utils.set_random_seed(42)

# Blue colours for water, grey for dry squares
water_colours = ListedColormap(plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
water_colours.set_bad("lightgrey")
