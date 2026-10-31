# Load all four floods (takes a few minutes)
regions = {}

for dataset in REGION_NAMES:
    regions[dataset] = load_region(dataset)
    print(dataset, regions[dataset]["depth"].shape)
