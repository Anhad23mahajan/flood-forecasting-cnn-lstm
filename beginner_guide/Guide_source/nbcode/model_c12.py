# The first map we can forecast: we need 4 earlier maps and the rain before them
FIRST_TARGET = INPUT_STEPS + 1

# Table of which hours are used for training, checking and testing
split_rows = []

for pair in PAIRS:
    train_region = regions[pair["train"]]

    train_hours = str(FIRST_TARGET * STEP_HOURS) + " - " + str(train_region["train_end"] * STEP_HOURS)
    check_hours = str(train_region["train_end"] * STEP_HOURS) + " - " + str(train_region["validation_end"] * STEP_HOURS)
    test_hours = str(train_region["validation_end"] * STEP_HOURS) + " - " + str(train_region["number_of_maps"] * STEP_HOURS)

    split_rows.append({"Model": pair["name"], "Region": REGION_NAMES[pair["train"]],
                       "Train (hours)": train_hours, "Validation (hours)": check_hours, "Test (hours)": test_hours})

    split_rows.append({"Model": pair["name"], "Region": REGION_NAMES[pair["test"]],
                       "Train (hours)": "-", "Validation (hours)": "-", "Test (hours)": "whole flood (never seen)"})

split_table = pd.DataFrame(split_rows)
display(split_table)
save_table(split_table, "01_data_split")
