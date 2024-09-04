import os

import pandas as pd
from CBFV import composition

# Prepare a .csv file that has a column either "Formula" or "formula"
csv_file_path = "data/features.csv"

# Load the data
df = pd.read_csv(csv_file_path)

# Load available features in CBFV
features = ["oliynyk", "jarvis", "magpie", "mat2vec"]

# Make a folder
output_folder_path = "outputs/CBFV"
os.makedirs(output_folder_path, exist_ok=True)

# Loop through each feature
for feature in features:
    print(f"Generating features for {feature}...")
    # CBFV requires a column with a target init all rows in the target column with 0
    df["target"] = 0
    df.rename(columns={"Formula": "formula"}, inplace=True)

    # CBFV requires the first adn second column to be the formula and target
    df = df[["formula", "target"]]
    X, y, formulas, skipped = composition.generate_features(
        df,
        elem_prop=feature,
        drop_duplicates=False,
        extend_features=True,
        sum_feat=True,
    )

    # Combine the first column with the formula
    X.insert(0, "Formula", formulas)
    X.to_csv(f"{output_folder_path}/{feature}.csv", index=False)
    print("These formulas are skipped: ", skipped)
    print(f"{feature} is saved in {output_folder_path}/{feature}.csv")

print("Done!")
