import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core import preprocess, prompt
from core.models import PLS_DA_plot

# Data paths
csv_file_paths = [
    "figure_8_9/56Features.csv",
    "figure_8_9/Blue_Red_features.csv",
]

for i, csv_file_path in enumerate(csv_file_paths, start=1):
    # Read df
    df = pd.read_csv(csv_file_path)
    y = df["Structure type"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    prompt.print_label_mapping(encoder)
    start_time = time.perf_counter()

    # Load the dataset
    X_df, X, columns = preprocess.prepare_standarlize_X_block_(csv_file_path)
    print(
        f"\nProcessing {csv_file_path} with {X.shape[1]} features ({i}/{len(csv_file_paths)})."
    )

    print("(1/1) Running PLS_DA n=2...")
    PLS_DA_plot.plot_two_component(X, y, csv_file_path)

    elapsed_time = time.perf_counter() - start_time
    print(f"===========Elapsed time: {elapsed_time:0.2f} seconds===========")
