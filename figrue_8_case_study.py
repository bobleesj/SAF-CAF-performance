import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from core import folder, preprocess, prompt

# Use the SAF's features as the "y" vector
df_SAF = pd.read_csv("figure_8_9/56Features.csv")
y = df_SAF["Structure type"]

script_dir_path = os.getcwd()
output_dir_path = os.path.join(script_dir_path, "outputs")

# outputs/SAF_CAF/binary_features.csv
SAF_CAF_feature_path = os.path.join(
    output_dir_path, "SAF_CAF", "binary_features.csv"
)
X_df, X, columns = preprocess.prepare_X_block(SAF_CAF_feature_path)


def plot_two_component(X, y, feature_file_path):
    # Convert string labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    prompt.print_label_mapping(encoder)

    # Load the dataset into PLS
    pls = PLSRegression(n_components=2)
    # fit_transform returns a tuple (X_scores, Y_scores)
    X_pls = pls.fit_transform(X, y_encoded)[0]

    # Calculate the variance explained by each component for X
    total_variance_X = np.var(X, axis=0).sum()

    # calculates the variance of the scores for the
    explained_variance_X = [
        np.var(X_pls[:, i]) / total_variance_X for i in range(pls.n_components)
    ]

    # Define the file path for saving the plot
    plot_path = folder.create_folder_get_output_path(
        "PLS_DA_plot",
        feature_file_path,
        "proxy_learning_n=2",
        ".png",
    )
    # Same color to TlI, FeB-b, FeAs, NiAs, CoSn
    unique = np.unique(y_encoded)
    colors = [plt.cm.jet(float(i) / max(unique)) for i in unique]

    # Define the same color scheme for these specific labels
    same_color_labels = ["TlI", "FeB-b", "FeAs", "NiAs", "CoSn"]

    with plt.style.context("ggplot"):
        for i, label in enumerate(unique):
            # Decode the label to its original form
            actual_label = encoder.inverse_transform([label])[0]

            # Extract points for the current label
            xi = [
                X_pls[j, 0]
                for j in range(len(X_pls[:, 0]))
                if y_encoded[j] == label
            ]
            yi = [
                X_pls[j, 1]
                for j in range(len(X_pls[:, 1]))
                if y_encoded[j] == label
            ]

            # Assign the same color for specific labels
            if actual_label in same_color_labels:
                color = "red"
            else:
                color = colors[i]

            # Plot the points
            plt.scatter(
                xi,
                yi,
                color=color,
                s=100,
                edgecolors="k",
                label=actual_label,  # Use the decoded label in the legend
            )

        plt.xlabel(f"LV 1 ({(explained_variance_X[0] * 100):.2f} %)")
        plt.ylabel(f"LV 2 ({(explained_variance_X[1] * 100):.2f} %)")
        plt.legend(loc="lower left")
        plt.savefig(plot_path, dpi=300)  # Save the plot as a PNG file
        plt.close()
        plt.show()


plot_two_component(X, y, SAF_CAF_feature_path)

# TlI/FeB and FeAs/NiAs/CoSn.
