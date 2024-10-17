import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_dataframe(df):
    """
    Cleans the DataFrame by replacing 'N/S' with NA and dropping columns with any missing values.
    Returns the cleaned DataFrame along with the number of columns removed.
    """
    # Replace 'N/S', 'non', or any other specific ill-defined values with NaN
    df.replace(["N/S", "non"], [None, None], inplace=True)

    # Get the initial number of columns
    initial_cols = df.shape[1]

    # Remove columns that have any NaN values (i.e., originally had 'N/S' or 'non')
    df.dropna(axis=1, inplace=True)

    # Drop columns that contain any missing values
    df.dropna(axis=1, how="any", inplace=True)

    # Calculate the number of columns removed
    removed_col_count = initial_cols - df.shape[1]

    return df, removed_col_count


def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def drop_columns(df):
    """
    Drops columns from a DataFrame if they exist.
    """
    cols_to_drop = [
        "Formula",
        "formula",
        "Structure",
        "Structure type",
        "structure_type",
        "A",
        "B",
        "Entry",
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


def prepare_standarlize_X_block_(csv_file_path):
    df = pd.read_csv(csv_file_path)
    X_df = drop_columns(df)
    df, removed_col_count = clean_dataframe(df)
    print("Removed", removed_col_count, "columns with N/S or non values.")

    # XBBoost can't read [pm]
    X_df.columns = X_df.columns.str.replace(r"\[pm\]", "", regex=True)

    X = standardize_data(X_df)
    columns = df.columns.tolist()
    return X_df, X, columns
