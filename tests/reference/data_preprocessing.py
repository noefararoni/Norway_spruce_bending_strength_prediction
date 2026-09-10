# data_preprocessing.py

import pandas as pd
from pathlib import Path
from config import SELECTED_FEATURES


DATA_DIR = Path("Data")
WOOD_TESTING_DIR = DATA_DIR / "Wood_testing"


def load_data():
    print("---------------- Load Data ----------------", flush=True)

    df_A = pd.read_csv(DATA_DIR / "Size_A.csv", encoding="latin1", sep=";")
    df_B = pd.read_csv(DATA_DIR / "Size_B.csv", encoding="latin1", sep=";")
    df_C = pd.read_csv(DATA_DIR / "Size_C.csv", encoding="latin1", sep=";")

    print("Data shape A:", df_A.shape, flush=True)
    print("Data shape B:", df_B.shape, flush=True)
    print("Data shape C:", df_C.shape, flush=True)

    return df_A, df_B, df_C


def merge_data(df_A, df_B, df_C):
    print("\n---------------- Merge Data ----------------", flush=True)

    common_cols = sorted(set(df_A.columns) & set(df_B.columns) & set(df_C.columns))

    print("Number of common features:", len(common_cols), flush=True)

    df_A_common = df_A[common_cols].copy()
    df_B_common = df_B[common_cols].copy()
    df_C_common = df_C[common_cols].copy()

    df_A_common["source_file"] = "Size_A"
    df_B_common["source_file"] = "Size_B"
    df_C_common["source_file"] = "Size_C"

    df_A_common["source_row"] = df_A_common.index
    df_B_common["source_row"] = df_B_common.index
    df_C_common["source_row"] = df_C_common.index

    df_all = pd.concat(
        [df_A_common, df_B_common, df_C_common],
        axis=0,
        ignore_index=True
    )

    expected_rows = len(df_A) + len(df_B) + len(df_C)
    actual_rows = len(df_all)

    print("Expected rows:", expected_rows, flush=True)
    print("Actual merged rows:", actual_rows, flush=True)

    if expected_rows != actual_rows:
        raise ValueError("Row mismatch after merging!")

    print("\nRows per source:", flush=True)
    print(df_all["source_file"].value_counts(), flush=True)

    print("Combined shape:", df_all.shape, flush=True)

    return df_all


def validate_merge(df_A, df_B, df_C, df_all):
    print("\n---------------- Validate Merge ----------------", flush=True)

    expected_rows = len(df_A) + len(df_B) + len(df_C)
    actual_rows = len(df_all)

    print("Expected rows:", expected_rows, flush=True)
    print("Actual rows:", actual_rows, flush=True)

    if expected_rows != actual_rows:
        raise ValueError("Mismatch: merged dataframe row count is incorrect.")

    expected_counts = {
        "Size_A": len(df_A),
        "Size_B": len(df_B),
        "Size_C": len(df_C),
    }

    actual_counts = df_all["source_file"].value_counts().to_dict()

    print("\nExpected source counts:", flush=True)
    print(expected_counts, flush=True)

    print("\nActual source counts:", flush=True)
    print(actual_counts, flush=True)

    for source, expected_count in expected_counts.items():
        actual_count = actual_counts.get(source, 0)

        if actual_count != expected_count:
            raise ValueError(
                f"Mismatch for {source}: expected {expected_count}, got {actual_count}"
            )

    print("\nMerge validation passed.", flush=True)


def select_features(df_all):
    print("\n---------------- Select Features ----------------", flush=True)

    tracking_cols = ["source_file", "source_row"]

    missing_features = [
        col for col in SELECTED_FEATURES
        if col not in df_all.columns
    ]

    if missing_features:
        raise ValueError(f"Missing selected features: {missing_features}")

    cols_to_keep = SELECTED_FEATURES + tracking_cols

    df_selected = df_all[cols_to_keep].copy()

    print("Selected features:", SELECTED_FEATURES, flush=True)
    print("Tracking columns:", tracking_cols, flush=True)
    print("Selected dataframe shape:", df_selected.shape, flush=True)

    return df_selected


def clean_data(df):
    print("\n---------------- Cleaning Data ----------------", flush=True)

    tracking_cols = ["source_file", "source_row"]
    existing_tracking_cols = [
        col for col in tracking_cols
        if col in df.columns
    ]

    feature_cols = [
        col for col in df.columns
        if col not in existing_tracking_cols
    ]

    df_clean = df.copy()

    df_clean[feature_cols] = df_clean[feature_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print("Before dropna:", df_clean.shape, flush=True)

    rows_before = len(df_clean)

    df_clean = df_clean.dropna(subset=feature_cols)

    rows_after = len(df_clean)

    print("After dropna:", df_clean.shape, flush=True)
    print("Rows removed during cleaning:", rows_before - rows_after, flush=True)

    if "source_file" in df_clean.columns:
        print("\nRemaining rows per source after cleaning:", flush=True)
        print(df_clean["source_file"].value_counts(), flush=True)

    return df_clean


def remove_outliers_by_indices(df, indices_to_remove):
    print("\n---------------- Remove Outliers ----------------", flush=True)

    if indices_to_remove is None or len(indices_to_remove) == 0:
        print("No outlier indices provided.", flush=True)
        return df

    print("Data shape before removing outliers:", df.shape, flush=True)
    print("Indices to remove:", indices_to_remove, flush=True)

    df_removed = df.drop(index=indices_to_remove, errors="ignore")

    print("Data shape after removing outliers:", df_removed.shape, flush=True)
    print("Removed rows:", len(df) - len(df_removed), flush=True)

    if "source_file" in df_removed.columns:
        print("\nRows per source after outlier removal:", flush=True)
        print(df_removed["source_file"].value_counts(), flush=True)

    return df_removed


def split_X_y(df):
    print("\n---------------- Split X / y ----------------", flush=True)

    target_col = "Bending strength"
    tracking_cols = ["source_file", "source_row"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    columns_to_drop = [target_col] + [
        col for col in tracking_cols
        if col in df.columns
    ]

    X = df.drop(columns=columns_to_drop)
    y = df[target_col].values

    print("X shape:", X.shape, flush=True)
    print("y shape:", y.shape, flush=True)

    return X, y



def save_final_dataframe(df, filename="final_clean_merged_dataframe.csv"):
    output_path = DATA_DIR/filename

    df.to_csv(output_path, index=False)

    print(f"\nSaved final dataframe to: {output_path}", flush=True)

def main():
    df_A, df_B, df_C = load_data()

    df_all = merge_data(df_A, df_B, df_C)

    validate_merge(df_A, df_B, df_C, df_all)

    df_selected = select_features(df_all)

    df_clean = clean_data(df_selected)

    indices_to_remove = [12, 35, 41]  # replace with your detected indices

    df_clean = remove_outliers_by_indices(df_clean, indices_to_remove)

    save_final_dataframe(df_clean)

    X, y = split_X_y(df_clean)

    print("\n>>> FINAL OUTPUT <<<", flush=True)
    print(X.head(), flush=True)
    print("y sample:", y[:5], flush=True)


if __name__ == "__main__":
    main()