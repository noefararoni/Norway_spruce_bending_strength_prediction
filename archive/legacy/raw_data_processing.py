# Data_preprocessing.py

import pandas as pd
from pathlib import Path


DATA_DIR = Path("Data")
WOOD_TESTING_DIR = DATA_DIR / "Wood_testing"


def load_data():
    print("---------------- Load Data ----------------")

    df_A = pd.read_csv(DATA_DIR / "Size_A.csv", encoding="latin1", sep=";")
    df_B = pd.read_csv(WOOD_TESTING_DIR / "Size_B.csv", encoding="latin1", sep=";")
    df_C = pd.read_csv(WOOD_TESTING_DIR / "Size_C.csv", encoding="latin1", sep=";")

    print("\nData shape A:", df_A.shape)
    print("Data shape B:", df_B.shape)
    print("Data shape C:", df_C.shape)

    return df_A, df_B, df_C


def merge_data(df_A, df_B, df_C):
    print("\n---------------- Merge Data ----------------")

    # 1) Find common columns (features)
    common_cols = sorted(set(df_A.columns) & set(df_B.columns) & set(df_C.columns))

    print("Number of common features:", len(common_cols))
    print("Common features:", common_cols)

    # 2) Keep only common columns and stack rows
    df_all = pd.concat(
        [df_A[common_cols], df_B[common_cols], df_C[common_cols]],
        axis=0,
        ignore_index=True
    )

    print("Combined shape:", df_all.shape)

    return df_all


def main():
    print("\n>>> START MAIN <<<")

    # Load data
    df_A, df_B, df_C = load_data()
    print("\neeeeeeeeeeeeeeeee")
    # Apply YOUR logic
    df_all = merge_data(df_A, df_B, df_C)

    print("\n>>> FINAL OUTPUT <<<")
    print(df_all.head())


if __name__ == "__main__":
    main()