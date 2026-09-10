# normal_LR_with_outlier_removal_and_generalization.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from config import LEARNING_RATE, N_ITERATIONS
from data_preprocessing import (
    split_X_y,
    clean_data,
    select_features,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("Data")

TRAINING_DATASET_PATH = (
    DATA_DIR / "final_clean_merged_dataframe.csv"
)

GENERALIZATION_DATASET_PATH = (
    DATA_DIR / "generalisation_validation.csv"
)

N_OUTLIERS_TO_REMOVE = 5

TARGET_COLUMN = "Bending strength"

TRACKING_COLUMNS = [
    "source_file",
    "source_row",
    "Specimen ID",
]


# ============================================================
# Output paths
# ============================================================

OUTPUT_ORIGINAL_LOOCV = (
    DATA_DIR / "loocv_original_errors.csv"
)

OUTPUT_CLEANED_LOOCV = (
    DATA_DIR / "loocv_after_outlier_removal.csv"
)

OUTPUT_REMOVED_OUTLIERS = (
    DATA_DIR / "removed_outliers.csv"
)

OUTPUT_ORIGINAL_GENERALIZATION = (
    DATA_DIR / "generalization_original_model.csv"
)

OUTPUT_CLEANED_GENERALIZATION = (
    DATA_DIR / "generalization_cleaned_model.csv"
)

OUTPUT_PERFORMANCE_SUMMARY = (
    DATA_DIR / "model_performance_summary.csv"
)


# ============================================================
# Linear Regression from Scratch
# ============================================================

class LinearRegressionScratch:
    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=5000,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if n_iterations <= 0:
            raise ValueError(
                "n_iterations must be greater than zero."
            )

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = (
                np.dot(X, self.weights)
                + self.bias
            )

            error = y_pred - y

            dw = (
                2.0 / n_samples
            ) * np.dot(X.T, error)

            db = (
                2.0 / n_samples
            ) * np.sum(error)

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError(
                "The model must be trained before prediction."
            )

        X = np.asarray(X, dtype=float)

        return (
            np.dot(X, self.weights)
            + self.bias
        )


# ============================================================
# CSV utilities
# ============================================================

def read_dataset(path):
    """
    Reads semicolon- or tab-separated datasets that use
    comma decimal notation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    attempts = [
        {
            "sep": ";",
            "decimal": ",",
        },
        {
            "sep": "\t",
            "decimal": ",",
        },
    ]

    last_error = None

    for options in attempts:
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                **options,
            )

            if len(df.columns) > 1:
                df.columns = [
                    str(column).strip()
                    for column in df.columns
                ]

                return df

        except Exception as error:
            last_error = error

    raise ValueError(
        f"Could not read dataset correctly: {path}"
    ) from last_error


def convert_columns_to_numeric(
    df,
    columns,
    dataset_name,
):
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' "
                f"in {dataset_name}."
            )

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    invalid_counts = (
        df[columns]
        .isna()
        .sum()
    )

    invalid_counts = invalid_counts[
        invalid_counts > 0
    ]

    if not invalid_counts.empty:
        raise ValueError(
            f"Invalid numeric values in {dataset_name}:\n"
            f"{invalid_counts}"
        )

    return df


# ============================================================
# Load training dataset
# ============================================================

def load_training_dataset(
    path=TRAINING_DATASET_PATH,
):
    print(
        "\n---------------- Load Training Dataset ----------------"
    )

    df_all = read_dataset(path)

    print("Dataset path:", path)
    print("Raw shape:", df_all.shape)
    print("Raw columns:", list(df_all.columns))

    df_selected = select_features(
        df_all
    )

    df_clean = clean_data(
        df_selected
    )

    X, y = split_X_y(
        df_clean
    )

    X = X.copy()
    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features:", list(X.columns))

    return (
        df_clean.reset_index(drop=True),
        X.reset_index(drop=True),
        y,
    )


# ============================================================
# Load independent validation dataset
# ============================================================

def load_generalization_dataset(
    training_features,
    path=GENERALIZATION_DATASET_PATH,
):
    print(
        "\n---------------- Load Generalization Dataset ----------------"
    )

    df = read_dataset(path)

    required_columns = (
        list(training_features)
        + [TARGET_COLUMN]
    )

    df = convert_columns_to_numeric(
        df=df,
        columns=required_columns,
        dataset_name="generalization dataset",
    )

    initial_rows = len(df)

    df = (
        df
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    print("Dataset path:", path)
    print("Initial rows:", initial_rows)
    print("Remaining rows:", len(df))
    print("Columns:", list(df.columns))

    X_generalization = (
        df[list(training_features)]
        .copy()
    )

    y_generalization = (
        df[TARGET_COLUMN]
        .to_numpy(dtype=float)
    )

    print(
        "Generalization X shape:",
        X_generalization.shape,
    )
    print(
        "Generalization y shape:",
        y_generalization.shape,
    )

    return (
        df,
        X_generalization,
        y_generalization,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    dataset_name,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    residuals = (
        y_true - y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    mean_error = np.mean(
        residuals
    )

    mape = np.mean(
        np.abs(
            residuals
            / np.maximum(
                np.abs(y_true),
                1e-12,
            )
        )
    ) * 100.0

    print(
        f"\n---------------- {dataset_name} ----------------"
    )
    print(
        f"Number of specimens: {len(y_true)}"
    )
    print(
        f"MSE:        {mse:.4f} MPa²"
    )
    print(
        f"RMSE:       {rmse:.4f} MPa"
    )
    print(
        f"MAE:        {mae:.4f} MPa"
    )
    print(
        f"R²:         {r2:.4f}"
    )
    print(
        f"MAPE:       {mape:.4f}%"
    )
    print(
        f"Mean error: {mean_error:.4f} MPa"
    )

    return {
        "Dataset": dataset_name,
        "Number of specimens": len(y_true),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE (%)": mape,
        "Mean error": mean_error,
    }


# ============================================================
# LOOCV
# ============================================================

def leave_one_out_linear_regression(
    X,
    y,
    experiment_name,
    output_path=None,
):
    print(
        f"\n---------------- {experiment_name} ----------------"
    )
    print(
        "Using Leave-One-Out Cross-Validation"
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    squared_errors = []
    absolute_errors = []

    error_records = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        loo.split(X),
        start=1,
    ):
        X_train = (
            X.iloc[train_index]
            .values
        )

        X_test = (
            X.iloc[test_index]
            .values
        )

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        scaler = StandardScaler()

        X_train_scaled = (
            scaler
            .fit_transform(X_train)
        )

        X_test_scaled = (
            scaler
            .transform(X_test)
        )

        model = LinearRegressionScratch(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        y_pred = model.predict(
            X_test_scaled
        )

        error = float(
            y_test[0]
            - y_pred[0]
        )

        absolute_error = abs(
            error
        )

        squared_error = (
            error ** 2
        )

        y_true_all.append(
            float(y_test[0])
        )

        y_pred_all.append(
            float(y_pred[0])
        )

        absolute_errors.append(
            absolute_error
        )

        squared_errors.append(
            squared_error
        )

        row_position = int(
            test_index[0]
        )

        error_records.append(
            {
                "fold": fold,
                "original_index": int(
                    X.index[row_position]
                ),
                "true_value": float(
                    y_test[0]
                ),
                "predicted_value": float(
                    y_pred[0]
                ),
                "error": error,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                **X.iloc[
                    row_position
                ].to_dict(),
            }
        )

        print(
            f"Fold {fold}: "
            f"True={y_test[0]:.4f}, "
            f"Predicted={y_pred[0]:.4f}, "
            f"Absolute error={absolute_error:.4f}",
            flush=True,
        )

    final_mse = mean_squared_error(
        y_true_all,
        y_pred_all,
    )

    final_rmse = np.sqrt(
        final_mse
    )

    final_mae = mean_absolute_error(
        y_true_all,
        y_pred_all,
    )

    final_r2 = r2_score(
        y_true_all,
        y_pred_all,
    )

    error_df = pd.DataFrame(
        error_records
    )

    error_df = (
        error_df
        .sort_values(
            by="absolute_error",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n---------------- Final LOOCV Results ----------------"
    )
    print(
        f"Average squared error: "
        f"{np.mean(squared_errors):.4f} "
        f"± {np.std(squared_errors):.4f}"
    )
    print(
        f"Average absolute error: "
        f"{np.mean(absolute_errors):.4f} "
        f"± {np.std(absolute_errors):.4f}"
    )
    print(
        f"Final MSE:  {final_mse:.4f}"
    )
    print(
        f"Final RMSE: {final_rmse:.4f}"
    )
    print(
        f"Final MAE:  {final_mae:.4f}"
    )
    print(
        f"Final R²:   {final_r2:.4f}"
    )

    print(
        "\nTop 10 highest-error specimens:"
    )

    print(
        error_df
        .head(10)
        .to_string(index=False)
    )

    if output_path is not None:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        error_df.to_csv(
            output_path,
            sep=";",
            decimal=",",
            index=False,
        )

        print(
            "\nSaved LOOCV errors to:",
            output_path,
        )

    results = {
        "Dataset": experiment_name,
        "Number of specimens": len(y),
        "MSE": final_mse,
        "RMSE": final_rmse,
        "MAE": final_mae,
        "R2": final_r2,
        "MAPE (%)": np.nan,
        "Mean error": np.mean(
            np.asarray(y_true_all)
            - np.asarray(y_pred_all)
        ),
    }

    return results, error_df


# ============================================================
# Remove top high-error specimens
# ============================================================

def remove_top_outliers_by_error(
    df_training,
    X,
    y,
    error_df,
    n_outliers,
):
    if n_outliers < 0:
        raise ValueError(
            "n_outliers cannot be negative."
        )

    if n_outliers >= len(X):
        raise ValueError(
            "n_outliers must be smaller than "
            "the number of training samples."
        )

    if n_outliers == 0:
        print(
            "\nNo outliers were removed."
        )

        return (
            df_training.copy(),
            X.copy(),
            np.asarray(y).copy(),
            error_df.head(0).copy(),
        )

    outlier_information = (
        error_df
        .head(n_outliers)
        .copy()
    )

    outlier_indices = (
        outlier_information[
            "original_index"
        ]
        .astype(int)
        .tolist()
    )

    print(
        "\n---------------- Removing High-Error Samples ----------------"
    )
    print(
        f"Number removed: {n_outliers}"
    )
    print(
        "Removed indices:",
        outlier_indices,
    )

    # X and df_training use corresponding row indices.
    X_cleaned = (
        X
        .drop(
            index=outlier_indices
        )
        .copy()
    )

    df_cleaned = (
        df_training
        .drop(
            index=outlier_indices
        )
        .copy()
    )

    y_series = pd.Series(
        np.asarray(y),
        index=X.index,
    )

    y_cleaned = (
        y_series
        .drop(
            index=outlier_indices
        )
        .to_numpy(dtype=float)
    )

    # Reset indices only after removing the selected samples.
    X_cleaned = (
        X_cleaned
        .reset_index(drop=True)
    )

    df_cleaned = (
        df_cleaned
        .reset_index(drop=True)
    )

    print(
        "Original training size:",
        len(X),
    )
    print(
        "Cleaned training size:",
        len(X_cleaned),
    )

    OUTPUT_REMOVED_OUTLIERS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outlier_information.to_csv(
        OUTPUT_REMOVED_OUTLIERS,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "Saved removed specimens to:",
        OUTPUT_REMOVED_OUTLIERS,
    )

    return (
        df_cleaned,
        X_cleaned,
        y_cleaned,
        outlier_information,
    )


# ============================================================
# Train a final model
# ============================================================

def train_final_model(
    X_train,
    y_train,
):
    scaler = StandardScaler()

    X_train_scaled = (
        scaler
        .fit_transform(X_train)
    )

    model = LinearRegressionScratch(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return (
        model,
        scaler,
        X_train_scaled,
    )


# ============================================================
# Evaluate final model
# ============================================================

def evaluate_final_model(
    model,
    scaler,
    df_evaluation,
    X_evaluation,
    y_evaluation,
    dataset_name,
    output_path,
):
    # Use the training scaler.
    # Never fit a scaler on the generalization dataset.
    X_scaled = (
        scaler
        .transform(X_evaluation)
    )

    y_pred = model.predict(
        X_scaled
    )

    metrics = calculate_metrics(
        y_true=y_evaluation,
        y_pred=y_pred,
        dataset_name=dataset_name,
    )

    predictions = (
        df_evaluation
        .copy()
        .reset_index(drop=True)
    )

    residuals = (
        np.asarray(y_evaluation)
        - np.asarray(y_pred)
    )

    predictions[
        "Measured bending strength"
    ] = y_evaluation

    predictions[
        "Predicted bending strength"
    ] = y_pred

    predictions[
        "Error"
    ] = residuals

    predictions[
        "Absolute error"
    ] = np.abs(
        residuals
    )

    predictions[
        "Squared error"
    ] = (
        residuals ** 2
    )

    predictions[
        "Absolute percentage error (%)"
    ] = (
        np.abs(residuals)
        / np.maximum(
            np.abs(y_evaluation),
            1e-12,
        )
    ) * 100.0

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "Saved predictions to:",
        output_path,
    )

    return (
        metrics,
        predictions,
    )


# ============================================================
# Main
# ============================================================

def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load original 99-sample dataset
    # --------------------------------------------------------

    (
        df_training,
        X,
        y,
    ) = load_training_dataset()

    # --------------------------------------------------------
    # Load independent 25-sample validation dataset
    # --------------------------------------------------------

    (
        df_generalization,
        X_generalization,
        y_generalization,
    ) = load_generalization_dataset(
        training_features=list(
            X.columns
        )
    )

    # --------------------------------------------------------
    # 1. LOOCV on original dataset
    # --------------------------------------------------------

    print(
        "\n================================================"
    )
    print(
        "LOOCV: ORIGINAL TRAINING DATASET"
    )
    print(
        "================================================"
    )

    original_loocv_metrics, error_df = (
        leave_one_out_linear_regression(
            X=X,
            y=y,
            experiment_name=(
                "LOOCV - Original Training Dataset"
            ),
            output_path=OUTPUT_ORIGINAL_LOOCV,
        )
    )

    # --------------------------------------------------------
    # 2. Remove selected high-error samples
    # --------------------------------------------------------

    (
        df_cleaned,
        X_cleaned,
        y_cleaned,
        removed_outliers,
    ) = remove_top_outliers_by_error(
        df_training=df_training,
        X=X,
        y=y,
        error_df=error_df,
        n_outliers=N_OUTLIERS_TO_REMOVE,
    )

    # --------------------------------------------------------
    # 3. LOOCV after outlier removal
    # --------------------------------------------------------

    print(
        "\n================================================"
    )
    print(
        f"LOOCV: AFTER REMOVING "
        f"{N_OUTLIERS_TO_REMOVE} SAMPLES"
    )
    print(
        "================================================"
    )

    cleaned_loocv_metrics, cleaned_error_df = (
        leave_one_out_linear_regression(
            X=X_cleaned,
            y=y_cleaned,
            experiment_name=(
                f"LOOCV - Training Dataset Without "
                f"{N_OUTLIERS_TO_REMOVE} High-Error Samples"
            ),
            output_path=OUTPUT_CLEANED_LOOCV,
        )
    )

    # --------------------------------------------------------
    # 4. Train final model using all 99 samples
    # --------------------------------------------------------

    print(
        "\n================================================"
    )
    print(
        "FINAL MODEL: ALL ORIGINAL TRAINING SAMPLES"
    )
    print(
        "================================================"
    )

    (
        original_model,
        original_scaler,
        original_X_scaled,
    ) = train_final_model(
        X_train=X,
        y_train=y,
    )

    # External validation of original model
    (
        original_generalization_metrics,
        original_generalization_predictions,
    ) = evaluate_final_model(
        model=original_model,
        scaler=original_scaler,
        df_evaluation=df_generalization,
        X_evaluation=X_generalization,
        y_evaluation=y_generalization,
        dataset_name=(
            "External Generalization - Original Model"
        ),
        output_path=OUTPUT_ORIGINAL_GENERALIZATION,
    )

    # --------------------------------------------------------
    # 5. Train final model after removing outliers
    # --------------------------------------------------------

    print(
        "\n================================================"
    )
    print(
        f"FINAL MODEL: AFTER REMOVING "
        f"{N_OUTLIERS_TO_REMOVE} SAMPLES"
    )
    print(
        "================================================"
    )

    (
        cleaned_model,
        cleaned_scaler,
        cleaned_X_scaled,
    ) = train_final_model(
        X_train=X_cleaned,
        y_train=y_cleaned,
    )

    # External validation of cleaned model
    (
        cleaned_generalization_metrics,
        cleaned_generalization_predictions,
    ) = evaluate_final_model(
        model=cleaned_model,
        scaler=cleaned_scaler,
        df_evaluation=df_generalization,
        X_evaluation=X_generalization,
        y_evaluation=y_generalization,
        dataset_name=(
            f"External Generalization - Model Without "
            f"{N_OUTLIERS_TO_REMOVE} High-Error Samples"
        ),
        output_path=OUTPUT_CLEANED_GENERALIZATION,
    )

    # --------------------------------------------------------
    # 6. Save combined summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            original_loocv_metrics,
            cleaned_loocv_metrics,
            original_generalization_metrics,
            cleaned_generalization_metrics,
        ]
    )

    summary_df.to_csv(
        OUTPUT_PERFORMANCE_SUMMARY,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "\n================================================"
    )
    print(
        "FINAL PERFORMANCE COMPARISON"
    )
    print(
        "================================================"
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nSaved summary to:",
        OUTPUT_PERFORMANCE_SUMMARY,
    )

    # --------------------------------------------------------
    # Direct external-generalization comparison
    # --------------------------------------------------------

    print(
        "\n---------------- Generalization Comparison ----------------"
    )

    print(
        "Original final model:"
    )
    print(
        f"  RMSE: "
        f"{original_generalization_metrics['RMSE']:.4f}"
    )
    print(
        f"  MAE:  "
        f"{original_generalization_metrics['MAE']:.4f}"
    )
    print(
        f"  R²:   "
        f"{original_generalization_metrics['R2']:.4f}"
    )

    print(
        f"\nFinal model after removing "
        f"{N_OUTLIERS_TO_REMOVE} samples:"
    )
    print(
        f"  RMSE: "
        f"{cleaned_generalization_metrics['RMSE']:.4f}"
    )
    print(
        f"  MAE:  "
        f"{cleaned_generalization_metrics['MAE']:.4f}"
    )
    print(
        f"  R²:   "
        f"{cleaned_generalization_metrics['R2']:.4f}"
    )


if __name__ == "__main__":
    main()