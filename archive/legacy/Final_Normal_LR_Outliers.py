# normal_LR_with_outlier_removal.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import LEARNING_RATE, N_ITERATIONS
from data_preprocessing import split_X_y, clean_data, select_features


DATA_DIR = Path("Data")
MERGED_DATASET_PATH = DATA_DIR / "final_clean_merged_dataframe.csv"

OUTPUT_ERROR_PATH = "loocv_high_error_samples.csv"
OUTPUT_ERROR_CLEANED_PATH = "loocv_after_outlier_removal.csv"

N_OUTLIERS_TO_REMOVE = 5


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def load_merged_dataset(path=MERGED_DATASET_PATH):
    print("\n---------------- Load Merged Dataset ----------------", flush=True)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find merged dataset at: {path}\n"
            "Please run data_preprocessing.py first."
        )

    df = pd.read_csv(path, sep=";", decimal=",")

    print("Loaded dataset:", path, flush=True)
    print("Dataset shape:", df.shape, flush=True)
    print("Columns:", list(df.columns), flush=True)

    return df


def leave_one_out_linear_regression(
    X,
    y,
    output_error_path=None,
    experiment_name="LOOCV Linear Regression"
):
    print(f"\n---------------- {experiment_name} ----------------", flush=True)
    print("Using Leave-One-Out Cross-Validation", flush=True)

    y = np.array(y)
    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    mse_scores = []
    rmse_scores = []
    error_records = []

    for fold, (train_index, test_index) in enumerate(loo.split(X), start=1):
        X_train = X.iloc[train_index].values
        X_test = X.iloc[test_index].values

        y_train = y[train_index]
        y_test = y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegressionScratch(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS
        )

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        error = y_test[0] - y_pred[0]
        abs_error = abs(error)
        squared_error = error ** 2

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mse_scores.append(mse)
        rmse_scores.append(rmse)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

        error_records.append({
            "fold": fold,
            "original_index": int(X.index[test_index[0]]),
            "current_index": int(test_index[0]),
            "true_value": float(y_test[0]),
            "predicted_value": float(y_pred[0]),
            "error": float(error),
            "absolute_error": float(abs_error),
            "squared_error": float(squared_error),
            "rmse": float(rmse),
            **X.iloc[test_index[0]].to_dict()
        })

        print(
            f"Fold {fold}: True={y_test[0]:.4f}, "
            f"Predicted={y_pred[0]:.4f}, RMSE={rmse:.4f}",
            flush=True
        )

    final_mse = mean_squared_error(y_true_all, y_pred_all)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(y_true_all, y_pred_all)

    error_df = pd.DataFrame(error_records)
    error_df = error_df.sort_values(by="absolute_error", ascending=False)

    print("\n---------------- Final Leave-One-Out Results ----------------", flush=True)
    print(f"Average MSE:   {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}", flush=True)
    print(f"Average RMSE:  {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}", flush=True)
    print(f"Final MSE:     {final_mse:.4f}", flush=True)
    print(f"Final RMSE:    {final_rmse:.4f}", flush=True)
    print(f"Final R²:      {final_r2:.4f}", flush=True)

    print("\n---------------- Top 10 Highest-Error Samples ----------------")
    print(error_df.head(10).to_string(index=False))

    if output_error_path is not None:
        error_df.to_csv(output_error_path, index=False)
        print(f"\nSaved LOOCV error analysis to: {output_error_path}")

    results = {
        "final_mse": final_mse,
        "final_rmse": final_rmse,
        "final_r2": final_r2,
        "average_mse": np.mean(mse_scores),
        "std_mse": np.std(mse_scores),
        "average_rmse": np.mean(rmse_scores),
        "std_rmse": np.std(rmse_scores),
    }

    return results, error_df


def remove_top_outliers_by_error(X, y, error_df, n_outliers=5):
    """
    Removes the top n_outliers samples with the highest LOOCV absolute error.
    """

    outlier_indices = error_df.head(n_outliers)["original_index"].values

    print("\n---------------- Removing Outliers ----------------")
    print(f"Number of removed outliers: {n_outliers}")
    print("Removed original indices:", outlier_indices)

    X_cleaned = X.drop(index=outlier_indices)
    y_series = pd.Series(y, index=X.index)
    y_cleaned = y_series.drop(index=outlier_indices).values

    print("Original dataset size:", len(X))
    print("Cleaned dataset size:", len(X_cleaned))

    return X_cleaned, y_cleaned


def main():
    df_all = load_merged_dataset()

    df_selected = select_features(df_all)
    df_clean = clean_data(df_selected)

    X, y = split_X_y(df_clean)

    print("\n\n================================================")
    print("FIRST RUN: ORIGINAL DATASET")
    print("================================================")

    original_results, error_df = leave_one_out_linear_regression(
        X=X,
        y=y,
        output_error_path=OUTPUT_ERROR_PATH,
        experiment_name="Original Dataset"
    )

    X_cleaned, y_cleaned = remove_top_outliers_by_error(
        X=X,
        y=y,
        error_df=error_df,
        n_outliers=N_OUTLIERS_TO_REMOVE
    )

    print("\n\n================================================")
    print(f"SECOND RUN: AFTER REMOVING TOP {N_OUTLIERS_TO_REMOVE} OUTLIERS")
    print("================================================")

    cleaned_results, cleaned_error_df = leave_one_out_linear_regression(
        X=X_cleaned,
        y=y_cleaned,
        output_error_path=OUTPUT_ERROR_CLEANED_PATH,
        experiment_name=f"Dataset Without Top {N_OUTLIERS_TO_REMOVE} Outliers"
    )

    print("\n\n================================================")
    print("PERFORMANCE COMPARISON")
    print("================================================")

    print(f"Original RMSE: {original_results['final_rmse']:.4f}")
    print(f"Cleaned RMSE:  {cleaned_results['final_rmse']:.4f}")

    print(f"Original R²:   {original_results['final_r2']:.4f}")
    print(f"Cleaned R²:    {cleaned_results['final_r2']:.4f}")

    print(f"Original MSE:  {original_results['final_mse']:.4f}")
    print(f"Cleaned MSE:   {cleaned_results['final_mse']:.4f}")


if __name__ == "__main__":
    main()