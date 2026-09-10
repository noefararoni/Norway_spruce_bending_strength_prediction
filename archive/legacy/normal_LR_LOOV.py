# normal_LR.py

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

            dw = (2 / n_samples) * np.dot(X.T, y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def load_merged_dataset(path=MERGED_DATASET_PATH):
    print("\n---------------- Load Merged Dataset ----------------", flush=True)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find merged dataset at: {path}\n"
            "Please run data_preprocessing.py first to generate the file."
        )

    df = pd.read_csv(path, sep=";", decimal=",")

    print("Loaded dataset:", path, flush=True)
    print("Dataset shape:", df.shape, flush=True)
    print("Columns:", list(df.columns), flush=True)

    return df


def leave_one_out_linear_regression(X, y):
    print("\n---------------- Normal Linear Regression From Scratch ----------------", flush=True)
    print("Using Leave-One-Out Cross-Validation", flush=True)

    y = np.array(y)

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    mse_scores = []
    rmse_scores = []

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

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mse_scores.append(mse)
        rmse_scores.append(rmse)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

        print(f"Fold {fold}: True={y_test[0]:.4f}, Predicted={y_pred[0]:.4f}, RMSE={rmse:.4f}", flush=True)

    final_mse = mean_squared_error(y_true_all, y_pred_all)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(y_true_all, y_pred_all)

    print("\n---------------- Final Leave-One-Out Results ----------------", flush=True)
    print(f"Average MSE:   {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}", flush=True)
    print(f"Average RMSE:  {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}", flush=True)
    print(f"Final MSE:     {final_mse:.4f}", flush=True)
    print(f"Final RMSE:    {final_rmse:.4f}", flush=True)
    print(f"Final R²:      {final_r2:.4f}", flush=True)

    return mse_scores, rmse_scores, final_r2


def main():
    df_all = load_merged_dataset()

    df_selected = select_features(df_all)
    df_clean = clean_data(df_selected)

    X, y = split_X_y(df_clean)

    leave_one_out_linear_regression(X, y)


if __name__ == "__main__":
    main()