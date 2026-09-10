# normal_LR.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import K_FOLDS, RANDOM_STATE, LEARNING_RATE, N_ITERATIONS
from data_preprocessing import split_X_y, clean_data,select_features



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

    
    df = pd.read_csv(path, sep=";",decimal=",")
    print(df.columns)
    print("Loaded dataset:", path, flush=True)
    print("Dataset shape:", df.shape, flush=True)
    print("Columns:", list(df.columns), flush=True)

    return df


def cross_validate_linear_regression(X, y):
    print("\n---------------- Normal Linear Regression From Scratch ----------------", flush=True)
    print(f"Using {K_FOLDS}-Fold Cross Validation", flush=True)

    y = np.array(y)

    kfold = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    mse_scores = []
    r2_scores = []
    rmse_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"\nFold {fold}", flush=True)

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
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print("MSE:", mse, flush=True)
        print("R²:", r2, flush=True)
        print("RMSE:", rmse, flush=True)

    print("\n---------------- Final Cross-Validation Results ----------------", flush=True)
    print(f"Average MSE:   {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}", flush=True)
    print(f"Average R²:    {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}", flush=True)
    print(f"Average RMSE:  {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}", flush=True)

    return mse_scores, r2_scores, rmse_scores


def main():
    df_all = load_merged_dataset()
    print("hhhhh",df_all)

    df_selected = select_features(df_all)

    df_clean = clean_data(df_selected)

    X, y = split_X_y(df_clean)

    cross_validate_linear_regression(X, y)


if __name__ == "__main__":
    main()