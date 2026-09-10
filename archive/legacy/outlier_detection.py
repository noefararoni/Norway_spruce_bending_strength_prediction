# normal_LR.py

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import K_FOLDS, RANDOM_STATE, LEARNING_RATE, N_ITERATIONS
from data_preprocessing import (
    load_data,
    merge_data,
    select_features,
    clean_data,
    split_X_y,
)


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

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2 / n_samples) * np.dot(X.T, y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 1000 == 0:
                print(f"Iteration {i}/{self.n_iterations}", flush=True)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def cross_validate_linear_regression(X, y, save_prefix="original"):
    print("\n---------------- Linear Regression From Scratch ----------------", flush=True)
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
    all_fold_results = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"\nStarting Fold {fold}", flush=True)

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
        r2_scores.append(r2)
        rmse_scores.append(rmse)

        fold_results = X.iloc[test_index].copy()

        fold_results["fold"] = fold

        # Real dataframe index labels
        fold_results["df_index"] = X.index[test_index]

        # Positional indices
        fold_results["position_index"] = test_index

        fold_results["y_true"] = y_test
        fold_results["y_pred"] = y_pred
        fold_results["residual"] = fold_results["y_true"] - fold_results["y_pred"]
        fold_results["absolute_error"] = np.abs(fold_results["residual"])
        fold_results["squared_error"] = fold_results["residual"] ** 2

        all_fold_results.append(fold_results)

        print(f"Fold {fold} MSE:  {mse:.4f}", flush=True)
        print(f"Fold {fold} R²:   {r2:.4f}", flush=True)
        print(f"Fold {fold} RMSE: {rmse:.4f}", flush=True)

    worst_fold = np.argmax(mse_scores) + 1
    worst_fold_results = all_fold_results[worst_fold - 1]

    worst_samples = (
        worst_fold_results
        .sort_values("squared_error", ascending=False)
        .head(20)
        .copy()
    )

    worst_df_indices = worst_samples["df_index"].tolist()
    worst_position_indices = worst_samples["position_index"].tolist()

    print("\n---------------- Final Cross-Validation Results ----------------", flush=True)
    print(f"Average MSE:  {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}", flush=True)
    print(f"Average R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}", flush=True)
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}", flush=True)

    print("\n---------------- Worst Fold Analysis ----------------", flush=True)
    print(f"Worst fold: Fold {worst_fold}", flush=True)
    print(f"Worst fold MSE: {mse_scores[worst_fold - 1]:.4f}", flush=True)
    print(f"Worst fold R²:  {r2_scores[worst_fold - 1]:.4f}", flush=True)

    print("\nIndices to remove from dataframe:", flush=True)
    print(worst_df_indices, flush=True)

    print("\nWorst samples:", flush=True)
    print(worst_samples, flush=True)

    worst_samples.to_csv(f"{save_prefix}_worst_fold_samples.csv", index=False)

    with open(f"{save_prefix}_worst_indices.txt", "w") as f:
        for idx in worst_df_indices:
            f.write(str(idx) + "\n")

    print(f"\nSaved worst samples to: {save_prefix}_worst_fold_samples.csv", flush=True)
    print(f"Saved worst indices to: {save_prefix}_worst_indices.txt", flush=True)

    return mse_scores, r2_scores, rmse_scores, worst_df_indices


def main():
    print("Starting script...", flush=True)

    df_A, df_B, df_C = load_data()
    print("Data loaded", flush=True)

    df_all = merge_data(df_A, df_B, df_C)
    print("Data merged:", df_all.shape, flush=True)

    df_selected = select_features(df_all)
    print("Features selected:", df_selected.shape, flush=True)

    df_clean = clean_data(df_selected)
    print("Data cleaned:", df_clean.shape, flush=True)

    X, y = split_X_y(df_clean)
    print("X shape before removal:", X.shape, flush=True)
    print("y shape before removal:", y.shape, flush=True)

    print("\n========== First Run: Before Removing Bad Samples ==========", flush=True)

    mse_scores, r2_scores, rmse_scores, worst_indices = cross_validate_linear_regression(
        X,
        y,
        save_prefix="before_removal"
    )

    print("\n========== Removing Bad Samples ==========", flush=True)

    df_clean_removed = df_clean.drop(index=worst_indices)

    print("Removed indices:", worst_indices, flush=True)
    print("Data shape before removal:", df_clean.shape, flush=True)
    print("Data shape after removal:", df_clean_removed.shape, flush=True)

    df_clean_removed.to_csv("df_clean_after_removing_bad_samples.csv", index=False)
    print("Saved cleaned dataframe to: df_clean_after_removing_bad_samples.csv", flush=True)

    X_removed, y_removed = split_X_y(df_clean_removed)

    print("X shape after removal:", X_removed.shape, flush=True)
    print("y shape after removal:", y_removed.shape, flush=True)

    print("\n========== Second Run: After Removing Bad Samples ==========", flush=True)

    cross_validate_linear_regression(
        X_removed,
        y_removed,
        save_prefix="after_removal"
    )

    print("\nFinished", flush=True)


if __name__ == "__main__":
    main()