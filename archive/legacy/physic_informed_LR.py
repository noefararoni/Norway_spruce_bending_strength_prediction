# Physic_informed_LR.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    K_FOLDS,
    RANDOM_STATE,
    LEARNING_RATE,
    N_ITERATIONS,
    PHYSICS_LAMBDA,
    WIDTH_COL,
    HEIGHT_COL,
    SPAN_COL,
)

from data_preprocessing import (
    load_data,
    merge_data,
    select_features,
    clean_data,
    split_X_y,
)


class PhysicsInformedLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=5000, physics_lambda=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.physics_lambda = physics_lambda
        self.weights = None
        self.bias = None
        self.alpha = None

    def fit(self, X_scaled, X_original, y):
        n_samples, n_features = X_scaled.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        width = X_original[WIDTH_COL].values
        height = X_original[HEIGHT_COL].values
        span = X_original[SPAN_COL].values

        eps = 1e-8

        # Physics relation:
        # fb = alpha * span / (width * height^2)
        # alpha represents the unknown term related to ultimate force
        phi = span / (width * height**2 + eps)

        # Initialize alpha from data
        self.alpha = np.sum(y * phi) / (np.sum(phi**2) + eps)

        for _ in range(self.n_iterations):
            # Linear regression prediction
            y_pred = np.dot(X_scaled, self.weights) + self.bias

            # Physics-based prediction
            y_phys = self.alpha * phi

            # Data loss gradient
            error_data = y_pred - y

            # Physics loss gradient
            error_physics = y_pred - y_phys

            d_y_pred = (
                (2 / n_samples) * error_data
                + self.physics_lambda * (2 / n_samples) * error_physics
            )

            dw = np.dot(X_scaled.T, d_y_pred)
            db = np.sum(d_y_pred)

            # Gradient for alpha
            d_alpha = self.physics_lambda * (2 / n_samples) * np.sum(
                (y_phys - y_pred) * phi
            )

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            self.alpha -= self.learning_rate * d_alpha

    def predict(self, X_scaled):
        return np.dot(X_scaled, self.weights) + self.bias


def cross_validate_physics_informed_lr(X, y):
    print("\n---------------- Physics-Informed Linear Regression ----------------")
    print("Constraint: fb ≈ alpha * span / (width * height^2)")
    print(f"Using {K_FOLDS}-Fold Cross Validation")

    kfold = KFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    mse_scores = []
    rmse_scores = []
    r2_scores = []
    alpha_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"\nFold {fold}")

        X_train_original = X.iloc[train_index]
        X_test_original = X.iloc[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_original.values)
        X_test_scaled = scaler.transform(X_test_original.values)

        model = PhysicsInformedLinearRegression(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            physics_lambda=PHYSICS_LAMBDA
        )

        model.fit(X_train_scaled, X_train_original, y_train)

        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        alpha_scores.append(model.alpha)

        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R²:", r2)
        print("Learned alpha:", model.alpha)

    print("\n---------------- Final Cross-Validation Results ----------------")
    print(f"Average MSE:  {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"Average R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average alpha:{np.mean(alpha_scores):.4f} ± {np.std(alpha_scores):.4f}")

    return rmse_scores, r2_scores


def main():
    df_A, df_B, df_C = load_data()
    df_all = merge_data(df_A, df_B, df_C)
    df_selected = select_features(df_all)
    df_clean = clean_data(df_selected)

    X, y = split_X_y(df_clean)

    cross_validate_physics_informed_lr(X, y)


if __name__ == "__main__":
    main()