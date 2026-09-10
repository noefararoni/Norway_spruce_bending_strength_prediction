# physics_informed_LR_with_outlier_removal.py

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

from config import (
    LEARNING_RATE,
    N_ITERATIONS,
    PHYSICS_LAMBDA,
    WIDTH_COL,
    HEIGHT_COL,
    SPAN_COL,
)

from data_preprocessing import (
    split_X_y,
    clean_data,
    select_features,
)


# ============================================================
# Paths and configuration
# ============================================================

DATA_DIR = Path("Data")

MERGED_DATASET_PATH = (
    DATA_DIR / "final_clean_merged_dataframe.csv"
)

OUTPUT_ERROR_PATH = (
    DATA_DIR / "pilr_loocv_high_error_samples.csv"
)

OUTPUT_ERROR_CLEANED_PATH = (
    DATA_DIR / "pilr_loocv_after_outlier_removal.csv"
)

OUTPUT_REMOVED_OUTLIERS_PATH = (
    DATA_DIR / "pilr_removed_outliers.csv"
)

OUTPUT_COMPARISON_PATH = (
    DATA_DIR / "pilr_before_after_outlier_comparison.csv"
)

N_OUTLIERS_TO_REMOVE = 5


# ============================================================
# Physics-Informed Linear Regression
# ============================================================

class PhysicsInformedLinearRegression:
    """
    Physics-informed linear regression for bending strength.

    Data-driven prediction:
        y_pred = X_scaled @ weights + bias

    Physics relation:
        bending_strength =
            3 * F_u * span / (2 * width * height^2)

    Total loss:
        total_loss =
            data_loss
            + physics_lambda * physics_loss

    where:
        data_loss =
            mean((y_pred - y_measured)^2)

        physics_loss =
            mean((y_pred - y_physics)^2)

    The ultimate force is constrained to remain positive:

        F_u = exp(alpha)
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=5000,
        physics_lambda=0.1,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if n_iterations <= 0:
            raise ValueError(
                "n_iterations must be greater than zero."
            )

        if physics_lambda < 0:
            raise ValueError(
                "physics_lambda cannot be negative."
            )

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.physics_lambda = physics_lambda

        self.weights = None
        self.bias = None

        self.log_force_alpha = None
        self.force_ultimate = None

        self.loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []

    @staticmethod
    def calculate_geometry_factor(X_original):
        """
        Calculates:

            g = 3 * span / (2 * width * height^2)

        so that:

            bending_strength = F_u * g
        """

        required_columns = [
            WIDTH_COL,
            HEIGHT_COL,
            SPAN_COL,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in X_original.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing geometry columns required by "
                f"the physics equation: {missing_columns}"
            )

        width = (
            X_original[WIDTH_COL]
            .to_numpy(dtype=float)
        )

        height = (
            X_original[HEIGHT_COL]
            .to_numpy(dtype=float)
        )

        span = (
            X_original[SPAN_COL]
            .to_numpy(dtype=float)
        )

        if np.any(width <= 0):
            raise ValueError(
                "All width values must be greater than zero."
            )

        if np.any(height <= 0):
            raise ValueError(
                "All height values must be greater than zero."
            )

        if np.any(span <= 0):
            raise ValueError(
                "All test-span values must be greater than zero."
            )

        geometry_factor = (
            3.0 * span
        ) / (
            2.0 * width * height ** 2
        )

        return geometry_factor

    def fit(
        self,
        X_scaled,
        X_original,
        y,
    ):
        X_scaled = np.asarray(
            X_scaled,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        if X_scaled.ndim != 2:
            raise ValueError(
                "X_scaled must be a two-dimensional array."
            )

        if len(X_scaled) != len(y):
            raise ValueError(
                "X_scaled and y must contain the same "
                "number of samples."
            )

        if len(X_original) != len(y):
            raise ValueError(
                "X_original and y must contain the same "
                "number of samples."
            )

        n_samples, n_features = X_scaled.shape

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.bias = 0.0

        geometry_factor = (
            self.calculate_geometry_factor(
                X_original
            )
        )

        eps = 1e-12

        # Least-squares initialization of F_u:
        #
        # y ≈ F_u * geometry_factor
        force_initial = (
            np.dot(
                y,
                geometry_factor,
            )
            / (
                np.dot(
                    geometry_factor,
                    geometry_factor,
                )
                + eps
            )
        )

        force_initial = max(
            float(force_initial),
            eps,
        )

        self.log_force_alpha = np.log(
            force_initial
        )

        self.loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []

        for _ in range(
            self.n_iterations
        ):
            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            y_pred = (
                np.dot(
                    X_scaled,
                    self.weights,
                )
                + self.bias
            )

            force_ultimate = np.exp(
                self.log_force_alpha
            )

            y_physics = (
                force_ultimate
                * geometry_factor
            )

            # ------------------------------------------------
            # Loss terms
            # ------------------------------------------------

            data_error = (
                y_pred - y
            )

            physics_error = (
                y_pred - y_physics
            )

            data_loss = np.mean(
                data_error ** 2
            )

            physics_loss = np.mean(
                physics_error ** 2
            )

            total_loss = (
                data_loss
                + self.physics_lambda
                * physics_loss
            )

            self.data_loss_history.append(
                float(data_loss)
            )

            self.physics_loss_history.append(
                float(physics_loss)
            )

            self.loss_history.append(
                float(total_loss)
            )

            # ------------------------------------------------
            # Gradients with respect to prediction
            # ------------------------------------------------

            prediction_gradient = (
                (2.0 / n_samples)
                * data_error
                + self.physics_lambda
                * (2.0 / n_samples)
                * physics_error
            )

            weight_gradient = np.dot(
                X_scaled.T,
                prediction_gradient,
            )

            bias_gradient = np.sum(
                prediction_gradient
            )

            # ------------------------------------------------
            # Gradient for F_u
            # ------------------------------------------------

            force_gradient = (
                self.physics_lambda
                * np.mean(
                    -2.0
                    * physics_error
                    * geometry_factor
                )
            )

            # F_u = exp(alpha)
            alpha_gradient = (
                force_gradient
                * force_ultimate
            )

            # ------------------------------------------------
            # Parameter update
            # ------------------------------------------------

            self.weights -= (
                self.learning_rate
                * weight_gradient
            )

            self.bias -= (
                self.learning_rate
                * bias_gradient
            )

            self.log_force_alpha -= (
                self.learning_rate
                * alpha_gradient
            )

            # Stop if numerical instability occurs.
            if not np.isfinite(
                self.log_force_alpha
            ):
                raise FloatingPointError(
                    "Training became numerically unstable. "
                    "Reduce LEARNING_RATE or PHYSICS_LAMBDA."
                )

        self.force_ultimate = float(
            np.exp(
                self.log_force_alpha
            )
        )

        return self

    def predict(
        self,
        X_scaled,
    ):
        if self.weights is None:
            raise RuntimeError(
                "The model must be fitted before prediction."
            )

        X_scaled = np.asarray(
            X_scaled,
            dtype=float,
        )

        return (
            np.dot(
                X_scaled,
                self.weights,
            )
            + self.bias
        )

    def predict_physics(
        self,
        X_original,
    ):
        if self.force_ultimate is None:
            raise RuntimeError(
                "The model must be fitted before "
                "physics prediction."
            )

        geometry_factor = (
            self.calculate_geometry_factor(
                X_original
            )
        )

        return (
            self.force_ultimate
            * geometry_factor
        )


# ============================================================
# Load dataset
# ============================================================

def load_merged_dataset(
    path=MERGED_DATASET_PATH,
):
    print(
        "\n---------------- Load Merged Dataset ----------------",
        flush=True,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find merged dataset at: {path}\n"
            "Please run data_preprocessing.py first."
        )

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
    )

    print(
        "Loaded dataset:",
        path,
        flush=True,
    )

    print(
        "Dataset shape:",
        df.shape,
        flush=True,
    )

    print(
        "Columns:",
        list(df.columns),
        flush=True,
    )

    return df


# ============================================================
# Physics-informed LOOCV
# ============================================================

def leave_one_out_physics_informed_lr(
    X,
    y,
    output_error_path=None,
    experiment_name="Physics-Informed LOOCV",
):
    print(
        f"\n---------------- {experiment_name} ----------------",
        flush=True,
    )

    print(
        "Using Leave-One-Out Cross-Validation",
        flush=True,
    )

    print(
        f"Physics lambda: {PHYSICS_LAMBDA}",
        flush=True,
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    y_physics_all = []

    mse_scores = []
    rmse_scores = []
    absolute_errors = []
    force_scores = []
    physics_residual_scores = []

    error_records = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        loo.split(X),
        start=1,
    ):
        X_train_original = (
            X.iloc[train_index]
            .copy()
        )

        X_test_original = (
            X.iloc[test_index]
            .copy()
        )

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        # Fit scaler only on the training portion.
        scaler = StandardScaler()

        X_train_scaled = (
            scaler
            .fit_transform(
                X_train_original
            )
        )

        X_test_scaled = (
            scaler
            .transform(
                X_test_original
            )
        )

        model = (
            PhysicsInformedLinearRegression(
                learning_rate=LEARNING_RATE,
                n_iterations=N_ITERATIONS,
                physics_lambda=PHYSICS_LAMBDA,
            )
        )

        model.fit(
            X_scaled=X_train_scaled,
            X_original=X_train_original,
            y=y_train,
        )

        y_pred = model.predict(
            X_test_scaled
        )

        y_physics = model.predict_physics(
            X_test_original
        )

        true_value = float(
            y_test[0]
        )

        predicted_value = float(
            y_pred[0]
        )

        physics_value = float(
            y_physics[0]
        )

        error = (
            true_value
            - predicted_value
        )

        absolute_error = abs(
            error
        )

        squared_error = (
            error ** 2
        )

        physics_residual = (
            predicted_value
            - physics_value
        )

        physics_residual_squared = (
            physics_residual ** 2
        )

        mse = mean_squared_error(
            y_test,
            y_pred,
        )

        rmse = np.sqrt(
            mse
        )

        y_true_all.append(
            true_value
        )

        y_pred_all.append(
            predicted_value
        )

        y_physics_all.append(
            physics_value
        )

        mse_scores.append(
            float(mse)
        )

        rmse_scores.append(
            float(rmse)
        )

        absolute_errors.append(
            absolute_error
        )

        force_scores.append(
            model.force_ultimate
        )

        physics_residual_scores.append(
            physics_residual_squared
        )

        original_index = int(
            X.index[
                test_index[0]
            ]
        )

        error_records.append(
            {
                "fold": fold,
                "original_index": original_index,
                "current_position": int(
                    test_index[0]
                ),
                "true_value": true_value,
                "predicted_value": predicted_value,
                "physics_prediction": physics_value,
                "error": error,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                "rmse": float(rmse),
                "physics_residual": physics_residual,
                "physics_residual_squared": (
                    physics_residual_squared
                ),
                "learned_ultimate_force": (
                    model.force_ultimate
                ),
                "final_total_loss": (
                    model.loss_history[-1]
                ),
                "final_data_loss": (
                    model.data_loss_history[-1]
                ),
                "final_physics_loss": (
                    model.physics_loss_history[-1]
                ),
                **X.iloc[
                    test_index[0]
                ].to_dict(),
            }
        )

        print(
            f"Fold {fold}: "
            f"True={true_value:.4f}, "
            f"Predicted={predicted_value:.4f}, "
            f"Physics={physics_value:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"Fu={model.force_ultimate:.4f}",
            flush=True,
        )

    # ========================================================
    # Aggregate metrics
    # ========================================================

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

    final_physics_mse = (
        mean_squared_error(
            y_pred_all,
            y_physics_all,
        )
    )

    mean_error = np.mean(
        np.asarray(y_true_all)
        - np.asarray(y_pred_all)
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
        "\n---------------- Final Leave-One-Out Results ----------------",
        flush=True,
    )

    print(
        f"Average MSE:  "
        f"{np.mean(mse_scores):.4f} "
        f"± {np.std(mse_scores):.4f}",
        flush=True,
    )

    print(
        f"Average absolute error: "
        f"{np.mean(absolute_errors):.4f} "
        f"± {np.std(absolute_errors):.4f}",
        flush=True,
    )

    print(
        f"Final MSE:    {final_mse:.4f}",
        flush=True,
    )

    print(
        f"Final RMSE:   {final_rmse:.4f}",
        flush=True,
    )

    print(
        f"Final MAE:    {final_mae:.4f}",
        flush=True,
    )

    print(
        f"Final R²:     {final_r2:.4f}",
        flush=True,
    )

    print(
        f"Mean error:   {mean_error:.4f}",
        flush=True,
    )

    print(
        f"Physics MSE:  {final_physics_mse:.4f}",
        flush=True,
    )

    print(
        f"Average Fu:   "
        f"{np.mean(force_scores):.4f} "
        f"± {np.std(force_scores):.4f}",
        flush=True,
    )

    print(
        "\n---------------- Top 10 Highest-Error Samples ----------------"
    )

    print(
        error_df
        .head(10)
        .to_string(index=False)
    )

    if output_error_path is not None:
        output_error_path = Path(
            output_error_path
        )

        output_error_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        error_df.to_csv(
            output_error_path,
            sep=";",
            decimal=",",
            index=False,
        )

        print(
            "\nSaved PILR LOOCV error analysis to:",
            output_error_path,
        )

    results = {
        "experiment": experiment_name,
        "number_of_samples": len(y),
        "physics_lambda": PHYSICS_LAMBDA,
        "final_mse": final_mse,
        "final_rmse": final_rmse,
        "final_mae": final_mae,
        "final_r2": final_r2,
        "mean_error": mean_error,
        "physics_residual_mse": final_physics_mse,
        "average_force_ultimate": np.mean(
            force_scores
        ),
        "std_force_ultimate": np.std(
            force_scores
        ),
        "average_fold_mse": np.mean(
            mse_scores
        ),
        "std_fold_mse": np.std(
            mse_scores
        ),
        "average_absolute_error": np.mean(
            absolute_errors
        ),
        "std_absolute_error": np.std(
            absolute_errors
        ),
    }

    return (
        results,
        error_df,
    )


# ============================================================
# Remove highest-error specimens
# ============================================================

def remove_top_outliers_by_error(
    X,
    y,
    error_df,
    n_outliers=5,
):
    """
    Removes the samples with the highest absolute PILR
    LOOCV prediction errors.
    """

    if n_outliers < 0:
        raise ValueError(
            "n_outliers cannot be negative."
        )

    if n_outliers >= len(X):
        raise ValueError(
            "n_outliers must be smaller than "
            "the number of available samples."
        )

    if n_outliers == 0:
        return (
            X.copy(),
            np.asarray(y).copy(),
            error_df.head(0).copy(),
        )

    removed_outliers = (
        error_df
        .head(n_outliers)
        .copy()
    )

    outlier_indices = (
        removed_outliers[
            "original_index"
        ]
        .astype(int)
        .to_numpy()
    )

    print(
        "\n---------------- Removing Outliers ----------------"
    )

    print(
        f"Number of removed outliers: {n_outliers}"
    )

    print(
        "Removed original indices:",
        outlier_indices,
    )

    X_cleaned = (
        X
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

    print(
        "Original dataset size:",
        len(X),
    )

    print(
        "Cleaned dataset size:",
        len(X_cleaned),
    )

    OUTPUT_REMOVED_OUTLIERS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    removed_outliers.to_csv(
        OUTPUT_REMOVED_OUTLIERS_PATH,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "Saved removed outliers to:",
        OUTPUT_REMOVED_OUTLIERS_PATH,
    )

    return (
        X_cleaned,
        y_cleaned,
        removed_outliers,
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
    # Load and preprocess dataset
    # --------------------------------------------------------

    df_all = load_merged_dataset()

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

    print(
        "\nSelected features:",
        list(X.columns),
    )

    print(
        "Number of specimens:",
        len(X),
    )

    # --------------------------------------------------------
    # First run: original dataset
    # --------------------------------------------------------

    print(
        "\n\n================================================"
    )

    print(
        "FIRST RUN: ORIGINAL DATASET"
    )

    print(
        "================================================"
    )

    (
        original_results,
        original_error_df,
    ) = leave_one_out_physics_informed_lr(
        X=X,
        y=y,
        output_error_path=OUTPUT_ERROR_PATH,
        experiment_name=(
            "Physics-Informed LR - Original Dataset"
        ),
    )

    # --------------------------------------------------------
    # Remove highest-error specimens
    # --------------------------------------------------------

    (
        X_cleaned,
        y_cleaned,
        removed_outliers,
    ) = remove_top_outliers_by_error(
        X=X,
        y=y,
        error_df=original_error_df,
        n_outliers=N_OUTLIERS_TO_REMOVE,
    )

    # --------------------------------------------------------
    # Second run: cleaned dataset
    # --------------------------------------------------------

    print(
        "\n\n================================================"
    )

    print(
        f"SECOND RUN: AFTER REMOVING TOP "
        f"{N_OUTLIERS_TO_REMOVE} OUTLIERS"
    )

    print(
        "================================================"
    )

    (
        cleaned_results,
        cleaned_error_df,
    ) = leave_one_out_physics_informed_lr(
        X=X_cleaned,
        y=y_cleaned,
        output_error_path=OUTPUT_ERROR_CLEANED_PATH,
        experiment_name=(
            f"Physics-Informed LR - Without Top "
            f"{N_OUTLIERS_TO_REMOVE} Outliers"
        ),
    )

    # --------------------------------------------------------
    # Performance comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        [
            original_results,
            cleaned_results,
        ]
    )

    comparison_df.to_csv(
        OUTPUT_COMPARISON_PATH,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "\n\n================================================"
    )

    print(
        "PERFORMANCE COMPARISON"
    )

    print(
        "================================================"
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\n---------------- Main Metric Changes ----------------"
    )

    print(
        f"Original RMSE: "
        f"{original_results['final_rmse']:.4f}"
    )

    print(
        f"Cleaned RMSE:  "
        f"{cleaned_results['final_rmse']:.4f}"
    )

    print(
        f"Original MAE:  "
        f"{original_results['final_mae']:.4f}"
    )

    print(
        f"Cleaned MAE:   "
        f"{cleaned_results['final_mae']:.4f}"
    )

    print(
        f"Original R²:   "
        f"{original_results['final_r2']:.4f}"
    )

    print(
        f"Cleaned R²:    "
        f"{cleaned_results['final_r2']:.4f}"
    )

    print(
        f"Original physics residual MSE: "
        f"{original_results['physics_residual_mse']:.4f}"
    )

    print(
        f"Cleaned physics residual MSE:  "
        f"{cleaned_results['physics_residual_mse']:.4f}"
    )

    print(
        f"Original average Fu: "
        f"{original_results['average_force_ultimate']:.4f}"
    )

    print(
        f"Cleaned average Fu:  "
        f"{cleaned_results['average_force_ultimate']:.4f}"
    )

    print(
        "\nSaved comparison to:",
        OUTPUT_COMPARISON_PATH,
    )


if __name__ == "__main__":
    main()