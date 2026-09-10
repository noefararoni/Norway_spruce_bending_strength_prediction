# normal_LR_with_outlier_removal_and_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
)

from data_preprocessing import (
    split_X_y,
    clean_data,
    select_features,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("Data")

MERGED_DATASET_PATH = (
    DATA_DIR / "final_clean_merged_dataframe.csv"
)

N_OUTLIERS_TO_REMOVE = 5


# ============================================================
# Output paths
# ============================================================

OUTPUT_ERROR_PATH = (
    DATA_DIR / "loocv_high_error_samples.csv"
)

OUTPUT_ERROR_CLEANED_PATH = (
    DATA_DIR / "loocv_after_outlier_removal.csv"
)

OUTPUT_ORIGINAL_PLOT = (
    DATA_DIR / "normal_lr_original_regression_plot.png"
)

OUTPUT_CLEANED_PLOT = (
    DATA_DIR / "normal_lr_cleaned_regression_plot.png"
)

OUTPUT_COMPARISON = (
    DATA_DIR / "normal_lr_performance_comparison.csv"
)


# ============================================================
# Linear Regression from Scratch
# ============================================================

class LinearRegressionScratch:
    """
    Normal multiple linear regression trained with gradient descent.

    Model:
        y_pred = X @ weights + bias

    Loss:
        MSE = mean((y_pred - y_true)^2)
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=5000,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = None

    def fit(
        self,
        X,
        y,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.bias = 0.0

        for _ in range(
            self.n_iterations
        ):
            y_pred = (
                np.dot(
                    X,
                    self.weights,
                )
                + self.bias
            )

            error = (
                y_pred - y
            )

            dw = (
                2.0 / n_samples
            ) * np.dot(
                X.T,
                error,
            )

            db = (
                2.0 / n_samples
            ) * np.sum(
                error
            )

            self.weights -= (
                self.learning_rate
                * dw
            )

            self.bias -= (
                self.learning_rate
                * db
            )

        return self

    def predict(
        self,
        X,
    ):
        if self.weights is None:
            raise RuntimeError(
                "The model must be fitted before prediction."
            )

        X = np.asarray(
            X,
            dtype=float,
        )

        return (
            np.dot(
                X,
                self.weights,
            )
            + self.bias
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
# Regression plot
# ============================================================

def plot_regression_results(
    y_true,
    y_pred,
    title,
    output_path,
):
    """
    Creates a measured-vs-predicted bending strength plot.

    X-axis:
        Measured bending strength

    Y-axis:
        Predicted bending strength

    Dashed diagonal:
        Perfect prediction y = x
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    minimum_value = min(
        np.min(y_true),
        np.min(y_pred),
    )

    maximum_value = max(
        np.max(y_true),
        np.max(y_pred),
    )

    value_range = (
        maximum_value
        - minimum_value
    )

    if value_range == 0:
        value_range = 1.0

    margin = (
        0.08
        * value_range
    )

    plot_min = (
        minimum_value
        - margin
    )

    plot_max = (
        maximum_value
        + margin
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Predictions
    ax.scatter(
        y_true,
        y_pred,
        s=55,
        alpha=0.75,
        label="LOOCV predictions",
    )

    # Perfect prediction line
    ax.plot(
        [
            plot_min,
            plot_max,
        ],
        [
            plot_min,
            plot_max,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction",
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Measured Bending Strength [MPa]",
        fontsize=12,
    )

    ax.set_ylabel(
        "Predicted Bending Strength [MPa]",
        fontsize=12,
    )

    ax.set_title(
        title,
        fontsize=13,
    )

    ax.set_xlim(
        plot_min,
        plot_max,
    )

    ax.set_ylim(
        plot_min,
        plot_max,
    )

    # Equal scaling makes the y=x reference line meaningful.
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper left",
    )

    # --------------------------------------------------------
    # Metric box
    # --------------------------------------------------------

    metric_text = (
        f"MSE = {mse:.2f} MPa²\n"
        f"RMSE = {rmse:.2f} MPa\n"
        f"MAE = {mae:.2f} MPa\n"
        f"$R^2$ = {r2:.3f}"
    )

    ax.text(
        0.97,
        0.03,
        metric_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
        ),
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "\nFigure successfully saved to:"
    )

    print(
        output_path.resolve()
    )

    # --------------------------------------------------------
    # Display figure
    # --------------------------------------------------------

    plt.show()

    # Close after saving and displaying.
    plt.close(
        fig
    )

    return {
        "MSE": mse,
        "RMSE": mae,
        #"MAE": mae,
        "R2": r2,
    }


# ============================================================
# Leave-One-Out Cross-Validation
# ============================================================

def leave_one_out_linear_regression(
    X,
    y,
    output_error_path=None,
    regression_plot_path=None,
    experiment_name="LOOCV Linear Regression",
):
    print(
        f"\n---------------- {experiment_name} ----------------",
        flush=True,
    )

    print(
        "Using Leave-One-Out Cross-Validation",
        flush=True,
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    mse_scores = []
    rmse_scores = []

    error_records = []

    # ========================================================
    # LOOCV loop
    # ========================================================

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        loo.split(X),
        start=1,
    ):
        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        X_train = (
            X.iloc[
                train_index
            ]
            .values
        )

        X_test = (
            X.iloc[
                test_index
            ]
            .values
        )

        y_train = (
            y[
                train_index
            ]
        )

        y_test = (
            y[
                test_index
            ]
        )

        # ----------------------------------------------------
        # Standardization
        #
        # IMPORTANT:
        # Fit scaler only on training data.
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = LinearRegressionScratch(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        # ----------------------------------------------------
        # Predict held-out specimen
        # ----------------------------------------------------

        y_pred = (
            model.predict(
                X_test_scaled
            )
        )

        true_value = float(
            y_test[0]
        )

        predicted_value = float(
            y_pred[0]
        )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

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

        mse = mean_squared_error(
            y_test,
            y_pred,
        )

        rmse = np.sqrt(
            mse
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        mse_scores.append(
            float(mse)
        )

        rmse_scores.append(
            float(rmse)
        )

        y_true_all.append(
            true_value
        )

        y_pred_all.append(
            predicted_value
        )

        error_records.append(
            {
                "fold": fold,

                "original_index": int(
                    X.index[
                        test_index[0]
                    ]
                ),

                "current_index": int(
                    test_index[0]
                ),

                "true_value": (
                    true_value
                ),

                "predicted_value": (
                    predicted_value
                ),

                "error": (
                    error
                ),

                "absolute_error": (
                    absolute_error
                ),

                "squared_error": (
                    squared_error
                ),

                "rmse": (
                    float(rmse)
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
            f"RMSE={rmse:.4f}",
            flush=True,
        )

    # ========================================================
    # Final LOOCV metrics
    # ========================================================

    y_true_all = np.asarray(
        y_true_all,
        dtype=float,
    )

    y_pred_all = np.asarray(
        y_pred_all,
        dtype=float,
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

    mean_error = np.mean(
        y_true_all
        - y_pred_all
    )

    # ========================================================
    # Error dataframe
    # ========================================================

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

    # ========================================================
    # Print final metrics
    # ========================================================

    print(
        "\n---------------- Final Leave-One-Out Results ----------------",
        flush=True,
    )

    print(
        f"Average MSE:   "
        f"{np.mean(mse_scores):.4f} "
        f"± {np.std(mse_scores):.4f}",
        flush=True,
    )

    print(
        f"Average RMSE:  "
        f"{np.mean(rmse_scores):.4f} "
        f"± {np.std(rmse_scores):.4f}",
        flush=True,
    )

    print(
        f"Final MSE:     "
        f"{final_mse:.4f}",
        flush=True,
    )

    print(
        f"Final RMSE:    "
        f"{final_rmse:.4f}",
        flush=True,
    )

    print(
        f"Final MAE:     "
        f"{final_mae:.4f}",
        flush=True,
    )

    print(
        f"Final R²:      "
        f"{final_r2:.4f}",
        flush=True,
    )

    print(
        f"Mean error:    "
        f"{mean_error:.4f}",
        flush=True,
    )

    # ========================================================
    # Top highest-error specimens
    # ========================================================

    print(
        "\n---------------- Top 10 Highest-Error Samples ----------------"
    )

    print(
        error_df
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save error table
    # ========================================================

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
            "\nSaved LOOCV error analysis to:"
        )

        print(
            output_error_path.resolve()
        )

    # ========================================================
    # Create regression plot
    # ========================================================

    if regression_plot_path is not None:
        print(
            "\nCreating regression plot..."
        )

        plot_regression_results(
            y_true=y_true_all,
            y_pred=y_pred_all,
            title=experiment_name,
            output_path=regression_plot_path,
        )

    # ========================================================
    # Return results
    # ========================================================

    results = {
        "experiment": experiment_name,

        "number_of_samples": (
            len(y_true_all)
        ),

        "final_mse": (
            final_mse
        ),

        "final_rmse": (
            final_rmse
        ),

        "final_mae": (
            final_mae
        ),

        "final_r2": (
            final_r2
        ),

        "mean_error": (
            mean_error
        ),

        "average_mse": (
            np.mean(
                mse_scores
            )
        ),

        "std_mse": (
            np.std(
                mse_scores
            )
        ),

        "average_rmse": (
            np.mean(
                rmse_scores
            )
        ),

        "std_rmse": (
            np.std(
                rmse_scores
            )
        ),
    }

    return (
        results,
        error_df,
        y_true_all,
        y_pred_all,
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
    Removes the n specimens with the highest
    LOOCV absolute prediction error.
    """

    if n_outliers < 0:
        raise ValueError(
            "n_outliers cannot be negative."
        )

    if n_outliers >= len(X):
        raise ValueError(
            "n_outliers must be smaller than "
            "the number of specimens."
        )

    # --------------------------------------------------------
    # Find highest-error samples
    # --------------------------------------------------------

    outlier_indices = (
        error_df
        .head(
            n_outliers
        )[
            "original_index"
        ]
        .astype(int)
        .values
    )

    print(
        "\n---------------- Removing Outliers ----------------"
    )

    print(
        f"Number of removed outliers: "
        f"{n_outliers}"
    )

    print(
        "Removed original indices:",
        outlier_indices,
    )

    # --------------------------------------------------------
    # Remove from X
    # --------------------------------------------------------

    X_cleaned = (
        X
        .drop(
            index=outlier_indices
        )
        .copy()
    )

    # --------------------------------------------------------
    # Remove from y
    # --------------------------------------------------------

    y_series = pd.Series(
        np.asarray(
            y,
            dtype=float,
        ),
        index=X.index,
    )

    y_cleaned = (
        y_series
        .drop(
            index=outlier_indices
        )
        .to_numpy(
            dtype=float
        )
    )

    print(
        "Original dataset size:",
        len(X),
    )

    print(
        "Cleaned dataset size:",
        len(X_cleaned),
    )

    return (
        X_cleaned,
        y_cleaned,
    )


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load and preprocess data
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

    # --------------------------------------------------------
    # Print settings
    # --------------------------------------------------------

    print(
        "\n================================================"
    )

    print(
        "MODEL SETTINGS"
    )

    print(
        "================================================"
    )

    print(
        "Selected features:"
    )

    for feature in X.columns:
        print(
            f"  - {feature}"
        )

    print(
        "\nNumber of specimens:",
        len(X),
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Iterations:",
        N_ITERATIONS,
    )

    print(
        "Number of outliers to remove:",
        N_OUTLIERS_TO_REMOVE,
    )

    # ========================================================
    # FIRST RUN: ORIGINAL DATASET
    # ========================================================

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
        error_df,
        y_true_original,
        y_pred_original,
    ) = leave_one_out_linear_regression(
        X=X,
        y=y,

        output_error_path=(
            OUTPUT_ERROR_PATH
        ),

        regression_plot_path=(
            OUTPUT_ORIGINAL_PLOT
        ),

        experiment_name=(
            "Normal LR - Original Dataset"
        ),
    )

    # ========================================================
    # OUTLIER REMOVAL
    # ========================================================

    (
        X_cleaned,
        y_cleaned,
    ) = remove_top_outliers_by_error(
        X=X,
        y=y,
        error_df=error_df,
        n_outliers=N_OUTLIERS_TO_REMOVE,
    )

    # ========================================================
    # SECOND RUN: CLEANED DATASET
    # ========================================================

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
        y_true_cleaned,
        y_pred_cleaned,
    ) = leave_one_out_linear_regression(
        X=X_cleaned,
        y=y_cleaned,

        output_error_path=(
            OUTPUT_ERROR_CLEANED_PATH
        ),

        regression_plot_path=(
            OUTPUT_CLEANED_PLOT
        ),

        experiment_name=(
            f"Normal LR - Dataset Without Top "
            f"{N_OUTLIERS_TO_REMOVE} High-Error Specimens"
        ),
    )

    # ========================================================
    # Performance comparison
    # ========================================================

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
        f"Original specimens: "
        f"{original_results['number_of_samples']}"
    )

    print(
        f"Cleaned specimens:  "
        f"{cleaned_results['number_of_samples']}"
    )

    print()

    print(
        f"Original MSE:  "
        f"{original_results['final_mse']:.4f}"
    )

    print(
        f"Cleaned MSE:   "
        f"{cleaned_results['final_mse']:.4f}"
    )

    print()

    print(
        f"Original RMSE: "
        f"{original_results['final_rmse']:.4f}"
    )

    print(
        f"Cleaned RMSE:  "
        f"{cleaned_results['final_rmse']:.4f}"
    )

    print()

    print(
        f"Original MAE:  "
        f"{original_results['final_mae']:.4f}"
    )

    print(
        f"Cleaned MAE:   "
        f"{cleaned_results['final_mae']:.4f}"
    )

    print()

    print(
        f"Original R²:   "
        f"{original_results['final_r2']:.4f}"
    )

    print(
        f"Cleaned R²:    "
        f"{cleaned_results['final_r2']:.4f}"
    )

    # ========================================================
    # Save performance comparison
    # ========================================================

    comparison_df = pd.DataFrame(
        [
            original_results,
            cleaned_results,
        ]
    )

    comparison_df.to_csv(
        OUTPUT_COMPARISON,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "\n================================================"
    )

    print(
        "FILES CREATED"
    )

    print(
        "================================================"
    )

    print(
        "\nOriginal error analysis:"
    )

    print(
        OUTPUT_ERROR_PATH.resolve()
    )

    print(
        "\nCleaned error analysis:"
    )

    print(
        OUTPUT_ERROR_CLEANED_PATH.resolve()
    )

    print(
        "\nOriginal regression plot:"
    )

    print(
        OUTPUT_ORIGINAL_PLOT.resolve()
    )

    print(
        "\nCleaned regression plot:"
    )

    print(
        OUTPUT_CLEANED_PLOT.resolve()
    )

    print(
        "\nPerformance comparison:"
    )

    print(
        OUTPUT_COMPARISON.resolve()
    )


# ============================================================
# Run program
# ============================================================

if __name__ == "__main__":
    main()