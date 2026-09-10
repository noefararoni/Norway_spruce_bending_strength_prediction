# physics_informed_LR.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import LEARNING_RATE, N_ITERATIONS
from data_preprocessing import split_X_y, clean_data, select_features


# ============================================================
# Physics Lambda
# ============================================================

try:
    from config import PHYSICS_LAMBDA
except ImportError:
    PHYSICS_LAMBDA = 0.1


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path("Data")

MERGED_DATASET_PATH = (
    DATA_DIR / "final_clean_merged_dataframe.csv"
)

OUTPUT_PLOT_PATH = (
    DATA_DIR / "physics_informed_lr_regression_plot.png"
)

OUTPUT_EXCEL_PATH = (
    DATA_DIR / "physics_informed_lr_predictions.xlsx"
)


# ============================================================
# Physics-Informed Linear Regression
# ============================================================

class PhysicsInformedLinearRegression:

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=5000,
        physics_lambda=0.1
    ):

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.physics_lambda = physics_lambda

        self.weights = None
        self.bias = None
        self.feature_names = None


    def fit(
        self,
        X,
        y,
        feature_names
    ):

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features
        )

        self.bias = 0.0

        self.feature_names = (
            feature_names
        )


        # ====================================================
        # Gradient Descent
        # ====================================================

        for _ in range(
            self.n_iterations
        ):

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            y_pred = (
                np.dot(
                    X,
                    self.weights
                )
                + self.bias
            )


            error = (
                y_pred
                - y
            )


            # ------------------------------------------------
            # Standard Linear Regression Gradient
            # ------------------------------------------------

            dw = (
                (2 / n_samples)
                * np.dot(
                    X.T,
                    error
                )
            )


            db = (
                (2 / n_samples)
                * np.sum(
                    error
                )
            )


            # ------------------------------------------------
            # Physics Gradient
            # ------------------------------------------------

            physics_grad = np.zeros_like(
                self.weights
            )


            for i, name in enumerate(
                feature_names
            ):

                name_lower = (
                    name.lower()
                )


                # ============================================
                # Positive Physical Influence
                #
                # Density ↑
                #     -> Bending strength ↑
                #
                # Dynamic modulus ↑
                #     -> Bending strength ↑
                # ============================================

                if (
                    "density" in name_lower
                    or "dynamic" in name_lower
                    or "edyn" in name_lower
                    or "modulus" in name_lower
                ):

                    if self.weights[i] < 0:

                        physics_grad[i] += (
                            2
                            * self.physics_lambda
                            * self.weights[i]
                        )


                # ============================================
                # Negative Physical Influence
                #
                # Growth-ring width ↑
                #     -> Bending strength ↓
                #
                # Span / size ↑
                #     -> Bending strength ↓
                # ============================================

                if (
                    "growth" in name_lower
                    or "grw" in name_lower
                    or "span" in name_lower
                    or "size" in name_lower
                ):

                    if self.weights[i] > 0:

                        physics_grad[i] += (
                            2
                            * self.physics_lambda
                            * self.weights[i]
                        )


            # ------------------------------------------------
            # Add Physics Gradient
            # ------------------------------------------------

            dw += physics_grad


            # ------------------------------------------------
            # Update Weights
            # ------------------------------------------------

            self.weights -= (
                self.learning_rate
                * dw
            )


            self.bias -= (
                self.learning_rate
                * db
            )


    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        X
    ):

        return (
            np.dot(
                X,
                self.weights
            )
            + self.bias
        )


    # ========================================================
    # Print Coefficients
    # ========================================================

    def print_coefficients(
        self
    ):

        print(
            "\n---------------- Physics-Informed Coefficients ----------------"
        )

        print(
            f"Bias: {self.bias:.6f}"
        )


        for name, weight in zip(
            self.feature_names,
            self.weights
        ):

            print(
                f"{name}: {weight:.6f}"
            )


# ============================================================
# Load Dataset
# ============================================================

def load_merged_dataset(
    path=MERGED_DATASET_PATH
):

    print(
        "\n---------------- Load Merged Dataset ----------------",
        flush=True
    )


    if not path.exists():

        raise FileNotFoundError(

            f"Could not find merged dataset at: {path}\n"
            "Please run data_preprocessing.py first "
            "to generate the file."

        )


    df = pd.read_csv(
        path,
        sep=";",
        decimal=","
    )


    print(
        "Loaded dataset:",
        path,
        flush=True
    )


    print(
        "Dataset shape:",
        df.shape,
        flush=True
    )


    print(
        "Columns:",
        list(df.columns),
        flush=True
    )


    return df


# ============================================================
# Plot Measured vs Predicted
# ============================================================

def plot_regression_results(
    y_true,
    y_pred,
    final_mse,
    average_rmse,
    final_r2,
    output_path=OUTPUT_PLOT_PATH
):

    """
    Creates the measured-vs-predicted plot using
    predictions already obtained during LOOCV.

    This function does NOT retrain the model.
    """


    y_true = np.asarray(
        y_true,
        dtype=float
    )


    y_pred = np.asarray(
        y_pred,
        dtype=float
    )


    # ========================================================
    # Determine Plot Limits
    # ========================================================

    minimum_value = min(
        np.min(y_true),
        np.min(y_pred)
    )


    maximum_value = max(
        np.max(y_true),
        np.max(y_pred)
    )


    value_range = (
        maximum_value
        - minimum_value
    )


    if value_range == 0:

        value_range = 1.0


    margin = (
        0.05
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


    # ========================================================
    # Create Figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    # --------------------------------------------------------
    # LOOCV Predictions
    # --------------------------------------------------------

    ax.scatter(
        y_true,
        y_pred,
        s=55,
        alpha=0.75,
        label="LOOCV predictions"
    )


    # --------------------------------------------------------
    # Perfect Prediction Line
    # --------------------------------------------------------

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction"
    )


    # ========================================================
    # Axis Labels
    # ========================================================

    ax.set_xlabel(
        "Measured Bending Strength [MPa]",
        fontsize=12
    )


    ax.set_ylabel(
        "Predicted Bending Strength [MPa]",
        fontsize=12
    )


    ax.set_title(
        "Physics-Informed LR - Original Dataset",
        fontsize=14
    )


    ax.set_xlim(
        plot_min,
        plot_max
    )


    ax.set_ylim(
        plot_min,
        plot_max
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend(
        loc="upper left"
    )


    # ========================================================
    # Performance Metrics Box
    # ========================================================

    metric_text = (

        f"MSE = {final_mse:.2f} MPa²\n"

        f"Average RMSE = {average_rmse:.2f} MPa\n"

        f"$R^2$ = {final_r2:.3f}"

    )


    ax.text(
        0.97,
        0.03,
        metric_text,

        transform=ax.transAxes,

        horizontalalignment="right",

        verticalalignment="bottom",

        fontsize=11,

        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9
        )
    )


    plt.tight_layout()


    # ========================================================
    # Save Figure
    # ========================================================

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    print(
        "\n---------------- Figure Created ----------------"
    )


    print(
        "Figure saved to:"
    )


    print(
        output_path.resolve()
    )


    # ========================================================
    # Display Figure
    # ========================================================

    plt.show()


    plt.close(
        fig
    )


# ============================================================
# Save LOOCV Predictions to Excel
# ============================================================

def save_predictions_to_excel(
    y_true,
    y_pred,
    size_association,
    original_indices,
    output_path=OUTPUT_EXCEL_PATH
):

    """
    Saves exactly the observations used in the final
    measured-vs-predicted figure.

    One row = one LOOCV prediction.
    """


    y_true = np.asarray(
        y_true,
        dtype=float
    )


    y_pred = np.asarray(
        y_pred,
        dtype=float
    )


    # ========================================================
    # Prediction Errors
    # ========================================================

    errors = (
        y_pred
        - y_true
    )


    absolute_errors = np.abs(
        errors
    )


    squared_errors = (
        errors ** 2
    )


    # ========================================================
    # Create Results Table
    # ========================================================

    results_df = pd.DataFrame({

        "Specimen":
            np.arange(
                1,
                len(y_true) + 1
            ),

        "Original_Index":
            original_indices,

        "Size":
            size_association,

        "Measured_Bending_Strength_MPa":
            y_true,

        "Predicted_Bending_Strength_MPa":
            y_pred,

        "Error_MPa":
            errors,

        "Absolute_Error_MPa":
            absolute_errors,

        "Squared_Error_MPa2":
            squared_errors,

    })


    # ========================================================
    # Save Excel
    # ========================================================

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results_df.to_excel(
        output_path,
        index=False
    )


    print(
        "\n---------------- Excel File Created ----------------"
    )


    print(
        "Predictions saved to:"
    )


    print(
        output_path.resolve()
    )


    print(
        "\nFirst rows of prediction table:"
    )


    print(
        results_df.head()
    )


    print(
        "\nSize distribution in Excel file:"
    )


    print(
        results_df["Size"].value_counts(
            dropna=False
        )
    )


    return results_df


# ============================================================
# Leave-One-Out Cross-Validation
# ============================================================

def leave_one_out_physics_informed_lr(
    X,
    y,
    size_association,
    original_indices
):

    print(
        "\n---------------- Physics-Informed Linear Regression From Scratch ----------------",
        flush=True
    )


    print(
        "Using Leave-One-Out Cross-Validation",
        flush=True
    )


    print(
        f"Learning rate: {LEARNING_RATE}",
        flush=True
    )


    print(
        f"Iterations: {N_ITERATIONS}",
        flush=True
    )


    print(
        f"Physics lambda: {PHYSICS_LAMBDA}",
        flush=True
    )


    y = np.array(
        y
    )


    # ========================================================
    # Reset Metadata Index
    #
    # This ensures that the LOOCV position corresponds exactly
    # to the size and original specimen index.
    # ========================================================

    size_association = (
        pd.Series(size_association)
        .reset_index(drop=True)
    )


    original_indices = (
        pd.Series(original_indices)
        .reset_index(drop=True)
    )


    feature_names = list(
        X.columns
    )


    loo = LeaveOneOut()


    y_true_all = []

    y_pred_all = []

    size_all = []

    original_index_all = []

    mse_scores = []

    rmse_scores = []


    final_model = None


    # ========================================================
    # LOOCV
    # ========================================================

    for fold, (
        train_index,
        test_index
    ) in enumerate(
        loo.split(X),
        start=1
    ):


        # ----------------------------------------------------
        # Training / Test Data
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
        # Fitted only on training data
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
        # Physics-Informed Model
        # ----------------------------------------------------

        model = PhysicsInformedLinearRegression(

            learning_rate=LEARNING_RATE,

            n_iterations=N_ITERATIONS,

            physics_lambda=PHYSICS_LAMBDA,

        )


        model.fit(

            X_train_scaled,

            y_train,

            feature_names

        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test_scaled
        )


        # ----------------------------------------------------
        # Fold Metrics
        # ----------------------------------------------------

        mse = mean_squared_error(
            y_test,
            y_pred
        )


        rmse = np.sqrt(
            mse
        )


        mse_scores.append(
            mse
        )


        rmse_scores.append(
            rmse
        )


        # ----------------------------------------------------
        # Store Prediction
        # ----------------------------------------------------

        y_true_all.append(
            y_test[0]
        )


        y_pred_all.append(
            y_pred[0]
        )


        # ----------------------------------------------------
        # Store Corresponding Size
        # ----------------------------------------------------

        test_position = (
            test_index[0]
        )


        size_all.append(
            size_association.iloc[
                test_position
            ]
        )


        # ----------------------------------------------------
        # Store Original Dataset Index
        # ----------------------------------------------------

        original_index_all.append(
            original_indices.iloc[
                test_position
            ]
        )


        final_model = model


        print(

            f"Fold {fold}: "

            f"Size={size_all[-1]}, "

            f"True={y_test[0]:.4f}, "

            f"Predicted={y_pred[0]:.4f}, "

            f"RMSE={rmse:.4f}",

            flush=True

        )


    # ========================================================
    # Final LOOCV Metrics
    # ========================================================

    final_mse = mean_squared_error(
        y_true_all,
        y_pred_all
    )


    final_rmse = np.sqrt(
        final_mse
    )


    final_r2 = r2_score(
        y_true_all,
        y_pred_all
    )


    # ========================================================
    # Average Fold RMSE
    # ========================================================

    average_rmse = np.mean(
        rmse_scores
    )


    std_rmse = np.std(
        rmse_scores
    )


    # ========================================================
    # Print Results
    # ========================================================

    print(
        "\n---------------- Final Leave-One-Out Results ----------------",
        flush=True
    )


    print(

        f"Average MSE:   "

        f"{np.mean(mse_scores):.4f} "

        f"± {np.std(mse_scores):.4f}",

        flush=True

    )


    print(

        f"Average RMSE:  "

        f"{average_rmse:.4f} "

        f"± {std_rmse:.4f}",

        flush=True

    )


    print(

        f"Final MSE:     "

        f"{final_mse:.4f}",

        flush=True

    )


    print(

        f"Final RMSE:    "

        f"{final_rmse:.4f}",

        flush=True

    )


    print(

        f"Final R²:      "

        f"{final_r2:.4f}",

        flush=True

    )


    # ========================================================
    # Print Physics-Informed Coefficients
    # ========================================================

    if final_model is not None:

        final_model.print_coefficients()


    # ========================================================
    # Create Regression Figure
    # ========================================================

    plot_regression_results(

        y_true=y_true_all,

        y_pred=y_pred_all,

        final_mse=final_mse,

        average_rmse=average_rmse,

        final_r2=final_r2,

        output_path=OUTPUT_PLOT_PATH

    )


    # ========================================================
    # Save EXACTLY the Same Points to Excel
    # ========================================================

    results_df = save_predictions_to_excel(

        y_true=y_true_all,

        y_pred=y_pred_all,

        size_association=size_all,

        original_indices=original_index_all,

        output_path=OUTPUT_EXCEL_PATH

    )


    # ========================================================
    # Return Results
    # ========================================================

    return (
        mse_scores,
        rmse_scores,
        final_r2,
        results_df
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load Original Dataset
    # ========================================================

    df_all = load_merged_dataset()


    # ========================================================
    # Extract Size Association
    #
    # This is done BEFORE select_features() because
    # source_file may be removed during feature selection.
    #
    # Expected examples:
    #
    # Size_A
    # Size_B
    # Size_C
    #
    # Result:
    #
    # A
    # B
    # C
    # ========================================================

    if "source_file" not in df_all.columns:

        raise KeyError(

            "The column 'source_file' was not found in the dataset.\n"
            "This column is required to determine whether each "
            "specimen belongs to Size A, B, or C."

        )


    size_association = (

        df_all["source_file"]

        .astype(str)

        .str.extract(
            r"Size[_\-\s]*([ABC])",
            expand=False
        )

        .str.upper()

    )


    # ========================================================
    # Keep Original Dataset Index
    # ========================================================

    original_indices = (
        df_all.index.copy()
    )


    print(
        "\n---------------- Size Association ----------------"
    )


    print(
        size_association.value_counts(
            dropna=False
        )
    )


    # ========================================================
    # Existing Preprocessing
    # ========================================================

    df_selected = select_features(
        df_all
    )


    df_clean = clean_data(
        df_selected
    )


    # ========================================================
    # IMPORTANT:
    # Keep metadata aligned with rows remaining after cleaning
    # ========================================================

    clean_indices = (
        df_clean.index
    )


    size_association = (

        size_association
        .loc[
            clean_indices
        ]
        .reset_index(
            drop=True
        )

    )


    original_indices = (

        pd.Series(
            original_indices,
            index=df_all.index
        )

        .loc[
            clean_indices
        ]

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # Split X and y
    # ========================================================

    X, y = split_X_y(
        df_clean
    )


    # ========================================================
    # Reset X Index
    #
    # Only changes DataFrame labels.
    # It does NOT change the data or model.
    # ========================================================

    X = X.reset_index(
        drop=True
    )


    if isinstance(
        y,
        pd.Series
    ):

        y = y.reset_index(
            drop=True
        )


    # ========================================================
    # Safety Check
    # ========================================================

    if not (
        len(X)
        == len(y)
        == len(size_association)
        == len(original_indices)
    ):

        raise ValueError(

            "Data alignment problem detected.\n"

            f"X rows: {len(X)}\n"
            f"y rows: {len(y)}\n"
            f"Size rows: {len(size_association)}\n"
            f"Index rows: {len(original_indices)}"

        )


    # ========================================================
    # Run Physics-Informed LR
    # ========================================================

    leave_one_out_physics_informed_lr(

        X,

        y,

        size_association,

        original_indices

    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()