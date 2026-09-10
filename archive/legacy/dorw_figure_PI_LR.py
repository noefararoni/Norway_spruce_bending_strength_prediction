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

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.feature_names = feature_names


        for _ in range(self.n_iterations):

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            y_pred = (
                np.dot(X, self.weights)
                + self.bias
            )

            error = (
                y_pred - y
            )


            # ------------------------------------------------
            # Normal Linear Regression Gradient
            # ------------------------------------------------

            dw = (
                (2 / n_samples)
                * np.dot(X.T, error)
            )

            db = (
                (2 / n_samples)
                * np.sum(error)
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
                # Positive physical influence
                #
                # Density should increase bending strength
                # Dynamic modulus should increase bending strength
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
                # Negative physical influence
                #
                # Growth-ring width should decrease
                # bending strength
                #
                # Span / size should decrease bending strength
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
            # Add Physics Constraint to Gradient
            # ------------------------------------------------

            dw += physics_grad


            # ------------------------------------------------
            # Update Parameters
            # ------------------------------------------------

            self.weights -= (
                self.learning_rate
                * dw
            )

            self.bias -= (
                self.learning_rate
                * db
            )


    def predict(
        self,
        X
    ):

        return (
            np.dot(X, self.weights)
            + self.bias
        )


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
    Plot the already obtained LOOCV results.

    IMPORTANT:
    This function does NOT train the model.
    It only visualizes the predictions already obtained
    during LOOCV.
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
    #
    # y_predicted = y_measured
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

        f"RMSE = {average_rmse:.2f} MPa\n"

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
# Leave-One-Out Cross-Validation
# ============================================================

def leave_one_out_physics_informed_lr(
    X,
    y
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


    feature_names = list(
        X.columns
    )


    loo = LeaveOneOut()


    y_true_all = []
    y_pred_all = []

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
        # Training and Test Samples
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
        # Fit ONLY on training data
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


        y_true_all.append(
            y_test[0]
        )


        y_pred_all.append(
            y_pred[0]
        )


        final_model = model


        print(

            f"Fold {fold}: "

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
    # Average RMSE
    #
    # This is the value you requested for the figure.
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
    # Create Figure
    #
    # IMPORTANT:
    # Training has already finished at this point.
    #
    # The plotting function ONLY uses:
    #
    #   y_true_all
    #   y_pred_all
    #   final_mse
    #   average_rmse
    #   final_r2
    #
    # Therefore, it cannot change your model results.
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
    # Original Return Values
    # ========================================================

    return (
        mse_scores,
        rmse_scores,
        final_r2
    )


# ============================================================
# Main
# ============================================================

def main():

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


    leave_one_out_physics_informed_lr(
        X,
        y
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()