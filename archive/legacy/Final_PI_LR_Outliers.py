# normal_and_physics_informed_LR_with_generalization.py

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

TRAINING_DATASET_PATH = (
    DATA_DIR / "final_clean_merged_dataframe.csv"
)

GENERALIZATION_DATASET_PATH = (
    DATA_DIR / "generalisation_validation.csv"
)

TARGET_COLUMN = "Bending strength"

N_OUTLIERS_TO_REMOVE = 5


# Select which model determines the removed specimens.
#
# "normal":
#     Removes the five highest-error samples identified by
#     the normal LR LOOCV.
#
# "physics":
#     Removes the five highest-error samples identified by
#     the physics-informed LR LOOCV.
#
# Using "normal" is useful when reproducing the outlier
# experiment described for multiple linear regression.
OUTLIER_SELECTION_MODEL = "normal"


# ============================================================
# Physics constraints
# ============================================================

# Bending strength is expected to increase with these features.
POSITIVE_FEATURES = [
    "Density",
    "Dynamic E",
]

# Bending strength is expected to decrease with these features.
NEGATIVE_FEATURES = [
    "Growth-ring width",
    "Height",
    "Test span",
    "Width",
]


# ============================================================
# Output paths
# ============================================================

OUTPUT_NORMAL_LOOCV_ORIGINAL = (
    DATA_DIR / "normal_lr_loocv_original.csv"
)

OUTPUT_PHYSICS_LOOCV_ORIGINAL = (
    DATA_DIR / "physics_lr_loocv_original.csv"
)

OUTPUT_NORMAL_LOOCV_CLEANED = (
    DATA_DIR / "normal_lr_loocv_cleaned.csv"
)

OUTPUT_PHYSICS_LOOCV_CLEANED = (
    DATA_DIR / "physics_lr_loocv_cleaned.csv"
)

OUTPUT_REMOVED_OUTLIERS = (
    DATA_DIR / "removed_high_error_specimens.csv"
)

OUTPUT_GENERALIZATION_NORMAL_ORIGINAL = (
    DATA_DIR / "generalization_normal_original.csv"
)

OUTPUT_GENERALIZATION_PHYSICS_ORIGINAL = (
    DATA_DIR / "generalization_physics_original.csv"
)

OUTPUT_GENERALIZATION_NORMAL_CLEANED = (
    DATA_DIR / "generalization_normal_cleaned.csv"
)

OUTPUT_GENERALIZATION_PHYSICS_CLEANED = (
    DATA_DIR / "generalization_physics_cleaned.csv"
)

OUTPUT_FINAL_COMPARISON = (
    DATA_DIR / "complete_model_comparison.csv"
)

OUTPUT_FINAL_COEFFICIENTS = (
    DATA_DIR / "final_model_coefficients.csv"
)

OUTPUT_FOLD_COEFFICIENTS = (
    DATA_DIR / "loocv_fold_coefficients.csv"
)


# ============================================================
# Normal Linear Regression
# ============================================================

class LinearRegressionScratch:
    """
    Normal multiple linear regression optimized using
    gradient descent.

    Prediction:
        y_pred = X @ weights + bias

    Loss:
        mean((y_pred - y_true)^2)
    """

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

        self.loss_history = []
        self.feature_names = None

    def fit(
        self,
        X,
        y,
        feature_names=None,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number "
                "of specimens."
            )

        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [
                f"feature_{index}"
                for index in range(n_features)
            ]

        self.feature_names = list(
            feature_names
        )

        if len(self.feature_names) != n_features:
            raise ValueError(
                "The number of feature names must match "
                "the number of columns in X."
            )

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.bias = 0.0
        self.loss_history = []

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

            loss = np.mean(
                error ** 2
            )

            weight_gradient = (
                2.0 / n_samples
            ) * np.dot(
                X.T,
                error,
            )

            bias_gradient = (
                2.0 / n_samples
            ) * np.sum(
                error
            )

            self.weights -= (
                self.learning_rate
                * weight_gradient
            )

            self.bias -= (
                self.learning_rate
                * bias_gradient
            )

            self.loss_history.append(
                float(loss)
            )

            if not (
                np.all(
                    np.isfinite(self.weights)
                )
                and np.isfinite(self.bias)
            ):
                raise FloatingPointError(
                    "Normal LR training became unstable. "
                    "Reduce LEARNING_RATE."
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
# Physics-Informed Linear Regression
# ============================================================

class PhysicsInformedLinearRegression:
    """
    Multiple linear regression with soft physical constraints.

    Prediction:
        y_pred = X @ weights + bias

    Total loss:
        total_loss =
            data_loss
            + physics_lambda * physics_penalty

    Data loss:
        data_loss =
            mean((y_pred - y_true)^2)

    Physics constraints:
        beta_Density >= 0
        beta_Dynamic_E >= 0

        beta_Growth_ring_width <= 0
        beta_Height <= 0
        beta_Test_span <= 0
        beta_Width <= 0

    The constraints are soft. A coefficient contributes to the
    physics penalty only when its sign violates the expected
    physical trend.
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=5000,
        physics_lambda=0.001,
        positive_features=None,
        negative_features=None,
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

        self.positive_features = list(
            positive_features or []
        )

        self.negative_features = list(
            negative_features or []
        )

        self.weights = None
        self.bias = None
        self.feature_names = None

        self.loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []

    def _validate_constraints(
        self,
    ):
        duplicated_features = set(
            self.positive_features
        ).intersection(
            self.negative_features
        )

        if duplicated_features:
            raise ValueError(
                "The following features appear in both positive "
                f"and negative constraints: {duplicated_features}"
            )

        constrained_features = (
            self.positive_features
            + self.negative_features
        )

        missing_features = [
            feature
            for feature in constrained_features
            if feature not in self.feature_names
        ]

        if missing_features:
            raise ValueError(
                "Physics-constrained features are missing from X: "
                f"{missing_features}\n"
                f"Available features: {self.feature_names}"
            )

    def _physics_penalty_and_gradient(
        self,
    ):
        penalty = 0.0

        gradient = np.zeros_like(
            self.weights
        )

        # Expected positive coefficients.
        for feature in self.positive_features:
            feature_index = self.feature_names.index(
                feature
            )

            coefficient = self.weights[
                feature_index
            ]

            if coefficient < 0:
                penalty += coefficient ** 2

                gradient[
                    feature_index
                ] += 2.0 * coefficient

        # Expected negative coefficients.
        for feature in self.negative_features:
            feature_index = self.feature_names.index(
                feature
            )

            coefficient = self.weights[
                feature_index
            ]

            if coefficient > 0:
                penalty += coefficient ** 2

                gradient[
                    feature_index
                ] += 2.0 * coefficient

        n_constraints = (
            len(self.positive_features)
            + len(self.negative_features)
        )

        if n_constraints > 0:
            penalty /= n_constraints
            gradient /= n_constraints

        return (
            float(penalty),
            gradient,
        )

    def fit(
        self,
        X,
        y,
        feature_names,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number "
                "of specimens."
            )

        n_samples, n_features = X.shape

        self.feature_names = list(
            feature_names
        )

        if len(self.feature_names) != n_features:
            raise ValueError(
                "The number of feature names must match "
                "the number of columns in X."
            )

        self._validate_constraints()

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.bias = 0.0

        self.loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []

        for _ in range(
            self.n_iterations
        ):
            # ------------------------------------------------
            # Data prediction
            # ------------------------------------------------

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

            data_loss = np.mean(
                error ** 2
            )

            data_weight_gradient = (
                2.0 / n_samples
            ) * np.dot(
                X.T,
                error,
            )

            bias_gradient = (
                2.0 / n_samples
            ) * np.sum(
                error
            )

            # ------------------------------------------------
            # Physics penalty
            # ------------------------------------------------

            (
                physics_penalty,
                physics_gradient,
            ) = self._physics_penalty_and_gradient()

            total_loss = (
                data_loss
                + self.physics_lambda
                * physics_penalty
            )

            total_weight_gradient = (
                data_weight_gradient
                + self.physics_lambda
                * physics_gradient
            )

            # ------------------------------------------------
            # Parameter update
            # ------------------------------------------------

            self.weights -= (
                self.learning_rate
                * total_weight_gradient
            )

            self.bias -= (
                self.learning_rate
                * bias_gradient
            )

            self.data_loss_history.append(
                float(data_loss)
            )

            self.physics_loss_history.append(
                float(physics_penalty)
            )

            self.loss_history.append(
                float(total_loss)
            )

            if not (
                np.all(
                    np.isfinite(self.weights)
                )
                and np.isfinite(self.bias)
            ):
                raise FloatingPointError(
                    "Physics-informed training became unstable. "
                    "Reduce LEARNING_RATE or PHYSICS_LAMBDA."
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

    def get_constraint_status(
        self,
    ):
        status = {}

        for feature in self.positive_features:
            feature_index = self.feature_names.index(
                feature
            )

            coefficient = float(
                self.weights[
                    feature_index
                ]
            )

            status[feature] = {
                "expected_sign": "positive",
                "coefficient": coefficient,
                "violated": coefficient < 0,
            }

        for feature in self.negative_features:
            feature_index = self.feature_names.index(
                feature
            )

            coefficient = float(
                self.weights[
                    feature_index
                ]
            )

            status[feature] = {
                "expected_sign": "negative",
                "coefficient": coefficient,
                "violated": coefficient > 0,
            }

        return status

    def count_constraint_violations(
        self,
    ):
        status = self.get_constraint_status()

        return sum(
            item["violated"]
            for item in status.values()
        )


# ============================================================
# Dataset reading
# ============================================================

def read_dataset(
    path,
):
    """
    Reads a semicolon- or tab-separated dataset using
    comma decimal notation.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    read_attempts = [
        {
            "sep": ";",
            "decimal": ",",
        },
        {
            "sep": "\t",
            "decimal": ",",
        },
        {
            "sep": ",",
            "decimal": ".",
        },
    ]

    last_error = None

    for options in read_attempts:
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
                f"Column '{column}' is missing from "
                f"{dataset_name}."
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
            f"Invalid or missing numeric values in "
            f"{dataset_name}:\n{invalid_counts}"
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

    df_all = read_dataset(
        path
    )

    print(
        "Training dataset:",
        path,
    )

    print(
        "Raw shape:",
        df_all.shape,
    )

    print(
        "Raw columns:",
        list(df_all.columns),
    )

    df_selected = select_features(
        df_all
    )

    df_clean = clean_data(
        df_selected
    )

    X, y = split_X_y(
        df_clean
    )

    X = (
        X
        .copy()
        .reset_index(drop=True)
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    df_clean = (
        df_clean
        .copy()
        .reset_index(drop=True)
    )

    if len(X) != len(y):
        raise ValueError(
            "Prepared X and y have different lengths."
        )

    print(
        "Prepared training shape:",
        X.shape,
    )

    print(
        "Features:",
        list(X.columns),
    )

    return (
        df_clean,
        X,
        y,
    )


# ============================================================
# Load generalization dataset
# ============================================================

def load_generalization_dataset(
    training_features,
    path=GENERALIZATION_DATASET_PATH,
):
    print(
        "\n---------------- Load Generalization Dataset ----------------"
    )

    df = read_dataset(
        path
    )

    required_columns = (
        list(training_features)
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The generalization dataset is missing columns: "
            f"{missing_columns}"
        )

    df = convert_columns_to_numeric(
        df=df,
        columns=required_columns,
        dataset_name="generalization dataset",
    )

    df = (
        df
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    X_generalization = (
        df[
            list(training_features)
        ]
        .copy()
    )

    y_generalization = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(dtype=float)
    )

    print(
        "Generalization dataset:",
        path,
    )

    print(
        "Generalization X shape:",
        X_generalization.shape,
    )

    print(
        "Generalization y shape:",
        y_generalization.shape,
    )

    print(
        "Generalization features:",
        list(X_generalization.columns),
    )

    return (
        df,
        X_generalization,
        y_generalization,
    )


# ============================================================
# Model creation
# ============================================================

def create_model(
    model_type,
):
    if model_type == "normal":
        return LinearRegressionScratch(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
        )

    if model_type == "physics":
        return PhysicsInformedLinearRegression(
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            physics_lambda=PHYSICS_LAMBDA,
            positive_features=POSITIVE_FEATURES,
            negative_features=NEGATIVE_FEATURES,
        )

    raise ValueError(
        "model_type must be 'normal' or 'physics'."
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    dataset_name,
    model_type,
    evaluation_type,
    training_version,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have equal lengths."
        )

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
        "Model": model_type,
        "Evaluation": evaluation_type,
        "Training data": training_version,
        "Number of specimens": len(y_true),
        "Physics lambda": (
            PHYSICS_LAMBDA
            if model_type == "physics"
            else 0.0
        ),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE (%)": mape,
        "Mean error": mean_error,
    }


# ============================================================
# Prediction table
# ============================================================

def create_prediction_table(
    base_dataframe,
    y_true,
    y_pred,
):
    results = (
        base_dataframe
        .copy()
        .reset_index(drop=True)
    )

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

    results[
        "Measured bending strength"
    ] = y_true

    results[
        "Predicted bending strength"
    ] = y_pred

    results[
        "Error"
    ] = residuals

    results[
        "Absolute error"
    ] = np.abs(
        residuals
    )

    results[
        "Squared error"
    ] = residuals ** 2

    results[
        "Absolute percentage error (%)"
    ] = (
        np.abs(residuals)
        / np.maximum(
            np.abs(y_true),
            1e-12,
        )
    ) * 100.0

    return results


# ============================================================
# LOOCV
# ============================================================

def leave_one_out_evaluation(
    X,
    y,
    model_type,
    experiment_name,
    training_version,
    output_path,
):
    print(
        f"\n---------------- {experiment_name} ----------------"
    )

    print(
        "Using Leave-One-Out Cross-Validation"
    )

    feature_names = list(
        X.columns
    )

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []

    error_records = []
    coefficient_records = []

    physics_penalties = []
    violation_counts = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        loo.split(X),
        start=1,
    ):
        X_train = (
            X.iloc[
                train_index
            ]
            .copy()
        )

        X_test = (
            X.iloc[
                test_index
            ]
            .copy()
        )

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        # Fit scaler only on the training fold.
        scaler = StandardScaler()

        X_train_scaled = (
            scaler
            .fit_transform(X_train)
        )

        X_test_scaled = (
            scaler
            .transform(X_test)
        )

        model = create_model(
            model_type
        )

        model.fit(
            X=X_train_scaled,
            y=y_train,
            feature_names=feature_names,
        )

        y_pred = model.predict(
            X_test_scaled
        )

        true_value = float(
            y_test[0]
        )

        predicted_value = float(
            y_pred[0]
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

        y_true_all.append(
            true_value
        )

        y_pred_all.append(
            predicted_value
        )

        current_position = int(
            test_index[0]
        )

        original_index = int(
            X.index[
                current_position
            ]
        )

        if model_type == "physics":
            final_physics_penalty = float(
                model.physics_loss_history[-1]
            )

            violation_count = (
                model.count_constraint_violations()
            )

            constraint_status = (
                model.get_constraint_status()
            )

            physics_penalties.append(
                final_physics_penalty
            )

            violation_counts.append(
                violation_count
            )

        else:
            final_physics_penalty = np.nan
            violation_count = np.nan
            constraint_status = {}

        error_record = {
            "fold": fold,
            "original_index": original_index,
            "current_position": current_position,
            "true_value": true_value,
            "predicted_value": predicted_value,
            "error": error,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "rmse": absolute_error,
            "final_physics_penalty": (
                final_physics_penalty
            ),
            "constraint_violations": (
                violation_count
            ),
            **X.iloc[
                current_position
            ].to_dict(),
        }

        for feature, status in constraint_status.items():
            safe_feature = (
                feature
                .replace(" ", "_")
                .replace("-", "_")
            )

            error_record[
                f"{safe_feature}_constraint_violated"
            ] = status["violated"]

        error_records.append(
            error_record
        )

        coefficient_record = {
            "fold": fold,
            "model": model_type,
            "training_data": training_version,
            "bias": model.bias,
        }

        for feature, coefficient in zip(
            feature_names,
            model.weights,
        ):
            coefficient_record[
                feature
            ] = coefficient

        coefficient_records.append(
            coefficient_record
        )

        print(
            f"Fold {fold}: "
            f"True={true_value:.4f}, "
            f"Predicted={predicted_value:.4f}, "
            f"Absolute error={absolute_error:.4f}",
            flush=True,
        )

    metrics = calculate_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        dataset_name=experiment_name,
        model_type=model_type,
        evaluation_type="LOOCV",
        training_version=training_version,
    )

    squared_errors = (
        np.asarray(y_true_all)
        - np.asarray(y_pred_all)
    ) ** 2

    absolute_errors = np.abs(
        np.asarray(y_true_all)
        - np.asarray(y_pred_all)
    )

    metrics[
        "Fold squared-error SD"
    ] = np.std(
        squared_errors
    )

    metrics[
        "Fold absolute-error SD"
    ] = np.std(
        absolute_errors
    )

    metrics[
        "Average physics penalty"
    ] = (
        np.mean(physics_penalties)
        if physics_penalties
        else np.nan
    )

    metrics[
        "Average constraint violations"
    ] = (
        np.mean(violation_counts)
        if violation_counts
        else np.nan
    )

    metrics[
        "Folds with violations"
    ] = (
        int(
            np.sum(
                np.asarray(
                    violation_counts
                ) > 0
            )
        )
        if violation_counts
        else np.nan
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

    coefficients_df = pd.DataFrame(
        coefficient_records
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
        "\nTop 10 highest-error specimens:"
    )

    print(
        error_df
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nSaved LOOCV results to:",
        output_path,
    )

    return (
        metrics,
        error_df,
        coefficients_df,
    )


# ============================================================
# Outlier removal
# ============================================================

def remove_top_outliers(
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
            "n_outliers must be smaller than the "
            "training dataset size."
        )

    if n_outliers == 0:
        return (
            df_training.copy(),
            X.copy(),
            np.asarray(y).copy(),
            error_df.head(0).copy(),
        )

    removed_specimens = (
        error_df
        .head(n_outliers)
        .copy()
    )

    removed_indices = (
        removed_specimens[
            "original_index"
        ]
        .astype(int)
        .tolist()
    )

    print(
        "\n---------------- Removing High-Error Specimens ----------------"
    )

    print(
        "Selection model:",
        OUTLIER_SELECTION_MODEL,
    )

    print(
        "Number removed:",
        n_outliers,
    )

    print(
        "Removed indices:",
        removed_indices,
    )

    X_cleaned = (
        X
        .drop(
            index=removed_indices
        )
        .copy()
    )

    df_cleaned = (
        df_training
        .drop(
            index=removed_indices
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
            index=removed_indices
        )
        .to_numpy(dtype=float)
    )

    X_cleaned = (
        X_cleaned
        .reset_index(drop=True)
    )

    df_cleaned = (
        df_cleaned
        .reset_index(drop=True)
    )

    OUTPUT_REMOVED_OUTLIERS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    removed_specimens.to_csv(
        OUTPUT_REMOVED_OUTLIERS,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "Original training size:",
        len(X),
    )

    print(
        "Cleaned training size:",
        len(X_cleaned),
    )

    print(
        "Saved removed specimens to:",
        OUTPUT_REMOVED_OUTLIERS,
    )

    return (
        df_cleaned,
        X_cleaned,
        y_cleaned,
        removed_specimens,
    )


# ============================================================
# Train final model
# ============================================================

def train_final_model(
    X_train,
    y_train,
    model_type,
):
    feature_names = list(
        X_train.columns
    )

    # Scaler is fitted only on the corresponding training set.
    scaler = StandardScaler()

    X_train_scaled = (
        scaler
        .fit_transform(X_train)
    )

    model = create_model(
        model_type
    )

    model.fit(
        X=X_train_scaled,
        y=y_train,
        feature_names=feature_names,
    )

    return (
        model,
        scaler,
    )


# ============================================================
# External generalization
# ============================================================

def evaluate_generalization(
    model,
    scaler,
    df_generalization,
    X_generalization,
    y_generalization,
    model_type,
    training_version,
    output_path,
):
    # Important:
    # transform() uses the mean and standard deviation learned
    # only from the corresponding training dataset.
    X_generalization_scaled = (
        scaler
        .transform(
            X_generalization
        )
    )

    y_pred = model.predict(
        X_generalization_scaled
    )

    dataset_name = (
        f"External Generalization - "
        f"{model_type.title()} LR - "
        f"{training_version.title()} Training Data"
    )

    metrics = calculate_metrics(
        y_true=y_generalization,
        y_pred=y_pred,
        dataset_name=dataset_name,
        model_type=model_type,
        evaluation_type="External generalization",
        training_version=training_version,
    )

    if model_type == "physics":
        metrics[
            "Average physics penalty"
        ] = float(
            model.physics_loss_history[-1]
        )

        metrics[
            "Average constraint violations"
        ] = (
            model.count_constraint_violations()
        )

        metrics[
            "Folds with violations"
        ] = np.nan

    else:
        metrics[
            "Average physics penalty"
        ] = np.nan

        metrics[
            "Average constraint violations"
        ] = np.nan

        metrics[
            "Folds with violations"
        ] = np.nan

    metrics[
        "Fold squared-error SD"
    ] = np.nan

    metrics[
        "Fold absolute-error SD"
    ] = np.nan

    prediction_df = create_prediction_table(
        base_dataframe=df_generalization,
        y_true=y_generalization,
        y_pred=y_pred,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        output_path,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "Saved external predictions to:",
        output_path,
    )

    return (
        metrics,
        prediction_df,
    )


# ============================================================
# Store final coefficients
# ============================================================

def create_final_coefficient_table(
    model,
    scaler,
    model_type,
    training_version,
    feature_names,
):
    """
    Reports coefficients in standardized space and converts
    them back to original feature units.

    Standardized model:
        y = sum(beta_scaled_j * z_j) + bias_scaled

    Original-space coefficient:
        beta_original_j =
            beta_scaled_j / scaler.scale_[j]
    """

    standardized_coefficients = np.asarray(
        model.weights,
        dtype=float,
    )

    original_coefficients = (
        standardized_coefficients
        / scaler.scale_
    )

    original_bias = (
        model.bias
        - np.sum(
            standardized_coefficients
            * scaler.mean_
            / scaler.scale_
        )
    )

    records = []

    constraint_status = {}

    if model_type == "physics":
        constraint_status = (
            model.get_constraint_status()
        )

    for index, feature in enumerate(
        feature_names
    ):
        if feature in constraint_status:
            expected_sign = constraint_status[
                feature
            ]["expected_sign"]

            violated = constraint_status[
                feature
            ]["violated"]

        else:
            expected_sign = "none"
            violated = False

        records.append(
            {
                "Model": model_type,
                "Training data": training_version,
                "Feature": feature,
                "Standardized coefficient": (
                    standardized_coefficients[index]
                ),
                "Original-unit coefficient": (
                    original_coefficients[index]
                ),
                "Expected sign": expected_sign,
                "Constraint violated": violated,
                "Standardized bias": model.bias,
                "Original-unit bias": original_bias,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Main
# ============================================================

def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Iterations:",
        N_ITERATIONS,
    )

    print(
        "Physics lambda:",
        PHYSICS_LAMBDA,
    )

    print(
        "Outliers to remove:",
        N_OUTLIERS_TO_REMOVE,
    )

    print(
        "Outlier-selection model:",
        OUTLIER_SELECTION_MODEL,
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        df_training,
        X,
        y,
    ) = load_training_dataset()

    (
        df_generalization,
        X_generalization,
        y_generalization,
    ) = load_generalization_dataset(
        training_features=list(
            X.columns
        )
    )

    # Validate physics feature names.
    required_physics_features = (
        POSITIVE_FEATURES
        + NEGATIVE_FEATURES
    )

    missing_physics_features = [
        feature
        for feature in required_physics_features
        if feature not in X.columns
    ]

    if missing_physics_features:
        raise ValueError(
            "The following physics features are missing "
            f"from the prepared dataset: "
            f"{missing_physics_features}"
        )

    # ========================================================
    # ORIGINAL DATASET: LOOCV
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "ORIGINAL DATASET: NORMAL LR LOOCV"
    )

    print(
        "================================================"
    )

    (
        normal_original_loocv,
        normal_original_errors,
        normal_original_fold_coefficients,
    ) = leave_one_out_evaluation(
        X=X,
        y=y,
        model_type="normal",
        experiment_name=(
            "Normal LR - Original Dataset - LOOCV"
        ),
        training_version="original",
        output_path=OUTPUT_NORMAL_LOOCV_ORIGINAL,
    )

    print(
        "\n\n================================================"
    )

    print(
        "ORIGINAL DATASET: PHYSICS-INFORMED LR LOOCV"
    )

    print(
        "================================================"
    )

    (
        physics_original_loocv,
        physics_original_errors,
        physics_original_fold_coefficients,
    ) = leave_one_out_evaluation(
        X=X,
        y=y,
        model_type="physics",
        experiment_name=(
            "Physics-Informed LR - Original Dataset - LOOCV"
        ),
        training_version="original",
        output_path=OUTPUT_PHYSICS_LOOCV_ORIGINAL,
    )

    # --------------------------------------------------------
    # Select outlier source
    # --------------------------------------------------------

    if OUTLIER_SELECTION_MODEL == "normal":
        outlier_error_df = (
            normal_original_errors
        )

    elif OUTLIER_SELECTION_MODEL == "physics":
        outlier_error_df = (
            physics_original_errors
        )

    else:
        raise ValueError(
            "OUTLIER_SELECTION_MODEL must be "
            "'normal' or 'physics'."
        )

    # --------------------------------------------------------
    # Remove selected high-error specimens
    # --------------------------------------------------------

    (
        df_cleaned,
        X_cleaned,
        y_cleaned,
        removed_specimens,
    ) = remove_top_outliers(
        df_training=df_training,
        X=X,
        y=y,
        error_df=outlier_error_df,
        n_outliers=N_OUTLIERS_TO_REMOVE,
    )

    # ========================================================
    # CLEANED DATASET: LOOCV
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "CLEANED DATASET: NORMAL LR LOOCV"
    )

    print(
        "================================================"
    )

    (
        normal_cleaned_loocv,
        normal_cleaned_errors,
        normal_cleaned_fold_coefficients,
    ) = leave_one_out_evaluation(
        X=X_cleaned,
        y=y_cleaned,
        model_type="normal",
        experiment_name=(
            "Normal LR - Cleaned Dataset - LOOCV"
        ),
        training_version="cleaned",
        output_path=OUTPUT_NORMAL_LOOCV_CLEANED,
    )

    print(
        "\n\n================================================"
    )

    print(
        "CLEANED DATASET: PHYSICS-INFORMED LR LOOCV"
    )

    print(
        "================================================"
    )

    (
        physics_cleaned_loocv,
        physics_cleaned_errors,
        physics_cleaned_fold_coefficients,
    ) = leave_one_out_evaluation(
        X=X_cleaned,
        y=y_cleaned,
        model_type="physics",
        experiment_name=(
            "Physics-Informed LR - Cleaned Dataset - LOOCV"
        ),
        training_version="cleaned",
        output_path=OUTPUT_PHYSICS_LOOCV_CLEANED,
    )

    # ========================================================
    # TRAIN FINAL MODELS ON ORIGINAL DATASET
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "TRAIN FINAL MODELS: ORIGINAL DATASET"
    )

    print(
        "================================================"
    )

    (
        normal_original_model,
        normal_original_scaler,
    ) = train_final_model(
        X_train=X,
        y_train=y,
        model_type="normal",
    )

    (
        physics_original_model,
        physics_original_scaler,
    ) = train_final_model(
        X_train=X,
        y_train=y,
        model_type="physics",
    )

    # ========================================================
    # GENERALIZATION: ORIGINAL DATASET MODELS
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "EXTERNAL GENERALIZATION: ORIGINAL MODELS"
    )

    print(
        "================================================"
    )

    (
        normal_original_generalization,
        normal_original_predictions,
    ) = evaluate_generalization(
        model=normal_original_model,
        scaler=normal_original_scaler,
        df_generalization=df_generalization,
        X_generalization=X_generalization,
        y_generalization=y_generalization,
        model_type="normal",
        training_version="original",
        output_path=(
            OUTPUT_GENERALIZATION_NORMAL_ORIGINAL
        ),
    )

    (
        physics_original_generalization,
        physics_original_predictions,
    ) = evaluate_generalization(
        model=physics_original_model,
        scaler=physics_original_scaler,
        df_generalization=df_generalization,
        X_generalization=X_generalization,
        y_generalization=y_generalization,
        model_type="physics",
        training_version="original",
        output_path=(
            OUTPUT_GENERALIZATION_PHYSICS_ORIGINAL
        ),
    )

    # ========================================================
    # TRAIN FINAL MODELS ON CLEANED DATASET
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "TRAIN FINAL MODELS: CLEANED DATASET"
    )

    print(
        "================================================"
    )

    (
        normal_cleaned_model,
        normal_cleaned_scaler,
    ) = train_final_model(
        X_train=X_cleaned,
        y_train=y_cleaned,
        model_type="normal",
    )

    (
        physics_cleaned_model,
        physics_cleaned_scaler,
    ) = train_final_model(
        X_train=X_cleaned,
        y_train=y_cleaned,
        model_type="physics",
    )

    # ========================================================
    # GENERALIZATION: CLEANED DATASET MODELS
    # ========================================================

    print(
        "\n\n================================================"
    )

    print(
        "EXTERNAL GENERALIZATION: CLEANED MODELS"
    )

    print(
        "================================================"
    )

    (
        normal_cleaned_generalization,
        normal_cleaned_predictions,
    ) = evaluate_generalization(
        model=normal_cleaned_model,
        scaler=normal_cleaned_scaler,
        df_generalization=df_generalization,
        X_generalization=X_generalization,
        y_generalization=y_generalization,
        model_type="normal",
        training_version="cleaned",
        output_path=(
            OUTPUT_GENERALIZATION_NORMAL_CLEANED
        ),
    )

    (
        physics_cleaned_generalization,
        physics_cleaned_predictions,
    ) = evaluate_generalization(
        model=physics_cleaned_model,
        scaler=physics_cleaned_scaler,
        df_generalization=df_generalization,
        X_generalization=X_generalization,
        y_generalization=y_generalization,
        model_type="physics",
        training_version="cleaned",
        output_path=(
            OUTPUT_GENERALIZATION_PHYSICS_CLEANED
        ),
    )

    # ========================================================
    # FINAL COEFFICIENT TABLE
    # ========================================================

    feature_names = list(
        X.columns
    )

    final_coefficients_df = pd.concat(
        [
            create_final_coefficient_table(
                model=normal_original_model,
                scaler=normal_original_scaler,
                model_type="normal",
                training_version="original",
                feature_names=feature_names,
            ),
            create_final_coefficient_table(
                model=physics_original_model,
                scaler=physics_original_scaler,
                model_type="physics",
                training_version="original",
                feature_names=feature_names,
            ),
            create_final_coefficient_table(
                model=normal_cleaned_model,
                scaler=normal_cleaned_scaler,
                model_type="normal",
                training_version="cleaned",
                feature_names=feature_names,
            ),
            create_final_coefficient_table(
                model=physics_cleaned_model,
                scaler=physics_cleaned_scaler,
                model_type="physics",
                training_version="cleaned",
                feature_names=feature_names,
            ),
        ],
        ignore_index=True,
    )

    final_coefficients_df.to_csv(
        OUTPUT_FINAL_COEFFICIENTS,
        sep=";",
        decimal=",",
        index=False,
    )

    # ========================================================
    # FOLD COEFFICIENTS
    # ========================================================

    fold_coefficients_df = pd.concat(
        [
            normal_original_fold_coefficients,
            physics_original_fold_coefficients,
            normal_cleaned_fold_coefficients,
            physics_cleaned_fold_coefficients,
        ],
        ignore_index=True,
    )

    fold_coefficients_df.to_csv(
        OUTPUT_FOLD_COEFFICIENTS,
        sep=";",
        decimal=",",
        index=False,
    )

    # ========================================================
    # COMPLETE PERFORMANCE COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        [
            normal_original_loocv,
            physics_original_loocv,
            normal_cleaned_loocv,
            physics_cleaned_loocv,
            normal_original_generalization,
            physics_original_generalization,
            normal_cleaned_generalization,
            physics_cleaned_generalization,
        ]
    )

    comparison_df.to_csv(
        OUTPUT_FINAL_COMPARISON,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "\n\n================================================"
    )

    print(
        "COMPLETE PERFORMANCE COMPARISON"
    )

    print(
        "================================================"
    )

    display_columns = [
        "Model",
        "Evaluation",
        "Training data",
        "Number of specimens",
        "Physics lambda",
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "MAPE (%)",
        "Mean error",
        "Average physics penalty",
        "Average constraint violations",
    ]

    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nSaved complete comparison to:",
        OUTPUT_FINAL_COMPARISON,
    )

    print(
        "Saved final coefficients to:",
        OUTPUT_FINAL_COEFFICIENTS,
    )

    print(
        "Saved LOOCV fold coefficients to:",
        OUTPUT_FOLD_COEFFICIENTS,
    )

    # ========================================================
    # GENERALIZATION-ONLY SUMMARY
    # ========================================================

    generalization_rows = comparison_df[
        comparison_df[
            "Evaluation"
        ] == "External generalization"
    ]

    print(
        "\n---------------- Generalization Summary ----------------"
    )

    print(
        generalization_rows[
            [
                "Model",
                "Training data",
                "MSE",
                "RMSE",
                "MAE",
                "R2",
                "MAPE (%)",
                "Mean error",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()