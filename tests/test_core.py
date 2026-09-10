import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression as ReferenceRegression
from sklearn.preprocessing import StandardScaler

from wood_regression.config import FEATURES, TARGET
from wood_regression.data import clean_dataset, read_dataset
from wood_regression.evaluation import coefficient_table, cross_validate, train_model
from wood_regression.models import LinearRegression


def test_csv_decimal_formats_and_tracking(tmp_path):
    frame = pd.DataFrame({name: [1.25, 2.5] for name in FEATURES + [TARGET]})
    frame["source_file"] = "Size_A"
    frame["source_row"] = [3, 8]
    for separator, decimal in [(";", ","), (",", ".")]:
        path = tmp_path / "input.csv"
        frame.to_csv(path, sep=separator, decimal=decimal, index=False)
        pd.testing.assert_frame_equal(clean_dataset(read_dataset(path)), frame)


def test_cleaning_preserves_indices_and_rejects_missing_features():
    frame = pd.DataFrame({name: [1, np.inf, "bad", 4] for name in FEATURES + [TARGET]})
    assert clean_dataset(frame).index.tolist() == [0, 3]
    with pytest.raises(ValueError, match="Missing"):
        clean_dataset(frame.drop(columns=FEATURES[0]))


def test_regression_matches_least_squares():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3))
    y = X @ [2, -3, 0.5] + 7
    model = LinearRegression(n_iterations=2000).fit(X, y, ["a", "b", "c"])
    reference = ReferenceRegression().fit(X, y)
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-8)


def test_soft_constraint_reduces_sign_violation():
    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = -2 * X[:, 0]
    plain = LinearRegression(n_iterations=1000).fit(X, y, ["Density"])
    physics = LinearRegression(
        n_iterations=1000, physics_lambda=1, positive_features=["Density"]
    ).fit(X, y, ["Density"])
    assert plain.weights[0] < physics.weights[0] < 0
    with pytest.raises(RuntimeError):
        LinearRegression().predict(X)


def test_cv_scaling_is_fitted_only_on_training_rows(monkeypatch):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(8, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = np.arange(8, dtype=float)
    fit_sizes = []
    original_fit = StandardScaler.fit

    def tracked_fit(self, X, *args, **kwargs):
        fit_sizes.append(len(X))
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", tracked_fit)
    scores, table = cross_validate(frame, lambda: LinearRegression(n_iterations=100), folds=4)
    assert fit_sizes == [6, 6, 6, 6]
    assert len(table) == 8 and np.isfinite(scores["rmse"])
    model, scaler = train_model(frame, lambda: LinearRegression(n_iterations=100))
    coefficients = coefficient_table(model, scaler)["coefficient"].to_numpy()
    np.testing.assert_allclose(
        coefficients[0] + frame[FEATURES].to_numpy() @ coefficients[1:],
        model.predict(scaler.transform(frame[FEATURES])),
        atol=1e-10,
    )
