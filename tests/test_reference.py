"""Numerical regression checks against the user's unmodified reference functions."""

import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from wood_regression.config import FEATURES, NEGATIVE_FEATURES, POSITIVE_FEATURES, TARGET
from wood_regression.data import clean_dataset, read_dataset
from wood_regression.evaluation import cross_validate
from wood_regression.models import LinearRegression

ROOT = Path(__file__).resolve().parents[1]


def reference_namespace(physics, iterations=100):
    path = ROOT / "tests/reference" / ("Last_drow_PI_LR.py" if physics else "Last_drow_LR.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = {
        "PhysicsInformedLinearRegression",
        "LinearRegressionScratch",
        "leave_one_out_physics_informed_lr",
        "leave_one_out_linear_regression",
        "remove_top_outliers_by_error",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected
    ]
    namespace = dict(
        np=np,
        pd=pd,
        Path=Path,
        LeaveOneOut=LeaveOneOut,
        StandardScaler=StandardScaler,
        mean_squared_error=mean_squared_error,
        mean_absolute_error=mean_absolute_error,
        r2_score=r2_score,
        LEARNING_RATE=0.1,
        N_ITERATIONS=iterations,
        PHYSICS_LAMBDA=0.004,
        OUTPUT_PLOT_PATH=None,
        OUTPUT_EXCEL_PATH=None,
        plot_regression_results=lambda **kwargs: None,
        save_predictions_to_excel=lambda **kwargs: kwargs,
    )
    # Run only reviewed model/evaluation definitions; replace display/export side effects.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def factory(physics, iterations=100):
    return LinearRegression(
        n_iterations=iterations,
        physics_lambda=0.004 if physics else 0,
        positive_features=POSITIVE_FEATURES if physics else (),
        negative_features=NEGATIVE_FEATURES if physics else (),
    )


@pytest.mark.parametrize("physics", [False, True])
def test_every_loocv_prediction_matches_user_reference(physics):
    df = clean_dataset(read_dataset(ROOT / "Data/final_clean_merged_dataframe.csv"))
    ns = reference_namespace(physics)
    sizes = df.source_file.str.extract(r"Size_([ABC])", expand=False)
    with redirect_stdout(StringIO()):
        if physics:
            expected = ns["leave_one_out_physics_informed_lr"](
                df[FEATURES], df[TARGET].to_numpy(), sizes, df.index
            )
            expected_predictions = np.array(expected[3]["y_pred"])
        else:
            expected = ns["leave_one_out_linear_regression"](
                df[FEATURES], df[TARGET].to_numpy(), sizes
            )
            expected_predictions = expected[3]
    scores, table = cross_validate(df, lambda: factory(physics))
    np.testing.assert_allclose(table.predicted_value, expected_predictions, atol=1e-10, rtol=0)
    assert scores["average_rmse"] == pytest.approx(scores["mae"])
    assert scores["rmse"] == pytest.approx(np.sqrt(scores["mse"]))
    if physics:
        assert scores["average_rmse"] == pytest.approx(np.mean(expected[1]))
    else:
        expected_removed = expected[1].head(5).original_index.to_list()
        actual_removed = (
            table.sort_values("absolute_error", ascending=False).head(5).original_index.to_list()
        )
        assert actual_removed == expected_removed
        cleaned = df.drop(index=actual_removed)
        with redirect_stdout(StringIO()):
            expected_cleaned = ns["leave_one_out_linear_regression"](
                cleaned[FEATURES], cleaned[TARGET].to_numpy(), sizes
            )
        _, actual_cleaned = cross_validate(cleaned, lambda: factory(False))
        np.testing.assert_allclose(
            actual_cleaned.predicted_value, expected_cleaned[3], atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("physics", [False, True])
def test_full_iteration_fit_matches_reference(physics):
    df = clean_dataset(read_dataset(ROOT / "Data/final_clean_merged_dataframe.csv"))
    X = StandardScaler().fit_transform(df[FEATURES])
    y = df[TARGET].to_numpy()
    ns = reference_namespace(physics, iterations=120750)
    if physics:
        model = ns["PhysicsInformedLinearRegression"](
            learning_rate=0.1, n_iterations=120750, physics_lambda=0.004
        )
        model.fit(X, y, FEATURES)
    else:
        model = ns["LinearRegressionScratch"](learning_rate=0.1, n_iterations=120750)
        model.fit(X, y)
    actual = factory(physics, iterations=120750).fit(X, y, FEATURES)
    np.testing.assert_allclose(actual.predict(X), model.predict(X), atol=1e-10, rtol=0)
