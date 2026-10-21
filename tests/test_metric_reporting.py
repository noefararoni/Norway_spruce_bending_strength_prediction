"""Prevent average LOOCV fold error from being reported as pooled RMSE."""

import numpy as np
import pandas as pd
import pytest

from wood_regression.config import TARGET
from wood_regression.evaluation import metrics
from wood_regression.plotting import plot_predictions


def test_pooled_rmse_is_not_average_absolute_error():
    scores = metrics([0, 0], [1, 3])
    assert scores["mse"] == 5
    assert scores["rmse"] == pytest.approx(np.sqrt(5))
    assert scores["mae"] == 2
    assert scores["rmse"] ** 2 == pytest.approx(scores["mse"])
    assert scores["rmse"] != scores["mae"]


@pytest.mark.parametrize("physics_style", [False, True])
def test_figure_reports_pooled_rmse_and_mae(tmp_path, monkeypatch, physics_style):
    from matplotlib.axes import Axes

    annotations = []
    original = Axes.text

    def capture(self, x, y, text, *args, **kwargs):
        annotations.append(text)
        return original(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture)
    table = pd.DataFrame({TARGET: [2, 4], "predicted_value": [1, 1], "fold": [1, 2]})
    plot_predictions(table, tmp_path / "figure.png", physics_style=physics_style)
    box = next(text for text in annotations if "MSE =" in text)
    assert "MSE = 5.00" in box
    assert "RMSE = 2.24" in box
    assert "MAE = 2.00" in box
    assert "Average" not in box
