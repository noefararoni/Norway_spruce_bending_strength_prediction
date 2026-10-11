"""Guard the historical external comparison and prevent accidental split reuse."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wood_regression.generalization import evaluate_external

ROOT = Path(__file__).resolve().parents[1]


def settings(tmp_path, method="historical"):
    return Namespace(
        data=ROOT / "Data/final_clean_merged_dataframe.csv",
        external=ROOT / "Data/generalisation_validation.csv",
        output=tmp_path / "results",
        iterations=120750,
        learning_rate=0.1,
        physics_lambda=0.004,
        physics_method=method,
        plot=False,
    )


def test_historical_generalization_matches_saved_reference(tmp_path):
    args = settings(tmp_path)
    summary = evaluate_external(args)
    expected = {
        "normal": (127.2263859046337, 0.3158976089879889),
        "physics": (114.03847576172102, 0.386809635585842),
    }
    for row in summary["results"]:
        assert row["training_specimens"] == 99
        assert row["external_specimens"] == 25
        mse, r2 = expected[row["model"]]
        assert row["mse"] == pytest.approx(mse, abs=1e-8)
        assert row["r2"] == pytest.approx(r2, abs=1e-10)
        assert row["rmse"] == pytest.approx(np.sqrt(mse))
        predictions = pd.read_csv(args.output / f"{row['model']}_predictions.csv")
        assert np.mean(predictions.error**2) == pytest.approx(mse, abs=1e-8)
    assert summary["physics_constraints"]["penalty"] == "mean"


def test_external_rows_never_enter_scaler_fit(tmp_path, monkeypatch):
    from wood_regression.generalization import train_model

    args = settings(tmp_path, "plot-reference")
    args.iterations = 20
    sizes = []

    def checked_fit(df, factory):
        sizes.append(len(df))
        return train_model(df, factory)

    monkeypatch.setattr("wood_regression.generalization.train_model", checked_fit)
    summary = evaluate_external(args)
    assert sizes == [99, 99]
    assert summary["physics_constraints"]["penalty"] == "sum"
    with pytest.raises(FileExistsError):
        evaluate_external(args)


def test_training_external_overlap_rejected(tmp_path):
    args = settings(tmp_path)
    args.external = args.data
    with pytest.raises(ValueError, match="identical observations"):
        evaluate_external(args)
    assert not args.output.exists()
