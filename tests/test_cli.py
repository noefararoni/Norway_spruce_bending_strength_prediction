"""Exercise installed CLI workflows against the supplied research datasets."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from wood_regression.config import FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Data/final_clean_merged_dataframe.csv"
EXTERNAL = ROOT / "Data/generalisation_validation.csv"


def invoke(tmp_path, *args):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    return subprocess.run(
        [sys.executable, "-m", "wood_regression", *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("model", ["normal", "physics"])
def test_full_cli_workflow(tmp_path, model):
    output = tmp_path / "run"
    options = ["--folds", "3"] if model == "normal" else ["--remove-outliers", "2"]
    result = invoke(
        tmp_path,
        "run",
        "--data",
        TRAINING,
        "--external",
        EXTERNAL,
        "--output",
        output,
        "--model",
        model,
        "--iterations",
        "100",
        "--plot",
        *options,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "metrics.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv")
    assert len(predictions) == summary["specimens"] == 99
    assert predictions.original_index.is_unique
    error = predictions[TARGET] - predictions.predicted_value
    assert summary["cross_validation"]["mse"] == pytest.approx(np.mean(error**2))
    assert summary["cross_validation"]["mae"] == pytest.approx(np.mean(abs(error)))
    external = pd.read_csv(output / "external_predictions.csv")
    coefficients = pd.read_csv(output / "coefficients.csv").set_index("feature").coefficient
    np.testing.assert_allclose(
        coefficients["intercept"] + external[FEATURES].to_numpy() @ coefficients[FEATURES],
        external.predicted_value,
        atol=1e-8,
    )
    assert summary["external_validation"]["mse"] == pytest.approx(
        np.mean((external[TARGET] - external.predicted_value) ** 2)
    )
    with Image.open(output / "predictions.png") as picture:
        picture.verify()
    if model == "physics":
        removed = pd.read_csv(output / "removed_specimens.csv")
        cleaned = pd.read_csv(output / "cleaned_predictions.csv")
        assert len(removed) == 2 and len(cleaned) == 97
        assert set(removed.original_index).isdisjoint(cleaned.original_index)
        assert set(removed.original_index) == set(
            predictions.nlargest(2, "absolute_error").original_index
        )
        assert "selection_warning" in summary


@pytest.mark.parametrize(
    "option,value",
    [
        ("--folds", "1"),
        ("--folds", "100"),
        ("--iterations", "0"),
        ("--remove-outliers", "98"),
        ("--remove-outliers", "-1"),
        ("--learning-rate", "nan"),
    ],
)
def test_invalid_settings_create_no_outputs(tmp_path, option, value):
    output = tmp_path / "invalid"
    result = invoke(tmp_path, "run", "--data", TRAINING, "--output", output, option, value)
    assert result.returncode != 0
    assert not output.exists()


def test_prepare_and_overwrite_protection(tmp_path):
    output = tmp_path / "merged.csv"
    result = invoke(tmp_path, "prepare", TRAINING, "--output", output)
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(output)) == 99
    original = output.read_bytes()
    result = invoke(tmp_path, "prepare", TRAINING, "--output", output)
    assert result.returncode != 0
    assert output.read_bytes() == original
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "keep.txt"
    marker.write_text("preserve this result")
    result = invoke(tmp_path, "run", "--data", TRAINING, "--output", run_dir, "--iterations", "1")
    assert result.returncode != 0
    assert marker.read_text() == "preserve this result"
    assert list(run_dir.iterdir()) == [marker]


def test_incomplete_raw_data_fails_without_output(tmp_path):
    output = tmp_path / "incomplete.csv"
    result = invoke(tmp_path, "prepare", ROOT / "Data/Size_A.csv", "--output", output)
    assert result.returncode != 0
    assert "Dynamic E" in result.stderr
    assert not output.exists()
