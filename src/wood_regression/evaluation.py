"""Shared evaluation with scaling fitted independently in each training fold."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, RANDOM_STATE, TARGET


def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def prediction_table(df, predicted):
    table = df.copy()
    table.insert(0, "original_index", df.index)
    table["predicted_value"] = predicted
    table["error"] = df[TARGET].to_numpy() - predicted
    table["absolute_error"] = table["error"].abs()
    table["squared_error"] = table["error"] ** 2
    if "source_file" in table:
        table["size"] = (
            table["source_file"]
            .astype(str)
            .str.extract(r"(?i)size[_\-\s]*([ABC])", expand=False)
            .str.upper()
        )
    return table


def cross_validate(df, model_factory, folds=None, progress=None):
    if len(df) < 2:
        raise ValueError("Cross-validation requires at least two specimens.")
    splitter = (
        LeaveOneOut()
        if folds is None
        else KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    )
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    predictions = np.empty(len(df))
    fold_numbers = np.empty(len(df), dtype=int)
    fold_mse = []
    total = splitter.get_n_splits(X)
    for fold, (train, test) in enumerate(splitter.split(X), start=1):
        scaler = StandardScaler().fit(X[train])
        model = model_factory().fit(scaler.transform(X[train]), y[train], FEATURES)
        predictions[test] = model.predict(scaler.transform(X[test]))
        fold_numbers[test] = fold
        fold_mse.append(mean_squared_error(y[test], predictions[test]))
        if progress is not None:
            progress(fold, total)
    scores = metrics(y, predictions)
    scores.update(
        average_mse=float(np.mean(fold_mse)),
        std_mse=float(np.std(fold_mse)),
        average_rmse=float(np.mean(np.sqrt(fold_mse))),
        std_rmse=float(np.std(np.sqrt(fold_mse))),
    )
    table = prediction_table(df, predictions)
    table["fold"] = fold_numbers
    return scores, table


def train_model(df, model_factory):
    scaler = StandardScaler().fit(df[FEATURES])
    model = model_factory().fit(scaler.transform(df[FEATURES]), df[TARGET].to_numpy(), FEATURES)
    return model, scaler


def coefficient_table(model, scaler):
    weights = model.weights / scaler.scale_
    return pd.DataFrame(
        {
            "feature": ["intercept"] + FEATURES,
            "coefficient": [model.bias - scaler.mean_ @ weights] + list(weights),
        }
    )
