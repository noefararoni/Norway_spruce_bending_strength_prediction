"""Command-line entry points for preparation and regression experiments."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .config import (
    FEATURES,
    LEARNING_RATE,
    N_ITERATIONS,
    NEGATIVE_FEATURES,
    PHYSICS_LAMBDA,
    POSITIVE_FEATURES,
    TARGET,
)
from .data import clean_dataset, merge_datasets, read_dataset
from .evaluation import coefficient_table, cross_validate, metrics, prediction_table, train_model
from .models import LinearRegression
from .plotting import plot_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Merge and clean input CSVs")
    prepare.add_argument("inputs", nargs="+", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Cross-validate and export a fitted model's coefficients")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New directory for this run")
    run.add_argument("--external", type=Path)
    run.add_argument("--model", choices=["normal", "physics"], default="normal")
    run.add_argument("--folds", type=int, help="Omit for leave-one-out evaluation")
    run.add_argument("--iterations", type=int, default=N_ITERATIONS)
    run.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    run.add_argument("--physics-lambda", type=float, default=PHYSICS_LAMBDA)
    run.add_argument(
        "--remove-outliers",
        type=int,
        default=None,
        help="Highest-error removal (default: normal=5, physics=0); use 0 to disable",
    )
    run.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        df = merge_datasets(args.inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error("Output already exists; choose a new filename.")
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} specimens to {args.output}")
        return
    df = clean_dataset(read_dataset(args.data))
    if args.remove_outliers is None:
        args.remove_outliers = 5 if args.model == "normal" else 0
    if not 0 <= args.remove_outliers < len(df) - 1:
        parser.error("Outlier count must leave at least two specimens.")
    if args.folds is not None and not 2 <= args.folds <= len(df) - args.remove_outliers:
        parser.error("Fold count must be between 2 and the remaining specimen count.")
    external = clean_dataset(read_dataset(args.external)) if args.external else None

    def factory():
        return LinearRegression(
            learning_rate=args.learning_rate,
            n_iterations=args.iterations,
            physics_lambda=args.physics_lambda if args.model == "physics" else 0,
            positive_features=POSITIVE_FEATURES if args.model == "physics" else (),
            negative_features=NEGATIVE_FEATURES if args.model == "physics" else (),
        )

    factory()  # Validate model settings before creating outputs.
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "specimens": len(df),
        "method": "Last_drow_LR / Last_drow_PI_LR reference equations; summed sign penalty",
        "training_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "features": FEATURES,
        "metric_definitions": {
            "rmse": "sqrt(overall MSE)",
            "average_rmse": "mean of per-fold RMSE; equals MAE for LOOCV",
            "error": "measured minus predicted",
        },
    }

    def progress(fold, total):
        if fold == 1 or fold % 10 == 0 or fold == total:
            print(f"{args.model}: completed fold {fold}/{total}", file=sys.stderr, flush=True)

    summary["cross_validation"], predictions = cross_validate(df, factory, args.folds, progress)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    model_title = "Physics-Informed LR" if args.model == "physics" else "Linear Regression"
    prediction_label = (
        "LOOCV predictions" if args.folds is None else f"{args.folds}-fold CV predictions"
    )
    if args.plot:
        plot_predictions(
            predictions,
            args.output / "predictions.png",
            title=f"{model_title} - Original Dataset",
            prediction_label=prediction_label,
            physics_style=args.model == "physics",
        )
    if args.remove_outliers:
        removed = predictions.sort_values("absolute_error", ascending=False).head(
            args.remove_outliers
        )
        removed.to_csv(args.output / "removed_specimens.csv", index=False)
        df = df.drop(index=removed.index)
        summary["selection_warning"] = (
            "Outliers selected using CV errors. Cleaned CV scores are exploratory and biased; "
            "use untouched external data for final assessment."
        )
        summary["cleaned_cross_validation"], cleaned = cross_validate(
            df, factory, args.folds, progress
        )
        cleaned.to_csv(args.output / "cleaned_predictions.csv", index=False)
        if args.plot:
            plot_predictions(
                cleaned,
                args.output / "cleaned_predictions.png",
                title=f"{model_title} - Cleaned Dataset (Exploratory)",
                prediction_label=prediction_label,
                physics_style=args.model == "physics",
            )
    model, scaler = train_model(df, factory)
    coefficient_table(model, scaler).to_csv(args.output / "coefficients.csv", index=False)
    if external is not None:
        predicted = model.predict(scaler.transform(external[FEATURES]))
        summary["external_validation"] = metrics(external[TARGET], predicted)
        external_table = prediction_table(external, predicted)
        external_table.to_csv(args.output / "external_predictions.csv", index=False)
        if args.plot:
            plot_predictions(
                external_table,
                args.output / "external_predictions.png",
                title=f"{model_title} - External Validation",
                prediction_label="External predictions",
                physics_style=args.model == "physics",
            )
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
