"""External evaluation of two models fitted to the same original training rows."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import FEATURES, NEGATIVE_FEATURES, POSITIVE_FEATURES, TARGET
from .data import clean_dataset, read_dataset
from .evaluation import coefficient_table, metrics, prediction_table, train_model
from .models import LinearRegression
from .plotting import plot_predictions


def evaluate_external(args):
    training = clean_dataset(read_dataset(args.data))
    external = clean_dataset(read_dataset(args.external))
    # Exact duplicate observations must not be reported as external validation.
    columns = FEATURES + [TARGET]
    overlap = training[columns].merge(external[columns], on=columns)
    if not overlap.empty:
        raise ValueError(
            "Training and external data contain identical observations; check the split."
        )
    historical = args.physics_method == "historical"
    negative = (
        ["Growth-ring width", "Height", "Test span", "Width"] if historical else NEGATIVE_FEATURES
    )

    def factory(physics):
        return LinearRegression(
            learning_rate=args.learning_rate,
            n_iterations=args.iterations,
            physics_lambda=args.physics_lambda if physics else 0,
            positive_features=POSITIVE_FEATURES if physics else (),
            negative_features=negative if physics else (),
            normalize_penalty=historical and physics,
        )

    factory(True)  # Validate settings before creating output files.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, physics in [("normal", False), ("physics", True)]:
        print(f"Fitting {name} on {len(training)} original specimens...", flush=True)
        model, scaler = train_model(training, lambda: factory(physics))
        predicted = model.predict(scaler.transform(external[FEATURES]))
        scores = metrics(external[TARGET], predicted)
        rows.append(
            {
                "model": name,
                "training_specimens": len(training),
                "external_specimens": len(external),
                **scores,
            }
        )
        table = prediction_table(external, predicted)
        table.to_csv(output / f"{name}_predictions.csv", index=False)
        coefficient_table(model, scaler).to_csv(output / f"{name}_coefficients.csv", index=False)
        if args.plot:
            label = "Normal LR" if not physics else f"Physics LR ({args.physics_method})"
            plot_predictions(
                table,
                output / f"{name}_predictions.png",
                title=f"{label} - External Generalization",
                prediction_label="External predictions",
                physics_style=physics,
            )
    summary = {
        "protocol": "Both models fitted on all original training rows; no outlier removal or external tuning",
        "physics_method": args.physics_method,
        "physics_reference": "Final_PI_LR_Outliers.py" if historical else "Last_drow_PI_LR.py",
        "physics_constraints": {
            "positive": POSITIVE_FEATURES,
            "negative": negative,
            "penalty": "mean" if historical else "sum",
        },
        "features": FEATURES,
        "settings": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "physics_lambda": args.physics_lambda,
        },
        "datasets": {
            key: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in [("training", args.data), ("external", args.external)]
        },
        "metric_definitions": {
            "rmse": "sqrt(mean squared error across external specimens)",
            "error": "measured minus predicted",
        },
        "results": rows,
    }
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return summary
