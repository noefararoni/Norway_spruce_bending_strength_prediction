# Wood regression

Predict wood bending strength using ordinary or physics-informed linear regression.
The active implementation is in `src/wood_regression`; historical scripts are in
`archive/legacy` for reference. They are not part of the installed package.

## Setup (Windows PowerShell)

Create an isolated `.venv` after cloning this repository, using Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Activation is optional. All commands below call the environment directly.
For the exact dependency versions verified here, install `requirements-lock.txt`
before installing this package with `pip install --no-deps -e .`.

## Run

Run these commands from the project root. Paths are explicit, so the installed
command also works from other directories when given absolute paths.
Each run requires a new output directory to protect previous results.

```powershell
# Fast functional check (not a converged research experiment)
.\.venv\Scripts\python.exe -m wood_regression run --data Data/final_clean_merged_dataframe.csv --output outputs/smoke --folds 2 --iterations 100 --plot

# Original iteration count, leave-one-out evaluation and external validation
.\.venv\Scripts\python.exe -m wood_regression run --data Data/final_clean_merged_dataframe.csv --external Data/generalisation_validation.csv --output outputs/normal

# Physics-informed model
.\.venv\Scripts\python.exe -m wood_regression run --data Data/final_clean_merged_dataframe.csv --external Data/generalisation_validation.csv --output outputs/physics --model physics

# Verification
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Use `--folds N` for K-fold evaluation, otherwise leave-one-out is used. Use
`--plot` to export a measured-versus-predicted figure. Outputs include predictions,
MSE/RMSE/MAE/R², run settings, and coefficients converted to original feature units.
External evaluation uses the scaler and model fitted only on training data.
Figures include MSE (MPa²), RMSE (MPa), R², a model/dataset title, and a
perfect-prediction reference line. Metrics are calculated from the plotted points.
With `--plot`, external validation also produces `external_predictions.png`;
optional outlier removal also produces `cleaned_predictions.png`.

## Structure

```text
src/wood_regression/   shared configuration, data, model, evaluation, plots and CLI
tests/                numerical, CSV, provenance and data-leakage checks
Data/                 existing input datasets and historical results (preserved)
generalization/       existing additional dataset (preserved)
outputs/              new experiment results, ignored by Git
archive/legacy/       superseded scripts and original configuration
archive/results/      historical results previously scattered in the project root
archive/environment/  previous non-working virtual environment, ignored by Git
pyproject.toml        package metadata, dependencies and tool configuration
requirements-lock.txt exact dependency snapshot from the verified environment
```

## Data and research assumptions

The target is `Bending strength`. Predictors are Width, Height, Density,
Growth-ring width, Test span and Dynamic E. Tracking fields never enter the model.
CSV reading supports semicolons with decimal commas and commas with decimal points.
Cleaning discards rows with missing, nonnumeric or infinite predictor/target values,
retaining original row indices and available source tracking fields.

The existing merged dataset is the default example because the raw Size_A/B/C
files do not all provide the required Dynamic E feature. The `prepare` command
accepts complete source CSVs and fails clearly when a required feature is missing:

```powershell
.\.venv\Scripts\python.exe -m wood_regression prepare path/to/source1.csv path/to/source2.csv --output outputs/merged.csv
```

Physics constraints follow the user-approved `Last_drow_PI_LR.py`: positive Density
and Dynamic E; negative Growth-ring width and Test span. Width and Height are
unconstrained. The penalty is summed, not averaged. These are soft sign assumptions.
Defaults preserve learning rate 0.1, 120750 iterations, and lambda 0.004.
The earlier six-feature normalized penalty was a different model.

Matching `Last_drow_LR.py`, normal regression defaults to original CV followed by CV
after removing the five highest-error specimens. Physics regression keeps all rows.
Use `--remove-outliers 0` to disable removal or `--remove-outliers N` to set a count.
Its cleaned cross-validation scores have selection bias and must not be interpreted
as independent performance estimates. Reserve untouched external data for assessment.

See `REVIEW.md` for the cleanup decisions and known limitations.

## Reference-aligned results

Local corrected full experiments were saved in `outputs/reference-normal` and
`outputs/reference-physics`. Generated outputs are not included in Git; rerun training
to create your own results. The verified scores are documented in `REVIEW.md`.
Use `--plot` on either training command to save the figures.

Physics CV figures show both overall RMSE and average fold RMSE. Overall RMSE is
sqrt(MSE); average fold RMSE is the mean of per-fold RMSEs, which equals MAE in
leave-one-out validation. The reference physics figure reports the latter.
Normal figures include MAE and use equal axes; all figures are saved at 300 DPI.

Each run records the training CSV SHA-256, feature order, settings and metric
definitions. Prediction CSVs retain original indices, source tracking, Size A/B/C,
fold number and errors. Signed error is consistently measured minus predicted;
the reference physics Excel helper used the opposite sign. Reports use CSV/JSON
and PNG; reference-specific Excel layouts remain in `tests/reference`.

Unmodified supplied scripts, configuration and preprocessing are in `tests/reference`.
Regression tests execute their actual model/evaluation bodies with display/export
side effects replaced, checking prediction equivalence. Progress prints every ten folds.
