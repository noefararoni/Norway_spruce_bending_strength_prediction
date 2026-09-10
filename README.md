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
The data was obtained from non-destructive measurements and three-point bending experiments. The non-destructive measurements considered equilibrium density at 65% air relative humidity and 20 °C, dynamic modulus of elasticity, growth-ring width, height, span between supports, and width. The three-point bending experiments delivered the bending strength.


## Reference
https://doi.org/10.1016/j.conbuildmat.2025.140719
