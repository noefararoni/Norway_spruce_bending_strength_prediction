# Background

A previous study proposed a method to project the ultimate force of a scaled hybrid glued softwood-to-hardwood column-to-slab connection to the full scale by accounting for different strengths at different sizes [1]. The study indicated that the projection accuracy depends on accurate strength values across different sizes. The projection method is defined as dimensional modelling by Szirtes and Rózsa [2] and is based on the Buckingham $\pi$ theorem [3]. In order to better predict bending strength necessary for the projection method, machine-learning models are evaluated.

# Wood regression

## Generalization test (historical comparison)

From the repository root, after installing the package with `python -m pip install -e .`, run:

```powershell
python -m wood_regression generalize --data Data/final_clean_merged_dataframe.csv --external Data/generalisation_validation.csv --output outputs/generalization-final --plot
```

This fits both models on the same 99 original training specimens and evaluates the
same 25 external specimens. It performs no outlier removal, cross-validation or
hyperparameter search on the external data. Scaling is fitted only on training rows.
An exact training/external observation overlap is rejected; this check does not prove
that datasets are independent at the tree, batch or source level.

The default `--physics-method historical` reproduces `Final_PI_LR_Outliers.py`:
positive Density/Dynamic E, negative Growth-ring width/Height/Test span/Width,
and a penalty averaged across six constrained features. It uses 120750 iterations,
learning rate 0.1 and lambda 0.004, fixed from the original configuration.

Outputs: `comparison.csv`, `metrics.json`, and each model's predictions, coefficients
and optional PNG. The manifest records dataset hashes, settings, exact physics
constraints and metric definitions. Choose a new output directory for each run.

The approved plotting model remains available with `--physics-method plot-reference`.
That method follows `Last_drow_PI_LR.py`, leaves Width/Height unconstrained and sums
the penalty; its results differ. Existing `run` commands retain their behavior.
In particular, `run --model normal --external ...` defaults to five-row removal;
use `generalize` above for a consistent original-data model comparison.

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


## References
[1] https://doi.org/10.1016/j.conbuildmat.2025.140719

[2] https://doi.org/10.1016/B978-0-12-370620-1.X5000-X

[3] https://doi.org/10.1103/PhysRev.4.345



