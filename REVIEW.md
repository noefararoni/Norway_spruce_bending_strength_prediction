# Review and cleanup decisions

## Current reference correction

The authoritative methods are now the user-supplied Last_drow_LR.py and
Last_drow_PI_LR.py, copied unchanged into tests/reference with their configuration
and preprocessing. The earlier selection of Final_PI_LR_Outliers.py changed the
physics model. This correction leaves Width/Height unconstrained and sums penalties
on Density, Dynamic E, Growth-ring width and Test span. Data fingerprints and
configuration match the reference Wood project.

Normal regression defaults to original CV then removal of five high-error specimens;
physics keeps all specimens. Both overall RMSE and average fold RMSE are saved and
clearly distinguished. Reference physics plots used average fold RMSE (= MAE in LOOCV).

All 19 tests pass, including all original/cleaned CV predictions against the supplied
functions at 100 iterations and both full 120750-iteration model fits (1e-10 tolerance).
Full corrected experiment runs are saved in outputs/reference-normal and
outputs/reference-physics. Earlier output folders have not been overwritten.

The CLI now records dataset SHA-256, feature order, metric definitions, fold progress,
size labels and squared errors. Figures are 300 DPI. CSV/JSON reports are retained;
reference-specific Excel layouts remain in tests/reference. Signed errors consistently
use measured minus predicted. Final coefficients fit all retained rows rather than
using the last CV model.


Cleaned cross-validation scores are exploratory because sample removal was selected
using validation errors. External validation and final all-row fitting are package
extensions to the supplied scripts.

## Full experiment results (120750 iterations per fit)

| Model / dataset | MSE | Overall RMSE | Average fold RMSE | R² |
|---|---:|---:|---:|---:|
| normal / cross_validation | 54.5839 | 7.3881 | 5.8780 | 0.6337 |
| normal / cleaned_cross_validation | 39.5512 | 6.2890 | 5.1679 | 0.7129 |
| physics / cross_validation | 56.3269 | 7.5051 | 5.8446 | 0.6221 |

All saved MSE values were independently recomputed from exported prediction CSVs.
The physics figures reproduce the expected rounded 56.33 MSE, 5.84 average fold RMSE
and 0.622 R². Full cross-validation, final fits, and external validation completed.
