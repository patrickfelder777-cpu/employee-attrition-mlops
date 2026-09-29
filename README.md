# Employee Attrition MLOps

An educational machine learning project that predicts employee attrition and explores practices for tracking, testing, and monitoring a model.

## What the project does

- Loads the included employee attrition dataset and splits it into training and test sets.
- Introduces a small amount of simulated missing data to test preprocessing.
- Imputes missing values, scales numerical features, and encodes categorical features.
- Trains a classification model and reports accuracy, precision, recall, F1, and ROC-AUC.
- Tracks experiments with MLflow and saves the trained pipeline and metrics.
- Uses Evidently to compare reference data with simulated production data for drift.
- Includes automated tests and a GitHub Actions workflow.

## Dataset and scope

The included CSV has 1,470 rows. The target is `Attrition` (`Yes` or `No`). This is a portfolio project; its predictions are not intended for employment decisions about real people.

## Key files

- `src/train.py` — training, evaluation, MLflow tracking, and saved outputs
- `src/preprocessing.py` — data validation and preprocessing
- `src/monitor_drift.py` — simulated drift analysis
- `configs/` — model and experiment settings
- `tests/` — automated checks
- `MONITORING.md` — monitoring approach and recorded results

## Reproducibility work in progress

The dependency file and DVC storage configuration need cleanup before a new user can reliably reproduce the project from a fresh clone. The repository currently includes the CSV used by the training script.

## Next improvements

Clean up the tracked Python environment, repair the dependency list, make the data setup portable, and verify the complete workflow in GitHub Actions.
