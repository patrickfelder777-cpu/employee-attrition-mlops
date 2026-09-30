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


## Results

The configured Random Forest achieved these results on the stratified 20% test split:

| Metric | Score |
|---|---:|
| Accuracy | 83.67% |
| Precision | 48.78% |
| Recall | 42.55% |
| F1-score | 0.4545 |
| ROC-AUC | 0.7705 |

Precision, recall, and F1 measure performance for the positive attrition class. Recall indicates that the model identified approximately 43% of employees labeled as leaving in this test split, so accuracy alone does not fully describe its performance.

The local training run completed and logged results to MLflow. All 24 automated tests also passed locally.

These results come from one train/test split. Future evaluation should compare against a baseline and use cross-validation before drawing broader conclusions.


## Run locally

With Python 3.12 installed, run these commands from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m src.train --config configs/config.yaml
```

The training CSV is included in the repository. Training saves the model under `models/` and metrics under `reports/`.

## Future improvements

- Configure a shareable DVC remote; the current remote refers to storage on the original development computer.
- Verify the drift-monitoring script and add monitoring checks to CI.
- Compare model performance against a baseline and use cross-validation.
