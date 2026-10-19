
# Employee Attrition MLOps

[![CI Pipeline](https://github.com/patrickfelder777-cpu/employee-attrition-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/patrickfelder777-cpu/employee-attrition-mlops/actions/workflows/ci.yml)

An educational, end-to-end machine learning project that predicts employee attrition and demonstrates model training, experiment tracking, automated testing, API development, and continuous integration.

**Project status:** GitHub Actions CI passed with 28 automated tests.

> This project is for educational and portfolio purposes only. Its predictions should not be used to make employment decisions about real people.

## Project Overview

The project uses an IBM HR Analytics employee attrition dataset to train a Random Forest classification model.

The workflow includes:

- Data validation and preprocessing with scikit-learn
- Numerical imputation, scaling, and categorical encoding
- Random Forest model training and evaluation
- Experiment tracking with MLflow
- Dataset version tracking with DVC
- Model serving through a Flask REST API
- Data drift analysis using Evidently
- Automated testing with pytest
- Continuous integration using GitHub Actions

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | scikit-learn, pandas, NumPy |
| API | Flask |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Monitoring | Evidently |
| Testing | pytest |
| CI/CD | GitHub Actions |

## Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Source:** [Kaggle dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- 1,470 records
- 35 original columns
- Target: `Attrition` (`Yes` or `No`)

The training CSV is tracked using DVC metadata rather than committed directly to Git.

The configured DVC remote uses local storage on the original development computer. For reproducible GitHub Actions runs, the workflow independently downloads the public Kaggle dataset and verifies its checksum before training.

## Model Performance

The configured Random Forest achieved the following results on a stratified 20% test split:

| Metric | Score |
|---|---:|
| Accuracy | 83.67% |
| Precision | 48.78% |
| Recall | 42.55% |
| F1-score | 0.4545 |
| ROC-AUC | 0.7705 |

Precision, recall, and F1 refer to the positive attrition class.

The model identified approximately 43% of the employees labeled as leaving in the test split. This illustrates why accuracy alone is insufficient for evaluating imbalanced classification problems.

These results are based on one train/test split. Further evaluation should include baseline comparisons and cross-validation.

## Flask Prediction API

The project includes a Flask REST API in `src/api.py`.

### Health Check

**Endpoint:** `GET /health`

Checks whether the prediction service is running.

### Generate Predictions

**Endpoint:** `POST /predict`

Accepts employee feature values in JSON format and returns:

- Predicted attrition class
- Prediction probabilities

The API requires a trained model saved at `models/attrition_model.joblib`.

The API is intended as a local demonstration, not a production employment decision system.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/patrickfelder777-cpu/employee-attrition-mlops.git
cd employee-attrition-mlops
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the dataset

```powershell
New-Item -ItemType Directory -Force -Path data/raw

curl.exe -L --fail --output employee-attrition.zip "https://www.kaggle.com/api/v1/datasets/download/pavansubhasht/ibm-hr-analytics-attrition-dataset"

Expand-Archive employee-attrition.zip -DestinationPath data/raw/downloaded -Force

Copy-Item data/raw/downloaded/WA_Fn-UseC_-HR-Employee-Attrition.csv data/raw/employee_attrition.csv
```

The expected MD5 checksum of `data/raw/employee_attrition.csv` is:

```text
AD8207459E5732574372CF8FF619883F
```

Verify the downloaded file:

```powershell
Get-FileHash data/raw/employee_attrition.csv -Algorithm MD5
```

### 5. Train the model

```powershell
python -m src.train --config configs/config.yaml
```

Training saves the model under `models/` and metrics under `reports/`. Experiment information is logged through MLflow.

### 6. Run automated tests

```powershell
python -m pytest tests/ -v -p no:cacheprovider
```

### 7. Start the Flask API

```powershell
python -m src.api
```

The API provides `/health` and `/predict` endpoints. The prediction endpoint requires JSON containing the model's expected employee features.

## Continuous Integration

The GitHub Actions workflow automatically:

1. Checks out the repository.
2. Installs Python and project dependencies.
3. Downloads the employee attrition dataset.
4. Verifies the dataset checksum.
5. Trains the Random Forest model.
6. Runs automated tests.

**Verified CI result:** 28 passed, 0 failed.

[View GitHub Actions](https://github.com/patrickfelder777-cpu/employee-attrition-mlops/actions)

## Project Structure

```text
employee-attrition-mlops/
├── .github/workflows/ci.yml
├── configs/
├── data/
│   └── raw/
├── models/
├── reports/
├── src/
│   ├── api.py
│   ├── train.py
│   ├── preprocessing.py
│   └── monitor_drift.py
├── tests/
├── MONITORING.md
├── requirements.txt
└── README.md
```

## Future Improvements

- Configure a shared DVC remote for easier dataset retrieval.
- Add API request and response examples.
- Expand model monitoring and add monitoring checks to CI.
- Compare model performance with baseline classifiers.
- Evaluate model stability using cross-validation.
- Address dependency deprecation warnings.
- Review and document the dataset's redistribution license.
