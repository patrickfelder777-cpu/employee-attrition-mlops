# Model Monitoring and Drift Analysis

## Monitoring approach

This project uses Evidently to compare the reference training data with simulated production data.

The monitoring pipeline evaluates feature distributions and generates an HTML drift report at:

`reports/data_drift_report.html`

The configured drift-share threshold is 10%.

## Drift results

The latest monitoring run evaluated 34 features.

- Drifted features: 0
- Overall drift share: 0.00%
- Configured threshold: 10.00%
- Monitoring status: Passed

No individual features were identified as drifted during the latest run.

## Potential impact on model performance

Because no statistically significant feature drift was detected, the production data currently appears similar to the reference data used during model development.

This suggests that the model should continue receiving data from distributions similar to those seen during training. However, the absence of detected drift does not guarantee that model accuracy, precision, recall, or other performance metrics have remained unchanged.

Model performance could still decline because of target drift, changes in employee behavior, changes in business processes, or changes in the relationship between the input features and employee attrition.

## Recommended action

No immediate retraining is required based on the current drift results.

The recommended action is to continue monitoring production data regularly and investigate the model when:

- The drift share exceeds the configured 10% threshold.
- Important features begin showing repeated drift.
- Model performance metrics decline.
- Business conditions or employee policies change significantly.

When meaningful drift is detected, the model should be evaluated using recent labeled data and retrained if performance has deteriorated.