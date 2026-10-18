# Track 1 Methodology Summary — Machine Failure Prediction

## Objective

The Track 1 objective is to identify imminent equipment failures in precision-manufacturing telemetry while limiting unnecessary alarms. The model consumes machine type, air/process temperature, rotational speed, torque, and cumulative tool wear, and predicts `Machine failure` (`0` healthy, `1` failure).

## Exploratory findings

The supplied training set contains 7,999 labelled observations and six input features. There are 271 failures (3.39%) and 7,728 healthy observations (96.61%); there are no missing values or duplicate rows. Machine types are distributed as 4,816 L, 2,393 M, and 790 H. This strong imbalance means accuracy is inappropriate as a primary metric: a model predicting every row as healthy would be 96.61% accurate yet would identify no failure. We therefore use Average Precision (the standard PR-AUC summary) for ranking models and precision, recall, F1, and a confusion matrix for thresholded decisions.

## Pipeline and validation

All preprocessing is inside sklearn `Pipeline` and `ColumnTransformer` objects. Numeric features are median-imputed; Logistic Regression is standardized; `Type` is most-frequent-imputed and one-hot encoded with unknown-category handling. This prevents fold leakage because transformers are fitted only on each cross-validation training fold.

We create one reproducible 80/20 stratified split with seed 42, reserving 1,600 observations (54 failures) as a final holdout. On the remaining 6,399 development rows, five-fold stratified cross-validation produces out-of-fold probabilities for Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting. All candidate models see the identical development/holdout boundary. Model choice is based on development OOF Average Precision, not accuracy.

The decision threshold is chosen only after model selection, using the selected model's development OOF probabilities. We search thresholds from 0.05 to 0.95 in steps of 0.01 and select maximum F1. This mimics a maintenance-alert decision: a lower threshold catches more failures at the cost of more inspections.

## Local results

| Model | Development OOF AP | Selected threshold | Development OOF F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.4139 | 0.88 | 0.4350 |
| Random Forest | 0.7199 | 0.35 | 0.6809 |
| Extra Trees | 0.6413 | 0.54 | 0.5982 |
| HistGradientBoosting | **0.7853** | **0.26** | **0.7378** |

HistGradientBoosting was selected because it had the best OOF Average Precision. Fitting it on the development rows and evaluating the untouched fixed holdout at threshold 0.26 gave AP **0.8086**, F1 **0.7193**, precision **0.6833**, and recall **0.7593**. The holdout confusion matrix was `[[1527, 19], [13, 41]]`: 41 of 54 failures were identified, with 19 healthy machines flagged for inspection.

## Business interpretation and limitations

The model produces a continuous failure probability, so maintenance teams can rank machines by risk instead of treating every alert equally. The selected F1 threshold is a balanced default, not a universal operational policy: if missed failures are extremely costly, lower it to favor recall; if inspections are expensive, raise it to favor precision. The recorded local holdout is a useful benchmark, but repeated iteration against it would make it less independent. The data has no timestamps, machine IDs, sites, or failure-cause information, so random-row validation cannot establish temporal, machine-level, or cross-site robustness. Hidden-test performance may differ under distribution shift.

## Reproducibility and submission

Run `python src\\run_experiment.py` to reproduce development selection and local holdout reporting. Then run `python src\\train_final.py` to fit the chosen pipeline on all labelled data and write the ignored deployment artifact to `models\\machine_failure_model.joblib`. For organizer data, run `python src\\make_submission.py path\\to\\test.csv --output submission.csv`. The script validates the Track 1 schema and writes continuous failure probabilities plus thresholded predictions.
