# Predictive Maintenance: Machine-Failure Classification

This is a completed **MODEL ARENA Track 1** submission project. It predicts the binary `Machine failure` outcome from a machine's type and operating measurements. It is an offline, reproducible ML workflow—not a production monitoring system.

## Data and class imbalance

`data/train.csv` has 7,999 observations. Inputs are `Type`, air/process temperature, rotational speed, torque, and tool wear; `Machine failure` is the positive class (`1`). The observed failure prevalence is 271/7,999 (3.39%). This imbalance makes accuracy misleading: a model that always predicts healthy is already 96.61% accurate, while its Average Precision is only the prevalence. Selection therefore uses Average Precision (PR-AUC) and reports precision, recall, F1, and confusion matrices.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the evaluation

```powershell
python src\explore_data.py
python src\run_experiment.py
```

`run_experiment.py` makes one fixed 80/20 stratified split (`random_state=42`). It compares Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting using five-fold stratified out-of-fold development predictions. Each pipeline fits all imputation, scaling, and one-hot encoding inside the relevant training fold. The model with the best out-of-fold Average Precision is selected; its decision threshold is chosen by best out-of-fold F1 over 0.05–0.95. Only then is the selected pipeline fitted on the development partition and evaluated on the fixed holdout. Results are written to ignored `results/experiment.json`.

The holdout metrics are an independent estimate for this fixed split, but model selection and threshold search still add some development-selection optimism. Re-running experiments against the same holdout after making choices would invalidate its independence. For repeated iteration, use a new untouched test set or nested CV.

### Measured experiment (2026-10-18)

Using the checked-in dataset and global Python environment (`scikit-learn 1.9.1`), the reproducible run selected HistGradientBoosting by development out-of-fold Average Precision. Failure prevalence was 3.39% (the no-skill AP reference).

| Model | Development OOF AP | F1-selected threshold | Development OOF F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.4139 | 0.88 | 0.4350 |
| Random Forest | 0.7199 | 0.35 | 0.6809 |
| Extra Trees | 0.6413 | 0.54 | 0.5982 |
| HistGradientBoosting | **0.7853** | **0.26** | **0.7378** |

The selected model was refit on the 6,399-row development partition and evaluated once against the fixed 1,600-row holdout (54 failures): AP **0.8086**, F1 **0.7193**, precision **0.6833**, recall **0.7593**, confusion matrix `[[1527, 19], [13, 41]]` (rows actual healthy/failure; columns predicted healthy/failure).

## Export and infer

After evaluating the holdout once:

```powershell
python src\train_final.py
python src\predict.py new_measurements.csv --output predictions.csv
```

The deployment artifact is excluded from Git in `models/`. It stores the fitted pipeline, feature order, model name, threshold, positive-class definition, seed, and training row count. `train_final.py` refits on all labelled data for deployment; it must not be described as a holdout evaluation.

The inference CSV must contain exactly these columns (any order is accepted): `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. Types must be `L`, `M`, or `H`; numeric values must be finite; rotational speed must be positive; tool wear must be non-negative. Output contains `failure_probability` and `predicted_failure`.

## Track 1 hidden-test submission

The organizers evaluate a hidden `test.csv` with the same six feature columns and no target. First export the model, then create a submission file:

```powershell
python src\train_final.py
python src\make_submission.py path\to\test.csv --output submission.csv
```

`submission.csv` contains `Machine failure probability` (the positive-class score appropriate for PR-AUC) and `Machine failure` (the thresholded class at the development-selected F1 threshold). Confirm the organizer's required output-column convention before submitting; use the probability column whenever it requests a PR-AUC prediction score.

The concise hackathon write-up, including EDA findings, validation design, local metrics, practical implications, and limitations, is in [METHODOLOGY.md](<D:\My Codings\First Hackathon\DEV VS codes\DEVS Club\METHODOLOGY.md>).

## Tests

```powershell
python -m pytest -q
```

Tests cover schema validation, artifact loading, reordered input columns, output shape, probability bounds, and malformed input.

## Limitations

The random split assumes rows are independent and identically distributed. The dataset carries no timestamp, machine identifier, operating site, or source provenance, so this cannot test temporal, machine-group, or distribution-shift generalization. The threshold is optimized for F1; a real maintenance policy should select it from the relative costs of missed failures and unnecessary inspections.
