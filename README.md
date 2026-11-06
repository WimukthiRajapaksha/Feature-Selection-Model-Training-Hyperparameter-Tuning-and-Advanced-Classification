# Breast Cancer Classification: Feature Analysis, Model Training, Hyperparameter Tuning, and Ensembles

An end-to-end supervised learning workflow that classifies breast tissue samples as **malignant** or **benign**, comparing tuned classical models, a neural network, and ensemble methods. It was completed as Assignment 2 for an MSc Applied AI & ML course.

**Result:** AdaBoost performed best in 10-fold cross-validation, with 97.5% accuracy, 96.7% recall, and 0.995 ROC-AUC. It was the only Part 2 model to reach the 95% recall target. These figures come from cross-validation on the training data, not a held-out test set (see [Limitations](#limitations)).

## The problem

In the assignment's scenario, a pathology lab has built a faster but slightly noisier way to measure the features in the Wisconsin breast cancer dataset. The lab has hand-labelled 300 samples and needs a model that meets two requirements:

1. Detect at least **95%** of malignant cases (recall ≥ 0.95).
2. Label no more than **1 in 10** benign cases as malignant (false positive rate ≤ 0.10).

## Dataset

| Property | Value |
| --- | --- |
| Samples | 300 (146 malignant, 154 benign) |
| Features | 30 numeric nuclei measurements |
| Feature groups | Mean, standard error, and "worst" (largest) value of 10 measurements |
| Missing values | None |
| Split | 240 train, 30 validation, 30 test (stratified) |

> **The dataset is not in this repository.** To run the notebook, place `assignment2.csv` next to it. The first column must be `label` (`benign` or `malignant`), followed by the 30 feature columns.

## Workflow

| Step | Task | Approach |
| --- | --- | --- |
| 1–2 | Load and explore | Plot class-separated histograms for each feature group and rank features by absolute correlation with the label |
| 3 | Split the data | 80/10/10 stratified split with a fixed seed |
| 4–5 | Build pipelines | Mean imputation and standard scaling, followed by each classifier |
| 6 | Tune hyperparameters | `GridSearchCV` with 5 folds, scored on accuracy, for Decision Tree, Random Forest, and SVM |
| 7–8 | Evaluate and select | Compare models on the validation set, then fit the best one (Random Forest) as the Part 1 final model |
| 9 | Predict new data | Return the predicted class and probability of malignancy for a new patient sample |
| 10 | Neural network | PyTorch network with 128, 64, and 32 ReLU hidden units, trained with Adam for 50 epochs per fold |
| 11 | Ensembles | Random Forest (bagging), AdaBoost, and XGBoost |
| 12 | Compare | Summarise the Part 2 models, discuss trade-offs, and make a recommendation |

The three features most correlated with the label are `worst concave points`, `worst perimeter`, and `worst radius`. This ranking is exploratory: all 30 features are used for training.

## Results

### Part 1: tuned models on the validation set

Results are from 30 validation samples (14 malignant, 16 benign). Precision, recall, and F1 are for the malignant class.

| Model | Best hyperparameters | Accuracy | Precision | Recall | F1 | Missed malignant | False alarms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | `n_estimators=200`, `max_depth=None`, `min_samples_split=2` | 0.97 | 1.00 | 0.93 | 0.96 | 1 | 0 |
| Decision Tree | `max_depth=None`, `min_samples_split=2` | 0.93 | 1.00 | 0.86 | 0.92 | 2 | 0 |
| SVM | `C=10`, `gamma='scale'`, `kernel='rbf'` | 0.93 | 0.93 | 0.93 | 0.93 | 1 | 1 |

With only 14 malignant samples, each missed case lowers recall by about 7 percentage points, so these results are a rough guide.

### Part 2: 10-fold cross-validation on the training set

Metrics are averaged across folds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **AdaBoost** | **0.975** | **0.984** | **0.967** | **0.975** | **0.995** |
| Neural Network | 0.958 | 0.955 | 0.941 | 0.948 | 0.988 |
| XGBoost | 0.938 | 0.935 | 0.942 | 0.937 | 0.990 |
| Random Forest (bagging) | 0.938 | 0.935 | 0.942 | 0.937 | 0.982 |

### Recommendation

**AdaBoost** is the recommended model. It scored highest on every metric and was the only Part 2 model above the 95% recall requirement. The notebook doesn't calculate the false positive rate directly, but AdaBoost's 0.984 precision suggests it is well under the 10% limit.

## Limitations

- **The recommended model has no held-out score.** The 30-sample test set is created but never evaluated, and AdaBoost's figures come from cross-validation on the training data. It should be scored on the test set, including its false positive rate, before any claim that it meets the requirements.
- **The new-patient prediction doesn't use AdaBoost.** Step 9 uses the Part 1 Random Forest, which predicted benign with an 8% probability of malignancy.
- **The comparison is not fully like for like.** The neural network uses shuffled `KFold`, while the other models use the stratified 10-fold default. Its scaler is also fitted on the full training set before cross-validation, which leaks a small amount of information into each fold.
- **Neural network results can vary.** PyTorch isn't seeded, so its metrics may change slightly between runs.
- **The sample is small.** Model differences of a few percentage points represent only a handful of samples.

This is coursework and is not a validated clinical tool.

## Getting started

The saved outputs were produced with Python 3.12.7. The versions below come from the Anaconda environment used for this project.

```bash
pip install scikit-learn==1.5.1 pandas==2.2.2 numpy==1.26.4 matplotlib==3.9.2 torch==2.5.1 xgboost==3.2.0 jupyter
jupyter notebook Assignment2_template.ipynb
```

Keep scikit-learn below 1.6, because the notebook uses the `needs_proba` argument of `make_scorer`, which scikit-learn's deprecation warning says is removed in 1.6.

## Repository contents

| File | Description |
| --- | --- |
| [`Assignment2_template.ipynb`](Assignment2_template.ipynb) | Assignment brief, code, saved outputs, and discussion |
| `README.md` | Project overview |
