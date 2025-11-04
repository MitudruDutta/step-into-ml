# 🔁 K-Fold Cross-Validation: Robust Model Evaluation

## Introduction
K-Fold Cross-Validation (CV) is a reliable way to estimate a model's out-of-sample performance by splitting the data into K folds, training on K-1 folds, and validating on the remaining fold—repeating this K times and averaging the scores.

This module shows how to use KFold with scikit-learn to:
- Train and evaluate with `cross_val_score`
- Compare multiple models (Logistic Regression, Decision Tree, Random Forest)
- Collect multiple metrics with `cross_validate` (e.g., accuracy, ROC AUC)

All examples use a synthetic dataset from `make_classification`, so no external data files are required.

## What’s inside
- `kfold.ipynb`: A notebook demonstrating KFold, cross-validation scores, and model comparison.
- `README.md`: This guide.

## Quick start (Windows, cmd)
Optional: create and activate a virtual environment, then install dependencies.

```cmd
:: (optional) create a virtual environment
python -m venv .venv

:: (optional) activate it
.venv\Scripts\activate

:: install project requirements
pip install -r requirements.txt

:: launch Jupyter and open kfold.ipynb
jupyter notebook
```

## Minimal example
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# 1) Data
X, y = make_classification(
    n_features=10,
    n_samples=1000,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# 2) CV splitter (shuffle for i.i.d. ordering and reproducibility)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3) Cross-validation for Logistic Regression
clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
print('Fold accuracies:', np.round(scores, 3))
print('Mean accuracy:', scores.mean())
```

Tip: Many scikit-learn solvers need a few more iterations on some datasets—setting `max_iter=1000` avoids convergence warnings.

## Compare multiple models
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogReg': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

for name, est in models.items():
    scores = cross_val_score(est, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
    print(f'{name}: mean={scores.mean():.3f}, std={scores.std():.3f}')
```

## Collect multiple metrics
```python
from sklearn.model_selection import cross_validate

# For ROC AUC, estimator must support predict_proba or decision_function
est = DecisionTreeClassifier(random_state=42)
cv_res = cross_validate(
    est, X, y, cv=kf,
    scoring=['accuracy', 'roc_auc'],
    return_train_score=False,
    n_jobs=-1
)
print('Accuracy (test):', cv_res['test_accuracy'].mean())
print('ROC AUC  (test):', cv_res['test_roc_auc'].mean())
```

## Manual split example (correct pattern)
If you need manual train/test indices from KFold, split on the full dataset (or its index), not on a toy list:
```python
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # fit/score per fold here
```

## Tips and pitfalls
- Shuffle: Use `shuffle=True` when data might have any order (time, grouping). Keep `random_state` for reproducibility.
- Stratification: For classification with class imbalance, prefer `StratifiedKFold` to preserve label ratios per fold.
- Data leakage: Wrap preprocessing in `Pipeline` so it’s fit inside each fold, not on the full dataset.
- Scoring: Choose metrics aligned with your goal (e.g., `roc_auc` for ranking, `f1` for imbalanced precision/recall balance).
- Variance: Report both mean and std across folds; tight std indicates stable performance.
- Speed: Use `n_jobs=-1` on CV utilities and parallelizable estimators to leverage all cores.

## Expected output (example)
- 5 fold accuracies printed and their mean (varies with random seed and model).
- For multi-metric evaluation, mean Accuracy and ROC AUC across folds.

## Next steps
- Swap in `StratifiedKFold` for imbalanced classification.
- Add more metrics: `precision`, `recall`, `f1`, `average_precision`.
- Use `Pipeline` + `StandardScaler` for models sensitive to feature scales.
- Try hyperparameter search with CV: `GridSearchCV` or `RandomizedSearchCV`.

## Files
- `kfold.ipynb`: End-to-end K-Fold examples and model comparison.
- `README.md`: You are here.

