# ⚖️ Stratified K-Fold Cross-Validation: Balanced Splits for Classification

## Introduction
Stratified K-Fold Cross-Validation preserves the class distribution in each fold. It’s the preferred choice for classification problems, especially when classes are imbalanced. Each fold gets approximately the same proportion of each class as the full dataset, reducing variance and giving a more reliable performance estimate than plain K-Fold on imbalanced data.

This module shows how to:
- Use `StratifiedKFold` with `cross_val_score`
- Compare multiple classifiers under stratified CV
- Evaluate multiple metrics with `cross_validate` (e.g., accuracy, F1, ROC AUC)
- Iterate manually over folds when you need custom logic

## What’s inside
- `stratifiedkfoldcross.ipynb`: Notebook demonstrating stratified CV and model comparisons.
- `README.md`: This guide.

## Quick start (Windows, cmd)
If you use a virtual environment, activate it first; then install dependencies and open the notebook.

```cmd
:: (optional) create a virtual environment
python -m venv .venv

:: (optional) activate it
.venv\Scripts\activate

:: install project requirements from repository root
pip install -r requirements.txt

:: launch Jupyter and open stratifiedkfoldcross.ipynb
jupyter notebook
```

## Minimal example (binary, imbalanced)
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) Create an imbalanced dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # 10% positives
    random_state=42
)

# 2) Stratified CV (preserves class ratios per fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) Evaluate logistic regression with accuracy
clf = LogisticRegression(max_iter=1000)
acc = cross_val_score(clf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print('Fold accuracy:', np.round(acc, 3))
print('Mean accuracy:', acc.mean())
```

Tip: For imbalanced data, accuracy can be misleading. Add a recall- or F1-oriented score as well.

## Multi-metric evaluation (F1 and ROC AUC)
```python
from sklearn.model_selection import cross_validate

clf = LogisticRegression(max_iter=1000)
cv_res = cross_validate(
    clf, X, y,
    cv=skf,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=False,
    n_jobs=-1
)
print('Accuracy (test):', cv_res['test_accuracy'].mean())
print('F1        (test):', cv_res['test_f1'].mean())
print('ROC AUC   (test):', cv_res['test_roc_auc'].mean())
```

## Compare multiple classifiers
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogReg': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}

for name, est in models.items():
    f1 = cross_val_score(est, X, y, cv=skf, scoring='f1', n_jobs=-1)
    print(f'{name}: F1 mean={f1.mean():.3f}, std={f1.std():.3f}')
```

## Manual split pattern (when you need custom logic)
```python
from sklearn.metrics import f1_score

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Fold F1:', f1_score(y_test, y_pred))
```

## Practical tips and pitfalls
- Metric choice: For imbalance, report `f1`, `recall`, `precision`, and `roc_auc` alongside accuracy.
- Shuffle + seed: Use `shuffle=True` with a `random_state` for reproducibility.
- Pipelines: Prevent data leakage by wrapping preprocessing inside a `Pipeline` so it’s fit within each fold.
- Class rarity: Extremely rare positive class may cause fold(s) with too few positives. Consider fewer folds or `StratifiedKFold(n_splits=5)` to keep at least one positive per fold.
- Repeats: Use `RepeatedStratifiedKFold` to average over multiple random splits for more stable estimates.
- Multiclass: `StratifiedKFold` supports multiclass by preserving per-class proportions across folds.
- Groups: If you have groups (e.g., users, sessions) that must not be split, use `StratifiedGroupKFold` or `GroupKFold` instead of plain stratification.

## Pipeline example (scale-sensitive models)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

acc = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print('Pipeline mean accuracy:', acc.mean())
```

## Expected output (example)
- Per-fold scores and mean values for selected metrics (F1, ROC AUC, accuracy).
- Comparable mean F1s across models to pick a baseline.

## Next steps
- Add PR AUC (Average Precision) if you care about ranking under imbalance.
- Try `GridSearchCV`/`RandomizedSearchCV` with `StratifiedKFold` for hyperparameter tuning.
- Evaluate calibration (`CalibratedClassifierCV`) if probability quality matters.

## Files
- `stratifiedkfoldcross.ipynb`: End-to-end stratified CV examples and model comparison.
- `README.md`: You are here.

