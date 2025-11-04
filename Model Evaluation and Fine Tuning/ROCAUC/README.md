# 📈 ROC Curve & AUC: Ranking Classifier Performance

## Introduction
Receiver Operating Characteristic (ROC) curves and the Area Under the Curve (AUC) evaluate how well a classifier ranks positive examples ahead of negatives across all thresholds. This module covers generating a ROC curve for the car ownership example and interpreting AUC properly—highlighting when to complement it with Precision-Recall metrics.

## Key Concepts
```
TPR (Recall, Sensitivity) = TP / (TP + FN)
FPR (Fall-out)            = FP / (FP + TN)
ROC Curve: plot TPR vs FPR while sweeping threshold.
AUC: Probability a random positive scores higher than a random negative.
```
- AUC = 1.0 → perfect ranking
- AUC = 0.5 → random guessing
- AUC < 0.5 → model is inversely ranking (can flip sign)

## When ROC AUC Shines
- Balanced datasets
- Early-stage model comparison (ranking focus)
- Need threshold-independent evaluation

## When to Prefer PR AUC
- Strong class imbalance (rare positives)
- Action costs tied to positive predictions (need high precision)

## Dataset
Reuses `car_ownership.csv` (present in other modules, e.g. `Supervised ML/LogisticRegression/car_ownership.csv`). Copy or reference one instance:
- `monthly_salary`
- `owns_car` (0/1)

```python
# Example robust loader (tries multiple known paths)
import pathlib
candidates = [
    'Supervised ML/LogisticRegression/car_ownership.csv',
    'Supervised ML/Precision_Recall/car_ownership.csv',
    'Supervised ML/F1_ConfusionMatrix/car_ownership.csv'
]
for p in candidates:
    if pathlib.Path(p).exists():
        data_path = p; break
```

## Minimal Example
1. Train logistic regression.
2. Get predicted probabilities.
3. Compute `roc_curve` and `roc_auc_score`.
4. Plot curve and diagonal baseline.
5. (Optional) pick threshold via Youden's J (max TPR − FPR) or domain cost.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv(data_path)
X = df[['monthly_salary']]
y = df['owns_car']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, proba)
auc_val = roc_auc_score(y_test, proba)
print(f'ROC AUC: {auc_val:.3f}')

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'ROC (AUC={auc_val:.2f})', lw=2)
plt.plot([0,1],[0,1],'--', color='grey', label='Random (0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('ROC Curve')
plt.legend(); plt.tight_layout(); plt.show()
```

## Sample Threshold Selection (Youden's J)
```python
import numpy as np
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_thresh = thresholds[best_idx]
print('Best threshold (Youden J):', best_thresh, 'J=', j_scores[best_idx])
```

## Cost-Based Threshold (Example)
If FN cost = 5, FP cost = 1:
```python
costs = []
for thr in thresholds:
    pred = (proba >= thr).astype(int)
    TP = ((pred==1)&(y_test==1)).sum()
    FP = ((pred==1)&(y_test==0)).sum()
    FN = ((pred==0)&(y_test==1)).sum()
    total_cost = 1*FP + 5*FN
    costs.append((thr, total_cost))
print(min(costs, key=lambda x: x[1]))
```

## Complementary Precision–Recall Curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score
prec, rec, thr_pr = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f'PR (AP={ap:.2f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curve')
plt.legend(); plt.tight_layout(); plt.show()
```

## Multi-Class ROC (One-vs-Rest)
```python
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
# Suppose y_multiclass has classes {0,1,2}
Y = label_binarize(y_multiclass, classes=[0,1,2])  # shape (n_samples, 3)
proba_mc = clf.predict_proba(X_test)               # same shape
auc_macro = roc_auc_score(Y, proba_mc, average='macro', multi_class='ovr')
print('Macro ROC AUC:', auc_macro)
# Per-class curves
for i, cls in enumerate([0,1,2]):
    fpr_i, tpr_i, _ = roc_curve(Y[:, i], proba_mc[:, i])
    plt.plot(fpr_i, tpr_i, label=f'Class {cls}')
plt.plot([0,1],[0,1],'--', color='grey')
plt.legend(); plt.title('One-vs-Rest ROC Curves'); plt.show()
```

## Calibration & Probability Quality
High AUC ≠ calibrated probabilities.
```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--')
plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title('Calibration Curve'); plt.show()
```
If miscalibrated and probabilities matter, wrap model with `CalibratedClassifierCV` (method='isotonic' or 'sigmoid').

## Bootstrapped Confidence Interval for AUC
```python
import numpy as np
rng = np.random.default_rng(42)
aucs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    aucs.append(roc_auc_score(y_test.iloc[idx], proba[idx]))
ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
print(f'AUC 95% CI: {ci_low:.3f} – {ci_high:.3f}')
```

## Partial AUC (Low FPR Focus)
```python
import numpy as np
fpr_limit = 0.1
mask = fpr <= fpr_limit
# Normalize by fpr_limit to report scaled partial AUC (0–1)
partial_auc = np.trapz(tpr[mask], fpr[mask]) / fpr_limit
print('Partial AUC (FPR<=0.1):', partial_auc)
```

## Additional Metrics & Relations
- Balanced Accuracy = (TPR + TNR)/2
- G-Mean = sqrt(TPR * TNR)
- Youden's J = TPR − FPR

## Extensions
- Add PR curve & Average Precision (included above).
- Calibration curve + Brier score.
- Bootstrapped CI (provided) or DeLong test (external libs) for comparing AUCs.
- Cost curve / expected utility visualization.
- Threshold optimization using custom utility functions.

## Pitfalls
- AUC alone can mislead under heavy imbalance (report PR AUC too).
- Marginal AUC improvements may be statistically insignificant.
- Data leakage in preprocessing inflates AUC—use pipelines.
- High AUC with poor calibration may harm probability-based decisions.
- ROC curves can look similar; rely on zoomed low-FPR region when needed.

## Files
- `rocauc.ipynb`: Notebook with ROC generation & visualization.
- `README.md`: This file.

## Summary
ROC AUC provides a global view of ranking performance; pair it with PR AUC, calibration analysis, cost-aware threshold selection, and statistical uncertainty estimates to make deployment-grade decisions in both balanced and imbalanced settings.
