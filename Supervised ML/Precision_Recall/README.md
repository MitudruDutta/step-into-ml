# 🎯 Precision, Recall, F1 (and More): Evaluating Classifiers

## Introduction

Accuracy alone can be misleading—especially when classes are imbalanced. This module focuses on **Precision**, **Recall**, **F1**, and related concepts, using a simple car ownership prediction task. You’ll also learn when to use each metric, how to tune thresholds, and how to compare models with **Precision–Recall curves**.

## 🧠 Core Concepts

### Confusion Matrix (Binary)

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| Actual 0       | TN          | FP          |
| Actual 1       | FN          | TP          |

Definitions:

-   TP: Predicted positive & actually positive
-   FP: Predicted positive but actually negative (false alarm)
-   FN: Predicted negative but actually positive (miss)
-   TN: Predicted negative & actually negative

### Key Metrics

```
Precision = TP / (TP + FP)
Recall (Sensitivity, TPR) = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall) = 2TP / (2TP + FP + FN)
Specificity (TNR) = TN / (TN + FP)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Fβ = (1 + β²) * (P * R) / (β² * P + R)
```

-   F2 (β=2) weights Recall higher; F0.5 weights Precision higher.
-   Use Fβ when domain costs are asymmetric.

### Precision vs Recall Trade-off

Lowering the decision threshold (for probability-based models) raises Recall but may lower Precision; raising threshold often increases Precision but lowers Recall. There is no universal best threshold—choose based on domain cost or optimize a target metric (e.g., F1, expected profit).

### Multi-Class Averaging Strategies

When extending to multi-class:

-   macro: Unweighted mean over classes (treat all equally).
-   weighted: Mean weighted by class support (accounts for imbalance but can hide poor minority performance).
-   micro: Global TP/FP/FN counts (good for overall performance, dominated by majority class).
-   samples: For multi-label datasets (averages per instance).

(scikit-learn: `f1_score(y_true, y_pred, average='macro')` etc.)

### Precision–Recall (PR) Curve

Plots Precision vs Recall across all probability thresholds.

-   More informative than ROC when there is **strong class imbalance** and positive class is rare.
-   A high area under the PR curve (PR AUC) indicates both high Recall and high Precision at various thresholds.

### ROC Curve (Contrast)

ROC plots TPR vs FPR. Under heavy imbalance, a model can achieve a high ROC AUC even if Precision is poor. Prefer PR curves for rare-event detection.

### When to Use Which

| Scenario | Prefer |
|----------|--------|
| Rare event detection (fraud, disease) | Recall, PR Curve, F2 |
| Costly false positives (e.g., blocking legit users) | Precision, F0.5 |
| Balanced need | F1 |
| Highly imbalanced, ranking quality | PR AUC + ROC AUC |
| Probability calibration needed | Log Loss + Calibration curves |

## 📊 Dataset

-   **File**: `car_ownership.csv`
-   Features: `monthly_salary`
-   Target: `owns_car` (1 = owns, 0 = not) (Simple binary dataset for metric illustration)

## 🛠 Implementation Outline

1.  Load dataset & inspect class balance.
2.  Train logistic regression (or other classifier).
3.  Generate predictions & predicted probabilities.
4.  Compute confusion matrix, classification report.
5.  Plot Precision–Recall curve & threshold vs F1.
6.  Select operating threshold aligned with business goal.
7.  (Optional) Compare against baseline (e.g., always predict majority class).

## 🧪 Minimal Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, f1_score, auc, average_precision_score)
import matplotlib.pyplot as plt
import numpy as np

# Load
df = pd.read_csv('car_ownership.csv')
X = df[['monthly_salary']]
y = df['owns_car']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

# Default threshold 0.5
y_pred = (proba >= 0.5).astype(int)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Precision–Recall curve
prec, rec, thr = precision_recall_curve(y_test, proba)
pr_auc = auc(rec, prec)
avg_prec = average_precision_score(y_test, proba)
print(f'PR AUC (trapezoid) = {pr_auc:.3f} | Average Precision = {avg_prec:.3f}')

plt.figure(figsize=(5,4))
plt.plot(rec, prec, label=f'PR Curve (AP={avg_prec:.2f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curve')
plt.legend(); plt.tight_layout(); plt.show()

# Threshold sweep (optimize F1)
thresholds = np.linspace(0.05, 0.95, 19)
vals = []
for t in thresholds:
    pred_t = (proba >= t).astype(int)
    vals.append((t, f1_score(y_test, pred_t)))

best_t, best_f1 = max(vals, key=lambda x: x[1])
print(f'Best threshold (F1): {best_t:.2f} -> F1={best_f1:.3f}')
```

## 🔧 Interpreting Results

-   If Precision is high but Recall low → model conservative; missing positives.
-   If Recall high but Precision low → many false alarms; threshold likely too low.
-   Flat PR curve near the baseline (positive prevalence) indicates weak model.

Baseline Precision (no-skill) equals positive class prevalence: `P = positives / total`. Any useful classifier’s PR curve should exceed this horizontal line.

## 🧮 Fβ Example

```python
from sklearn.metrics import fbeta_score
for beta in [0.5, 1, 2]:
    print(f'F{beta}:', fbeta_score(y_test, y_pred, beta=beta))
```

Choose β based on weighting false negatives vs false positives.

## 📐 Averaging (Multi-Class / Multi-Label)

If extended to multi-class car ownership scenarios (hypothetical):

-   Use `average=` parameter consistently; report macro + weighted + micro to surface imbalance impacts.
-   For multi-label tasks, also consider `subset accuracy` (strict) vs averaged F1 (lenient).

## ⚠️ Common Pitfalls

-   Reporting only Accuracy on imbalanced data.
-   Comparing F1 across datasets with very different class prevalence (context matters).
-   Choosing threshold on test set (data snooping). Use validation split for tuning.
-   Ignoring calibration; high Precision doesn’t imply well-calibrated probabilities.
-   Over-optimizing a single metric without stakeholder cost alignment.

## 🔄 Extensions

-   Add ROC curve & compute ROC AUC for comparison.
-   Calibrate probabilities: `CalibratedClassifierCV` (Platt / isotonic).
-   Use cost-sensitive evaluation: expected cost = FP_cost * FP + FN_cost * FN.
-   Explore **PR-Gain** curves for better visual separation near extremes.
-   Apply **SMOTE** / class weighting to handle severe imbalance.
-   Deploy threshold optimization with custom utility functions.

## ✅ Key Takeaways

-   Precision & Recall capture complementary error types (false alarms vs misses).
-   F1 (or Fβ) summarizes trade-off, but threshold choice must reflect real costs.
-   PR curves are superior to ROC when positives are rare.
-   Always separate threshold tuning (validation) from final evaluation (test).

## 📂 Files

-   `precision_recall.ipynb`: Notebook with full code & plots.
-   `car_ownership.csv`: Dataset.
-   `precision_recall.xlsx`: (Optional) manual calculations / notes.

---

## 🧾 Summary

You computed precision, recall, F1, and PR curves, tuned decision thresholds, and learned when to prefer PR over ROC. With these tools you can responsibly evaluate and deploy classifiers in imbalanced, cost-sensitive domains.
