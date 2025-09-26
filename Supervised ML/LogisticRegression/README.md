# 📖 Logistic Regression: Predicting Car Ownership

## Introduction

Logistic Regression is a foundational classification algorithm used to predict a **binary outcome** (0/1). Unlike linear regression (unbounded outputs), logistic regression maps inputs to a probability in (0,1) using the **sigmoid (logistic) function**. This project predicts whether a person owns a car based on monthly salary.

---

## 🧠 Theory

### 1. Linear Score → Probability

```
Linear score (logit):  z = β₀ + β₁ x
Sigmoid:               σ(z) = 1 / (1 + e^(−z))
Predicted probability: p = P(y=1 | x) = σ(z)
Decision (default):    ŷ = 1  if  p ≥ 0.5  else 0
```

### 2. Log-Odds Interpretation

```
log( p / (1 − p) ) = β₀ + β₁ x
```

β₁ is the change in **log-odds** per unit increase in x. Exp(β₁) is the **odds ratio**.

### 3. Maximum Likelihood & Loss

For a dataset (xᵢ, yᵢ), yᵢ ∈ {0,1}:

```
Likelihood:      Π pᵢ^{yᵢ} (1−pᵢ)^{1−yᵢ}
Log-Likelihood:  Σ [ yᵢ log pᵢ + (1−yᵢ) log (1−pᵢ) ]
Loss (minimize): Binary Cross-Entropy (Log Loss)
L = −(1/n) Σ [ yᵢ log pᵢ + (1−yᵢ) log (1−pᵢ) ]
```

No closed-form solution → use gradient-based optimization (e.g., LBFGS, SAG, liblinear).

### 4. Regularization

Adds penalty to prevent overfitting (especially with multiple features):

```
L_reg = LogLoss + α * (   ||β||₁  )   (L1 / Lasso)
L_reg = LogLoss + α * (1/2)||β||²₂   (L2 / Ridge)
```

scikit-learn uses parameter C = 1/α (inverse strength). Smaller C ⇒ stronger regularization.

### 5. Multiclass Extensions

- One-vs-Rest (OvR): default for binary & optional for multiclass.
- Multinomial (softmax): joint optimization over all classes.
  Set `multi_class='multinomial'` with a suitable solver (e.g., `lbfgs`).

### 6. Threshold Tuning

Default threshold = 0.5. Adjust to balance Precision vs Recall depending on business cost.

### 7. Imbalanced Data

Accuracy may mislead. Prefer metrics: Precision, Recall, F1, ROC AUC, PR AUC. Use:

- `class_weight='balanced'`
- Resampling (oversample minority / undersample majority)
- Threshold optimization via F1 / cost curve

---

## 📊 Dataset

- **File**: `car_ownership.csv`
- **Columns**:
  - `monthly_salary` (feature)
  - `owns_car` (target: 0/1)
    Single-feature example for clarity (naturally extends to multi-feature).

---

## 🛠 Implementation Steps

1. Load dataset.
2. Exploratory plot (salary vs ownership probability trend).
3. Split into train/test (stratify when possible for class balance).
4. Fit logistic regression.
5. Inspect coefficients & odds ratio.
6. Evaluate (confusion matrix, classification report, ROC AUC, PR curve).
7. Tune threshold if needed.
8. (Optional) Add regularization / multiple features.

---

## 🧪 Minimal Code (Binary Classification)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, auc)

# Load
df = pd.read_csv('car_ownership.csv')
X = df[['monthly_salary']]
y = df['owns_car']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(C=1.0, solver='lbfgs')
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, proba))

# Precision-Recall AUC (useful for imbalance)
prec, rec, thr = precision_recall_curve(y_test, proba)
pr_auc = auc(rec, prec)
print('PR AUC:', pr_auc)
```

### Threshold Optimization Example

```python
import numpy as np
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 17)
results = [(t, f1_score(y_test, (proba >= t).astype(int))) for t in thresholds]
best_t, best_f1 = max(results, key=lambda x: x[1])
print(f'Best threshold={best_t:.2f}, F1={best_f1:.3f}')
```

### Odds Ratio Interpretation

```python
coef = model.coef_[0][0]
odds_ratio = np.exp(coef)
print(f'β1={coef:.4f}, odds ratio={odds_ratio:.3f}')
```

If odds_ratio = 1.20 ⇒ each unit salary increase multiplies odds of ownership by 1.20 (holding others constant if multi-feature).

---

## 📊 Key Metrics (Binary)

| Metric    | Formula            | Notes                                      |
|-----------|--------------------|--------------------------------------------|
| Precision | TP / (TP+FP)       | Reliability of positive predictions         |
| Recall    | TP / (TP+FN)       | Coverage of actual positives               |
| F1        | 2PR/(P+R)          | Balance of Precision & Recall              |
| Accuracy  | (TP+TN)/(Total)    | Misleading if imbalance                     |
| ROC AUC   | Prob. rank quality | Threshold-invariant                         |
| PR AUC    | Area under P-R curve | Better under heavy imbalance               |
| Log Loss  | Cross-entropy      | Punishes overconfident wrong probs        |

---

## ⚠️ Common Pitfalls

- Relying only on accuracy with imbalanced classes.
- Ignoring probability calibration (check with reliability curves if used for risk scoring).
- Forgetting to scale when mixing heterogeneous magnitude features (for convergence & interpretability with regularization).
- Treating coefficients as causal effects (they are associational unless design ensures causality).
- Using default threshold instead of aligning with business cost function.

---

## 🔄 Extensions

- Add multiple predictors (categorical → one-hot encode via `OneHotEncoder` in a `ColumnTransformer`).
- Apply regularization (`penalty='l1'` with `solver='liblinear'` or `saga` for sparse coefficients).
- Multiclass: `multi_class='multinomial'`.
- Probability calibration: `CalibratedClassifierCV` (Platt scaling / isotonic).
- Handle imbalance: `class_weight='balanced'` or resampling pipelines.
- Compare with tree-based or ensemble classifiers.

---

## 🚀 Running the Notebook

From project root:

```batch
jupyter notebook "Supervised ML/LogisticRegression/logisticregression.ipynb"
```

Dependencies: pandas, scikit-learn, matplotlib, seaborn (already in requirements).

---

## ✅ Key Takeaways

- Logistic Regression models log-odds linearly & outputs calibrated (often decent) probabilities.
- Cross-entropy (log loss) drives probability-focused training; accuracy alone is insufficient.
- Threshold selection & class imbalance handling are crucial for real-world deployment.
- Regularization improves generalization; interpret odds ratios carefully.

---

## 📂 Files

- `logisticregression.ipynb`
- `car_ownership.csv`

---

## 🧾 Summary

You trained a probabilistic linear classifier, interpreted coefficients via odds, evaluated with multiple metrics, and explored threshold tuning & regularization strategies—forming a repeatable pattern for future classification tasks.
