# 🛡️ L1 & L2 Regularization: Preventing Overfitting

## Introduction

**Regularization** is a set of techniques used to prevent **overfitting** in machine learning models. Overfitting occurs when a model learns the training data too well—including noise—causing poor generalization to unseen data.

This module compares the two most common penalties: **L2 (Ridge)** and **L1 (Lasso)** using a high‑dimensional synthetic regression dataset.

## 🧠 Theory

### The Problem of Overfitting

Too many or highly correlated features can let a model fit spurious patterns. OLS (ordinary least squares) has variance that grows when features are noisy or collinear.

### Core Idea

Add a penalty to the loss so that large coefficients are discouraged. This trades a little bias for lower variance.

### Objective Functions (with MSE loss)

Let X ∈ ℝ^{n×p}, y ∈ ℝ^n, β ∈ ℝ^p, predictions ŷ = Xβ.

```
MSE(β) = (1/n) ||y − Xβ||²₂
Ridge:  minimize  (1/2n) ||y − Xβ||²₂ + α ||β||²₂
Lasso:  minimize  (1/2n) ||y − Xβ||²₂ + α ||β||₁
Elastic Net: (hybrid) (1/2n)||y − Xβ||²₂ + α [ (1−l1_ratio)/2 ||β||²₂ + l1_ratio ||β||₁ ]
```

(Intercept is typically excluded from the penalty.)

### Closed Form (Ridge)

```
β̂_ridge = (Xᵀ X + 2nα I)⁻¹ Xᵀ y
```

(No closed form for Lasso / Elastic Net; solved via coordinate descent / proximal gradient.)

### Geometric Intuition

-   Ridge: L2 ball constraint → shrinks coefficients smoothly toward origin.
-   Lasso: L1 diamond constraint → corners encourage exact zeros (sparse solution).

### Effects Summary

| Aspect                                      | Ridge (L2)                     | Lasso (L1)                     |
|---------------------------------------------|---------------------------------|---------------------------------|
| Coefficient shrinkage                       | Smooth                          | Can become exactly 0           |
| Feature selection                           | No                              | Yes (implicit)                 |
| Stability w/ correlated features            | Distributes weight              | Arbitrarily selects one (or few) |
| Closed-form                                 | Yes                             | No                              |
| Good when                                   | Many small/true signals        | Few strong signals              |

### Multicollinearity

Ridge mitigates variance under multicollinearity (adds λI to XᵀX). Lasso may unpredictably keep one variable and drop others. Elastic Net combines both: keeps groups and still performs selection.

### Bias–Variance Tradeoff

Increasing α raises bias but lowers variance. There exists an optimal α (selected via cross‑validation) minimizing expected generalization error.

### Why Scaling Matters

Penalties act on raw coefficient magnitudes. Features on larger scales would be penalized less relatively; always scale (e.g., StandardScaler) before regularization (except tree models).

## 📊 Dataset

-   **File**: `dataset.csv`
-   **Features**: 150 numeric columns (`f1` … `f150`), many are noisy.
-   **Target**: Continuous variable constructed so only a subset influences y.
-   **Generator**: `dataset_generator.ipynb` (creates reproducible synthetic data).

## 🛠 Implementation Steps

1.  Load dataset.
2.  Train/test split (e.g., 80/20).
3.  Baseline OLS (`LinearRegression`) → observe potential overfitting (low train error, higher test error).
4.  Ridge model: tune α via cross‑validation.
5.  Lasso model: tune α; inspect sparsity (count non-zero coefficients).
6.  (Optional) Elastic Net: balance between L1 and L2.
7.  Plot coefficient magnitudes for OLS vs Ridge vs Lasso.
8.  Compare metrics (RMSE / MAE / R²) on test set.

## 🧪 Minimal Code (Ridge & Lasso with Pipelines)

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load
df = pd.read_csv('dataset.csv')
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
ridge_params = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge_pipe, ridge_params, cv=5, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X_train, y_train)

# Lasso
lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=5000))
])
lasso_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso_pipe, lasso_params, cv=5, scoring='neg_root_mean_squared_error')
lasso_cv.fit(X_train, y_train)

# Evaluate
for name, est in [('Ridge', ridge_cv), ('Lasso', lasso_cv)]:
    y_pred = est.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: best_alpha={est.best_params_['model__alpha']} RMSE={rmse:.3f} R²={r2:.3f}")

# Sparsity (Lasso)
lasso_best = lasso_cv.best_estimator_.named_steps['model']
nonzero = (lasso_best.coef_ != 0).sum()
print(f"Lasso non-zero coefficients: {nonzero}/{len(lasso_best.coef_)}")
```

## 🔍 Coefficient Path (Optional Visualization)

You can trace how coefficients shrink as α grows using `sklearn.linear_model.lasso_path` or `ridge_path` (manually iterating). Useful for identifying stability.

## ⚠️ Common Pitfalls

-   Skipping feature scaling → distorted penalty impact.
-   Using Lasso when features are highly correlated → unstable selection.
-   Relying only on R²; prefer RMSE/MAE and cross‑validated performance.
-   Too large α → underfitting (coefficients ~0).
-   Interpreting shrunk coefficients causally without domain validation.

## 🔄 Extensions

-   **Elastic Net**: Middle ground (helps with correlated groups). `ElasticNet(alpha=..., l1_ratio=...)`.
-   **Cross-Validation variants**: `RidgeCV`, `LassoCV`, `ElasticNetCV` (built-in efficient solvers).
-   **Feature importance pruning**: After Lasso, retrain a clean Ridge/OLS on selected features.
-   **Stability selection**: Run Lasso with bootstraps to find consistently selected features.
-   **Generalized Linear Models**: Extend regularization to logistic / Poisson models.

## ✅ Key Takeaways

-   Regularization combats variance by penalizing coefficient magnitude.
-   Ridge keeps all features (good for many weak signals, multicollinearity).
-   Lasso performs embedded feature selection (sparse solutions).
-   Elastic Net blends both to stabilize selection when predictors are correlated.
-   Proper α chosen via cross‑validation is critical.

## 📂 Files

-   `l1l2regularization.ipynb`: Notebook with full workflows & plots.
-   `dataset.csv`: High-dimensional synthetic dataset.
-   `dataset_generator.ipynb`: Data creation process (reproducibility).

---

## 🧾 Summary

You compared OLS, Ridge, and Lasso on a high-dimensional regression task, observed variance reduction and sparsity effects, and learned how tuning and scaling enable robust generalization. Next steps: explore Elastic Net, model pipelines, and feature stability analyses.
