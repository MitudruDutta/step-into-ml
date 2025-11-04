# 🔪 Train-Test Split: The Foundation of Model Evaluation

## Introduction

The **Train-Test Split** is one of the most fundamental and critical concepts in machine learning. It partitions a dataset into two (or more) disjoint subsets:

1.  **Training Set** – used to fit the model.
2.  **Test Set** – used only at the end to estimate generalization to unseen data.

This module demonstrates these ideas by predicting a car's fuel efficiency (MPG). We also extend beyond the basics to cover validation sets, cross‑validation, leakage, stratification, and time‑series considerations.

## 🧠 Why Split the Data?

A model evaluated on the same data it was trained on can appear deceptively good because it has partially memorized patterns (overfitting). Holding out data provides an unbiased proxy for future performance. This enforces the separation between: (1) learning the parameters, (2) tuning model choices, and (3) assessing final performance.

### Typical Dataset Workflow

```
Raw Data
  └─ Train Set  ──► Model fit + hyperparameter tuning (with internal CV or validation)
  └─ Test Set   ──► Final, once-only evaluation (no tuning allowed)
```

### Train / Validation / Test (When Needed)

For small projects a single train/test split is enough. For model selection:

-   **Train**: Fit parameters.
-   **Validation**: Tune hyperparameters / choose model variant.
-   **Test**: Report final, unbiased metric (used once).

Cross‑validation can replace a distinct validation set when data is limited.

## 📏 Choosing Split Ratios

Common defaults:

-   80/20 (train/test) – general starting point.
-   70/15/15 (train/val/test) – for larger datasets.
-   90/10 – if data is scarce and model choice is simple.

Heuristic: ensure the test set is large enough to give a stable estimate (hundreds of samples if possible; for very small data consider repeated CV instead of a single split).

## ⚠️ Data Leakage

**Data leakage** occurs when information from the test (or future) data influences the model during training. Examples:

-   Scaling using the entire dataset before splitting.
-   Feature engineering with target statistics computed globally.
-   Time-series shuffling, destroying temporal ordering.

Mitigation: perform preprocessing inside a **Pipeline** after splitting (or using cross‑validation utilities that fit only on training folds).

## 🔁 Cross-Validation (Brief Overview)

Instead of one test split, **k-fold CV** rotates the validation fold k times. Useful for stable performance estimation on small datasets. Still keep a final holdout test if you intend to publish or finalize a model.

## 🕒 Time-Series Exception

Do NOT randomly shuffle chronological data. Use time-aware splits (train on earlier timestamps, test on later). Techniques: expanding window, rolling window, `TimeSeriesSplit` in scikit‑learn.

## 🎯 Stratification (Classification)

For classification tasks with imbalanced classes, use `train_test_split(..., stratify=y)` to preserve class proportions in both splits. (Not applied here because MPG regression target is continuous.)

## 🧮 Core Regression Metrics (Used After Split)

```
Residual:   e_i = y_i − ŷ_i
MSE:        (1/n) Σ e_i²
RMSE:       √MSE
MAE:        (1/n) Σ |e_i|
R²:         1 − Σ e_i² / Σ (y_i − ȳ)²
```

Pick RMSE when you want to penalize large errors more, MAE when robustness to outliers matters, and R² for variance explanation (compare models of same target only).

## 🧪 Minimal Example (scikit-learn)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel (requires openpyxl)
df = pd.read_excel('mpg.xlsx')

# Basic cleaning example (drop rows with NA target)
df = df.dropna(subset=['mpg'])

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
X = df[features]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}  |  R²: {r2:.3f}")
```

### With a Pipeline (if scaling or encoding needed)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LinearRegression())
])
pipe.fit(X_train, y_train)
```

Pipeline ensures transformations are fit only on training data, reducing leakage.

## 🔍 Validating Split Quality

-   Check distribution shift: compare summary stats between train and test.
-   Ensure no duplicate leakage (e.g., same entity appears in both sets when independence is violated).
-   Re-run with different `random_state` seeds to confirm metrics stability.

## ⚠️ Common Pitfalls

-   Performing feature scaling / encoding before splitting.
-   Using test set repeatedly for hyperparameter tuning.
-   Ignoring temporal ordering in time-series data.
-   Very small test size (unstable metric) or very large test size (under-trained model).
-   Data imbalance ignored in classification (lack of stratification).

## 🔄 Extensions

-   k-fold cross‑validation (`cross_val_score`) for robust estimates.
-   Nested CV for unbiased hyperparameter performance estimation.
-   Repeated train/test splits (Monte Carlo CV) for variance assessment.
-   TimeSeriesSplit for sequence data.
-   Group-aware splitting (`GroupKFold`) to prevent entity leakage.

## 📊 Dataset

-   **File**: `mpg.xlsx` (Auto MPG dataset)
-   **Features**: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`
-   **Target**: `mpg`

## 🛠 Implementation Steps (Recap)

1.  Load & clean dataset.
2.  Separate features (X) and target (y).
3.  Perform train/test split with fixed `random_state` for reproducibility.
4.  Fit model on training data only.
5.  Evaluate on test data using appropriate metrics.
6.  (Optional) Introduce validation or cross‑validation for hyperparameter tuning.

## ✅ Key Takeaways

-   A proper split is the first guard against overfitting.
-   Keep the test set isolated until the very end.
-   Use pipelines to prevent preprocessing leakage.
-   Adjust strategy for time series, imbalanced classes, or grouped data.

## 📂 Files

-   `train_test_split.ipynb`: Hands-on notebook.
-   `mpg.xlsx`: Dataset.

---

## 🧾 Summary

You learned how and why to partition data, avoid leakage, choose metrics, and extend the basic split to more advanced validation strategies (CV, time-aware splits, stratification). Mastering this foundation ensures every subsequent model experiment is trustworthy.
