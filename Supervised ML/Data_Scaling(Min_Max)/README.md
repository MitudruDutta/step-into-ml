# 📏 Data Scaling: Min-Max and Standardization

## Introduction

**Data Scaling** is a crucial preprocessing step in machine learning. Many algorithms, especially those that rely on distance calculations (like k-NN, SVM) or gradient descent (like linear/logistic regression, neural networks), can perform poorly if input features are on very different scales.

This mini‑project demonstrates two core techniques:
1. **Min-Max Scaling** (Normalization)
2. **Standardization** (Z-score Normalization)

## 🧠 Theory

### Why is Scaling Necessary?
If one feature ranges 0–1 and another 0–100000, the larger‑range feature can dominate distance metrics and gradient magnitudes. Scaling ensures all features contribute more proportionally.

### Min-Max Scaling (Normalization)
Rescales each feature to a fixed range, usually [0, 1].
- **Formula**: `X_scaled = (X - X_min) / (X_max - X_min)`
- **Pros**: Preserves shape of original distribution; keeps values bounded.
- **Cons**: Very sensitive to outliers (min/max shift).
- **Good for**: Algorithms needing bounded features (e.g., neural nets with certain activations), or when original distribution shape matters.

### Standardization (Z-score)
Centers data (mean 0) and scales to unit variance.
- **Formula**: `X_scaled = (X - mean) / std`
- **Pros**: Less affected by outliers than Min-Max; commonly assumed by many linear models and SVM.
- **Cons**: Produces unbounded values.
- **Good for**: Most linear models, SVM, logistic regression, PCA.

### Important Principle: Fit on Train Only
Never compute scaling parameters on the full dataset before splitting. Fit scaler on `X_train`, then transform `X_train` and `X_test` separately to avoid data leakage.

## 📊 Dataset
- **File**: `Raisin_Dataset.xlsx`
- **Description**: Contains morphological measurements of raisins. Feature magnitudes differ (e.g., `Area` vs. `Eccentricity`), showing how scaling impacts SVM performance.

## 🛠 Implementation Outline
1. Load dataset with pandas.
2. Separate features/target.
3. Train/Test split.
4. Baseline model (e.g., SVM) without scaling.
5. Pipeline with `MinMaxScaler` + SVM.
6. Pipeline with `StandardScaler` + SVM.
7. Compare accuracy (and optionally precision/recall, F1) across versions.

## 🧪 Sample Code Snippet
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
# df = pd.read_excel('Raisin_Dataset.xlsx')
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipelines = {
    'baseline_svm': Pipeline([
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]),
    'minmax_svm': Pipeline([
        ('scaler', MinMaxScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ]),
    'standard_svm': Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(name, accuracy_score(y_test, preds))
```

## ✅ Interpreting Results
Typical pattern you may observe:
- Baseline (unscaled): noticeably lower accuracy / unstable performance.
- Min-Max: improved convergence and accuracy.
- Standardization: similar or slightly better than Min-Max for SVM (because SVM with RBF benefits from standardized feature variance).

Exact numbers depend on random state and parameter tuning.

## 🧩 Best Practices & Tips
- Use `Pipeline` so scaling happens inside cross-validation (prevents leakage).
- For models like tree-based methods (Decision Trees, Random Forest, Gradient Boosting), scaling usually isn’t necessary.
- If strong outliers exist, consider `RobustScaler` (uses median & IQR) or apply outlier handling first.
- After inverse transforming predictions (e.g., regression tasks), keep scaler objects to recover original scale with `scaler.inverse_transform()`.

## 📂 Files
- `scaling.ipynb`: End-to-end notebook with exploration, scaling, modeling, and comparison.
- `Raisin_Dataset.xlsx`: Raw dataset.

## 🚀 Next Steps
Try adding:
- `RobustScaler` comparison.
- PCA before SVM (after scaling).
- Hyperparameter tuning (`GridSearchCV`) within the pipeline.

---
Return to main index: [Root README](../../README.md)
