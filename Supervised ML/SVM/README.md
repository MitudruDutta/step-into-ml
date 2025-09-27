# 🍇 Support Vector Machine (SVM): Classifying Raisin Types

## Introduction

Support Vector Machine (SVM) is a powerful supervised learning algorithm for classification and regression. In classification, the core idea is to find an optimal **hyperplane** that separates classes with the **maximum margin**. This example classifies two types of raisins (Kecimen vs Besni) from morphological measurements.

## 🧠 Core Concepts

- **Hyperplane**: Decision boundary between classes.
- **Support Vectors**: Training samples that lie closest to the hyperplane; they determine the boundary.
- **Margin**: Distance between the hyperplane and the nearest support vectors (SVM maximizes this).
- **Kernel Trick**: Implicitly maps inputs into a higher‑dimensional feature space so a linear separator can work there.
  - `linear` – for (almost) linearly separable data
  - `rbf` – default, flexible for non‑linear patterns
  - `poly` – polynomial interactions
  - `sigmoid` – behaves like a neural activation; rarely the best, but illustrative

## 📊 Dataset

- **File**: `Raisin_Dataset.xlsx`
- **Samples**: 900 (two classes)
- **Features**: `Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`, `Extent`, `Perimeter`
- **Target**: `Class` (`Kecimen`, `Besni`)

## 🔧 Why Scaling Matters
SVM (especially with RBF / polynomial kernels) is sensitive to feature scales. Always apply feature scaling (e.g., `StandardScaler`). In the original minimal notebook cells, scaling was not yet applied—consider using a Pipeline (recommended version below).

## 🚀 Minimal Working Example (Recommended)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_excel('Raisin_Dataset.xlsx')
X = df[['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=10))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## 🔁 Comparing Kernels
```python
from sklearn.metrics import accuracy_score
kernels = ["linear", "rbf", "poly", "sigmoid"]
results = []
for k in kernels:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=k, probability=False, random_state=10))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    results.append((k, acc))

for k, acc in results:
    print(f"Kernel={k:<7} | Accuracy={acc:.4f}")
```

## 🎯 Evaluation Metrics
Typical metrics you may compute:
- **Accuracy**: Overall correctness.
- **Precision / Recall / F1**: For class balance evaluation.
- **Confusion Matrix**: Breakdown of predictions.
- **ROC AUC** (enable `probability=True` when constructing `SVC`).

Example for ROC AUC (binary classes):
```python
from sklearn.metrics import roc_auc_score, roc_curve
prob_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel='rbf', probability=True, random_state=10))
])
prob_model.fit(X_train, y_train)
probs = prob_model.predict_proba(X_test)[:, 1]  # Probability of class (alphabetically second label)
roc_auc = roc_auc_score(y_test, probs)
print("ROC AUC:", roc_auc)
```
Note: Ensure classes are encoded correctly; `roc_auc_score` treats the positive class as the lexicographically larger label unless you convert labels.

## 🔍 Hyperparameter Tuning (Grid Search)
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 50],
    'svc__gamma': ['scale', 0.01, 0.001],
    'svc__kernel': ['rbf', 'linear']
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=10))
])

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best F1 (macro):", grid.best_score_)
```

## 📈 Interpreting Model Complexity
Inspect support vectors:
```python
best_model = grid.best_estimator_
svc = best_model.named_steps['svc']
print("Number of support vectors per class:", svc.n_support_)
```

## ⚠️ Common Pitfalls
- Forgetting to scale features (hurts performance).
- Using `probability=True` unnecessarily (adds cross‑validation cost internally); enable only if you need calibrated probabilities / ROC curves.
- Relying only on accuracy with class imbalance.
- Neglecting `stratify=y` in the train/test split.

## 🧪 Reproducibility
Set `random_state` for deterministic splits and SVM reproducibility (affects probability estimates and some kernels).

## 📂 Files
- `SVM.ipynb`: Notebook with kernel comparisons (consider updating it to include scaling as shown above).
- `Raisin_Dataset.xlsx`: Input dataset.

## ✅ Next Suggested Improvements
- Add the scaling + pipeline flow into the notebook cells (current notebook trains kernels without scaling).
- Log performance metrics in a comparative table.
- Add ROC curve visualization when probabilities are enabled.
- Persist the trained model with `joblib` for reuse.

## 📎 Reference
- Scikit-learn SVM documentation: https://scikit-learn.org/stable/modules/svm.html

---
Feel free to extend this example with feature importance approximations (e.g., permutation importance) or dimensionality reduction (PCA) before SVM for visualization.
