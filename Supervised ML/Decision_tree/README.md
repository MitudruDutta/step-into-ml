# 🌳 Decision Tree Classifier: Predicting Salaries

## Introduction

A Decision Tree is a versatile and intuitive supervised learning algorithm for **classification** and **regression**. It recursively partitions feature space into regions that are (ideally) pure with respect to the target. Each internal node tests a feature; each leaf outputs a class (or distribution / value).

This project predicts whether a person's salary exceeds $100k based on company, job role, and degree.

## 🧠 Theory

### Key Concepts

-   **Root Node**: Represents the full training set before any split.
-   **Internal Node**: A decision rule on a single feature (e.g., job == 'sales executive').
-   **Branch**: Outcome of a decision (True/False or categorical value path).
-   **Leaf Node**: Terminal node containing class prediction (often majority class of samples reaching it) and class distribution.

### Impurity Measures

For classification, splits aim to reduce impurity. Common criteria:

-   **Gini Impurity**: `Gini = 1 - Σ p_i^2`
-   **Entropy**: `H = - Σ p_i log2 p_i`
-   **Information Gain** (Entropy criterion): `IG = H(parent) - Σ (n_k / n_parent) * H(child_k)`

The tree chooses the feature & threshold (or category split) that maximizes impurity reduction.

### Stopping & Overfitting

Unrestricted trees can overfit (memorize noise). Control complexity with hyperparameters:

-   `max_depth`: Maximum tree depth.
-   `min_samples_split`: Minimum samples to attempt a further split.
-   `min_samples_leaf`: Minimum samples required in a leaf (promotes smoothing).
-   `max_features`: Number of features to consider at each split (introduces randomness; used in ensembles).
-   `ccp_alpha`: Cost Complexity Pruning parameter (post-pruning).

### Pruning (Cost Complexity)

Compute `R(T) + α * |leaves(T)|`; increasing α penalizes more leaves. Use `DecisionTreeClassifier(ccp_alpha=...)` or derive candidate α values via `cost_complexity_pruning_path` then cross‑validate.

### Handling Categorical Features

scikit-learn decision trees require numeric encodings:

-   **One-Hot Encoding** (retain interpretability; can increase dimensionality).
-   **Ordinal / Label Encoding** (fast, but imposes artificial order—acceptable only if categories are unordered and tree can still split effectively on equality; one-hot preferred for small category counts).

### Prediction

At inference, an instance travels down the tree following decision rules until reaching a leaf whose class distribution is converted to probabilities; the class with highest probability is predicted.

## 📊 Dataset

-   **File**: `salaries.csv`
-   **Columns**:
    -   `company`: e.g., google, facebook.
    -   `job`: job title.
    -   `degree`: education level.
    -   `salary_more_then_100k`: target (1 if > 100k else 0).

## 🛠 Implementation Steps

1.  Load dataset with pandas.
2.  Inspect class balance & category levels.
3.  Encode categorical features (One-Hot via `OneHotEncoder`).
4.  Split into train/test sets.
5.  Train baseline `DecisionTreeClassifier`.
6.  Evaluate (accuracy, confusion matrix, classification report).
7.  Tune depth / leaf constraints to mitigate overfitting.
8.  (Optional) Prune via cost complexity path.
9.  Visualize tree / feature importances.

## 🧪 Minimal Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('salaries.csv')
X = df[['company', 'job', 'degree']]
y = df['salary_more_then_100k']

cat_features = X.columns.tolist()
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

clf = Pipeline([
    ('prep', preprocess),
    ('tree', DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_leaf=2))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Access underlying tree & feature names for visualization
model = clf.named_steps['tree']
feature_names = clf.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features)
plt.figure(figsize=(14,6))
plot_tree(model, feature_names=feature_names, class_names=['<=100k','>100k'], filled=True, max_depth=3, fontsize=8)
plt.tight_layout(); plt.show()
```

### Cost Complexity Pruning Example
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Fit preprocessing separately to get encoded matrix
X_train_enc = preprocess.fit_transform(X_train)

# Derive pruning path on an unpruned tree
dt = DecisionTreeClassifier(random_state=42)
path = dt.cost_complexity_pruning_path(X_train_enc, y_train)
ccp_alphas = path.ccp_alphas

cv_scores = []
for alpha in ccp_alphas:
    dt_alpha = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    scores = cross_val_score(dt_alpha, X_train_enc, y_train, cv=5, scoring='accuracy')
    cv_scores.append((alpha, scores.mean()))

best_alpha, best_score = max(cv_scores, key=lambda t: t[1])
print(f"Best alpha={best_alpha:.5f}, CV Accuracy={best_score:.3f}")
```

## 🔍 Feature Importance
`model.feature_importances_` gives normalized importance (sum=1) based on impurity reduction. Beware bias toward high-cardinality or continuous-like (after encoding) features.

## ⚠️ Common Pitfalls
-   Overfitting due to unrestricted depth.
-   Treating feature importances as causal evidence.
-   Using label encoding for high-cardinality unordered categories (splits may become inefficient).
-   Ignoring class imbalance (tune `class_weight` if needed).
-   Large, unpruned trees = poor interpretability & unstable generalization.

## 🔄 Extensions
-   **Ensembles**: Random Forest (bagging), Gradient Boosted Trees (sequential), XGBoost / LightGBM / CatBoost.
-   **Calibration**: Calibrate probabilities (trees can output poorly calibrated probabilities).
-   **Reduced Error Pruning / Minimal Cost-Complexity**: Explore pruning systematically.
-   **Permutation Importance**: More robust importance than impurity-based.
-   **SHAP Values**: For local explanation of predictions.

## ✅ Key Takeaways
-   Decision trees partition feature space with axis-aligned rules, optimizing impurity reduction.
-   They are interpretable but prone to overfitting without constraints.
-   Proper preprocessing (encoding) and pruning / hyperparameter tuning are essential.
-   Often best used as components inside ensemble methods for superior performance.

## 📂 Files
-   `decisiontree.ipynb`: Notebook with full workflow.
-   `salaries.csv`: Dataset.

---
## 🧾 Summary
You trained and interpreted a decision tree classifier, learned impurity-based splitting, applied encoding, controlled complexity to reduce overfitting, and identified paths toward more powerful ensemble methods.
