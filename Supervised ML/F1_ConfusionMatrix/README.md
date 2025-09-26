# 📊 Confusion Matrix and F1-Score: A Deep Dive

## Introduction

This project serves as a practical guide to two of the most important tools for evaluating a classification model: the **Confusion Matrix** and the **F1-Score**. While accuracy tells you the overall percentage of correct predictions, these metrics give you a much more nuanced understanding of your model's performance, especially when dealing with imbalanced datasets.

We will explore these concepts by building a model to predict car ownership.

## 🧠 Theory

### The Confusion Matrix: A Detailed Look

The Confusion Matrix is the starting point for most classification evaluation. It's a table that breaks down the predictions made by a classifier and compares them to the actual outcomes.

|                | **Predicted: Negative** | **Predicted: Positive** |
|----------------|-------------------------|-------------------------|
| **Actual: Negative** | True Negative (TN)      | False Positive (FP)     |
| **Actual: Positive** | False Negative (FN)     | True Positive (TP)      |

- **True Positives (TP)**: The model correctly predicted the positive class.
- **True Negatives (TN)**: The model correctly predicted the negative class.
- **False Positives (FP)**: The model incorrectly predicted the positive class (Type I Error).
- **False Negatives (FN)**: The model incorrectly predicted the negative class (Type II Error).

### The F1-Score: Balancing Precision and Recall

While Precision and Recall are excellent metrics, you often need to balance them. The **F1-Score** is the **harmonic mean** of Precision and Recall, providing a single score that represents both.

- **F1 Formula**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- High only when **both** Precision and Recall are high.
- Useful under **class imbalance** or **uneven error costs**.

### 🔢 Key Formulas (Binary Case)

```
Precision = TP / (TP + FP)
Recall (Sensitivity, TPR) = TP / (TP + FN)
Specificity (TNR) = TN / (TN + FP)
F1 = 2 * TP / (2TP + FP + FN)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```
- Use **F2** when Recall is more important (missed positives costly).
- Use **F0.5** when Precision is more important (false alarms costly).

### 🧮 Averaging Strategies (Multi-class / Imbalanced Sets)
If you extend this to multi-class classification:
- **macro**: unweighted mean of per-class metrics (treats all classes equally).
- **weighted**: mean weighted by class support (accounts for imbalance but can hide poor minority performance).
- **micro**: aggregates global TP/FP/FN before computing metrics (good for overall performance under imbalance).

In scikit-learn: `f1_score(y_true, y_pred, average='macro')`

### 🎯 When to Use Which Metric
- **Accuracy**: Classes balanced and equal error cost.
- **Precision** important: False positives expensive (e.g., flagging legitimate users as fraud).
- **Recall** important: False negatives expensive (e.g., missing a disease case).
- **F1**: Need a single score balancing Precision & Recall.
- **Fβ**: Domain prioritizes one error type (β > 1 for Recall focus, β < 1 for Precision focus).

### ⚖️ Threshold Tuning
Logistic regression (and many classifiers) output probabilities. The default 0.5 cutoff may not be optimal.

Basic threshold sweep example:
```python
from sklearn.metrics import f1_score
import numpy as np

proba = model.predict_proba(X_valid)[:, 1]
thresholds = np.linspace(0.1, 0.9, 17)
results = [(t, f1_score(y_valid, (proba >= t).astype(int))) for t in thresholds]
best_t, best_f1 = max(results, key=lambda x: x[1])
print(best_t, best_f1)
```

### 🧵 Example (Core Pattern)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv("car_ownership.csv")
X = df[["monthly_salary"]]
y = df["owns_car"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
```

### 🧪 Interpreting the Confusion Matrix
If you see (for example):
```
[[40  5]
 [ 7 28]]
```
- TN = 40, FP = 5, FN = 7, TP = 28
- Recall = 28 / (28 + 7) ≈ 0.80
- Precision = 28 / (28 + 5) ≈ 0.85
- F1 ≈ 0.82 (balanced strength)

---

## 📊 Dataset

- **File**: `car_ownership.csv`
- **Description**: A simple binary classification dataset.
  - `monthly_salary`: The monthly salary of an individual.
  - `owns_car`: The target variable (1 if they own a car, 0 if they do not).

## 🛠 Implementation Steps

1. **Load Data**: The `car_ownership.csv` dataset is loaded.
2. **Model Training**: A `LogisticRegression` model is trained on the data.
3. **Generate and Visualize the Confusion Matrix**: A confusion matrix is created and plotted with clear labels to visualize the model's predictions.
4. **Interpret the F1-Score**: The `classification_report` is used to generate the F1-score, and its meaning is explained in the context of the problem.
5. (Optional) **Tune Threshold**: Explore trade-offs between Precision and Recall.

## 🚀 Running the Notebook (Quickstart)
From the project root (after installing dependencies as per top-level README):

```batch
jupyter notebook Supervised%20ML/F1_ConfusionMatrix/f1_confusion_matrix.ipynb
```
Or open it through the Jupyter UI.

Dependencies (already in `requirements.txt`): pandas, scikit-learn, matplotlib, seaborn.

## 🧩 Common Pitfalls
- Reporting only Accuracy on imbalanced data.
- Ignoring threshold tuning when business costs differ.
- Comparing F1 scores across datasets with different class balance (avoid—context matters).
- Using weighted F1 and assuming minority class handled well (inspect per-class report!).

## 🔄 Possible Extensions
- Plot Precision-Recall curve (`from sklearn.metrics import precision_recall_curve`).
- Add ROC & AUC comparison.
- Introduce F2 vs F0.5 for domain-driven emphasis.
- Handle synthetic imbalance with `class_weight='balanced'` or resampling.

## 📂 Files

- `f1_confusion_matrix.ipynb`: The Jupyter Notebook with the code and detailed explanations.
- `car_ownership.csv`: The dataset used for the demonstration.
- `f1_confusion_matrix.xlsx`: A spreadsheet that may contain manual calculations or explanations.

---

## ✅ Summary
Use the confusion matrix to understand error types; use Precision/Recall to quantify type-specific performance; use F1 (or Fβ) when you need a single, balanced metric—especially under imbalance.
