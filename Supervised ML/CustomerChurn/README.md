# 📈 Customer Churn Prediction: An End-to-End Classification Project

## Introduction

**Customer Churn** is a critical business metric that refers to the rate at which customers stop doing business with a company. Predicting which customers are likely to churn is a common and high-impact problem that can be solved with machine learning.

This project provides an end-to-end walkthrough of building a churn prediction model. It covers the entire machine learning lifecycle, from data exploration and preprocessing to handling class imbalance and model evaluation.

## 🧠 Theory

### The Problem of Class Imbalance

In churn datasets, the number of customers who churn is typically much smaller than the number of customers who do not. This is known as **class imbalance**. If not handled properly, a model trained on imbalanced data will be biased towards the majority class and will perform poorly at identifying the minority class (the churners), which is the very thing we want to predict.

### Techniques to Handle Imbalance

1.  **Undersampling**: This method reduces the number of instances in the majority class to balance it with the minority class. A common technique is **Random Undersampling**.
    -   **Pros**: Can help improve runtime and reduce storage.
    -   **Cons**: Can lead to loss of important information from the majority class.

2.  **Oversampling**: This method increases the number of instances in the minority class. The most popular technique is **SMOTE** (Synthetic Minority Over-sampling Technique), which creates new, synthetic data points for the minority class.
    -   **Pros**: No information loss.
    -   **Cons**: Can increase the likelihood of overfitting.

(You can also explore **class_weight='balanced'**, **ADASYN**, or **ensemble methods** as future extensions.)

## 📊 Dataset

- **File**: `churn.csv`
- **Description**: A dataset containing various attributes of customers of a telecommunications company.
  - **Features** (raw columns as in CSV — note the embedded spaces):
    - `Call  Failure`, `Complains`, `Subscription  Length`, `Charge  Amount`, `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`, `Age Group`, `Tariff Plan`, `Status`, `Age`, `Customer Value`
  - **Target**: `Churn` (1 if the customer churned, 0 otherwise).

### 🧹 Recommended Column Renaming (for cleaner code)
```python
cols = {
    'Call  Failure':'call_failure',
    'Complains':'complains',
    'Subscription  Length':'subscription_length',
    'Charge  Amount':'charge_amount',
    'Seconds of Use':'seconds_of_use',
    'Frequency of use':'frequency_use',
    'Frequency of SMS':'frequency_sms',
    'Distinct Called Numbers':'distinct_called_numbers',
    'Age Group':'age_group',
    'Tariff Plan':'tariff_plan',
    'Status':'status',
    'Age':'age',
    'Customer Value':'customer_value',
    'Churn':'churn'
}

df = pd.read_csv('churn.csv').rename(columns=cols)
```

### 🔎 Inspect Class Distribution
```python
import pandas as pd
from collections import Counter

churn_counts = Counter(df['churn'])
print(churn_counts)
print({k: f"{v/len(df):.2%}" for k, v in churn_counts.items()})
```
Use this before and after resampling to verify balancing.

## 🛠 Implementation Steps

1. **Exploratory Data Analysis (EDA)**: Inspect shapes, nulls, class distribution, basic correlations.
2. **Preprocessing**:
   - Rename columns (optional, for convenience).
   - Encode categorical columns (`tariff_plan`, maybe `age_group` if categorical codes not numeric semantics).
   - Scale features if using distance-based models (not strictly required for logistic regression, but helpful if comparing to SVM / KNN later).
3. **Baseline Model**: Train `LogisticRegression` on imbalanced data (record Precision, Recall, F1, ROC-AUC, PR-AUC for the minority class).
4. **Undersampling**: Use `RandomUnderSampler` on training split only; retrain and evaluate.
5. **Oversampling (SMOTE)**: Apply **only to X_train / y_train** (never to the test set); retrain and evaluate.
6. **Threshold Tuning**: Optimize decision threshold for a cost-sensitive metric (e.g., maximize Recall at acceptable Precision, or maximize F1 / Youden’s J).
7. **Model Comparison**: Tabulate metrics across strategies.
8. **Interpretability**: Examine logistic coefficients (after scaling, if applied) to understand direction/magnitude.
9. **Reporting**: Include confusion matrices and classification reports.

## 📏 Evaluation Metrics Focus

Given churn is usually rare, default **accuracy can be misleading**.

Key metrics:
- **Recall (Sensitivity)**: TP / (TP + FN) — ability to catch churners (critical for retention).
- **Precision**: TP / (TP + FP) — proportion of flagged churners who truly churn.
- **F1-Score**: Harmonic mean of Precision & Recall (balance).
- **ROC-AUC**: Ranking quality across thresholds.
- **PR-AUC (Average Precision)**: More informative under heavy class imbalance.

### 📌 Sample Metric Computation
```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

proba = model.predict_proba(X_test)[:,1]
y_pred = (proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=3))
print('ROC-AUC:', roc_auc_score(y_test, proba))
print('PR-AUC:', average_precision_score(y_test, proba))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

### 🎚 Threshold Tuning Example
```python
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

thresholds = np.linspace(0.1, 0.9, 81)
rows = []
for t in thresholds:
    yp = (proba >= t).astype(int)
    rows.append((t, recall_score(y_test, yp), precision_score(y_test, yp), f1_score(y_test, yp)))

best = max(rows, key=lambda r: r[3])   # by F1
print(f"Best threshold={best[0]:.2f} Recall={best[1]:.3f} Precision={best[2]:.3f} F1={best[3]:.3f}")
```

## ✅ Key Takeaways

- Handling class imbalance is a critical step in building a useful churn model.
- Undersampling + Oversampling each have trade-offs; try both (and maybe hybrid).
- Optimize for **Recall** (catch at-risk customers) but monitor Precision to avoid alert fatigue.
- Threshold tuning often yields larger gains than swapping algorithms early.
- Always evaluate with metrics calculated on the untouched test set.

## 🧪 Minimal Pattern (Pseudocode)
```python
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)

# Resample example
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_res, y_res)
```

## ⚠️ Common Pitfalls
- Fitting SMOTE / undersampling on the full dataset (data leakage). Always resample only training data.
- Reporting post-resampling class balance without clarifying it applies only to training set.
- Comparing models on resampled test sets (never alter the test set!).
- Ignoring calibration — if using probabilities for retention budget allocation, check calibration curve.
- Not logging the original churn rate (baseline prevalence is important context).

## 🔄 Possible Extensions
- Add **calibration** (`CalibratedClassifierCV`) for better probability quality.
- Try **XGBoost / LightGBM** (often strong on tabular churn data).
- Use **cross-validation with StratifiedKFold** for more stable estimates.
- Create a **cost matrix** (e.g., cost_miss = 10× cost_false_alarm) and choose threshold minimizing expected cost.
- Add **feature importance** (model coefficients or permutation importance).
- Plot **Precision-Recall** and **ROC** curves.

## 🚀 Running the Notebook
From project root (after installing dependencies as per top-level README):
```batch
jupyter notebook "Supervised ML/CustomerChurn/churnprediction.ipynb"
```
Or open via Jupyter Lab/Notebook interface.

## 🧩 Files

- `churnprediction.ipynb`: The Jupyter Notebook with the full, end-to-end Python code and explanations.
- `churn.csv`: The dataset used for the project.

## 🧾 Summary
This module demonstrates building a churn classifier with a focus on handling class imbalance and evaluating beyond accuracy. Emphasis is placed on Recall, F1, PR-AUC, and threshold tuning to align the model with retention goals. The workflow is extensible to more advanced models and cost-sensitive decision making.
