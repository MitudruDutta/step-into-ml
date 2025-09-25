# 🎯 Precision, Recall, and F1-Score: Evaluating Classifiers

## Introduction

In classification tasks, **accuracy** (the percentage of correct predictions) is often not enough to truly understand a model's performance. This is especially true for datasets with a **class imbalance** (where one class is much more frequent than the other). **Precision**, **Recall**, and the **F1-Score** are more informative metrics that provide a deeper insight into a model's strengths and weaknesses.

This project explains these concepts by building a classifier to predict car ownership and analyzing its performance using these advanced metrics.

## 🧠 Theory

### The Confusion Matrix

Everything starts with the **Confusion Matrix**, a table that summarizes the performance of a classification model.

|                | **Predicted: NO** | **Predicted: YES** |
|----------------|-------------------|--------------------|
| **Actual: NO** | True Negative (TN)  | False Positive (FP) |
| **Actual: YES**| False Negative (FN) | True Positive (TP)  |

-   **True Positives (TP)**: Correctly predicted positive cases.
-   **True Negatives (TN)**: Correctly predicted negative cases.
-   **False Positives (FP)**: Incorrectly predicted positive cases (a "false alarm").
-   **False Negatives (FN)**: Incorrectly predicted negative cases (a "miss").

### Precision

Precision answers the question: **"Of all the positive predictions the model made, how many were actually correct?"**

-   **Formula**: `Precision = TP / (TP + FP)`
-   **Use Case**: High precision is important when the cost of a False Positive is high. For example, in spam detection, you want to be very sure that an email is spam before you send it to the spam folder (you don't want to misclassify an important email).

### Recall (Sensitivity)

Recall answers the question: **"Of all the actual positive cases, how many did the model correctly identify?"**

-   **Formula**: `Recall = TP / (TP + FN)`
-   **Use Case**: High recall is important when the cost of a False Negative is high. For example, in medical diagnosis for a serious disease, you want to identify all patients who actually have the disease, even if it means some healthy patients are incorrectly flagged for more tests.

### F1-Score

The F1-Score is the **harmonic mean** of Precision and Recall. It provides a single metric that balances both concerns.

-   **Formula**: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
-   **Use Case**: It is a good general-purpose metric for evaluating a classifier's overall performance, especially when you have an uneven class distribution.

## 📊 Dataset

-   **File**: `car_ownership.csv`
-   **Description**: A simple binary classification dataset.
    -   `monthly_salary`: The monthly salary of an individual.
    -   `owns_car`: The target variable (1 if they own a car, 0 if they do not).

## 🛠 Implementation Steps

1.  **Load Data**: The `car_ownership.csv` dataset is loaded.
2.  **Model Training**: A `LogisticRegression` model is trained on the data.
3.  **Generate a Confusion Matrix**: A confusion matrix is created and visualized to see the counts of TP, TN, FP, and FN.
4.  **Generate a Classification Report**: `scikit-learn`'s `classification_report` is used to automatically calculate and display the precision, recall, and F1-score for each class.

## 📂 Files

-   `precision_recall.ipynb`: The Jupyter Notebook with the code and detailed explanations.
-   `car_ownership.csv`: The dataset used for the demonstration.
-   `precision_recall.xlsx`: A spreadsheet that may contain manual calculations or explanations.
