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

-   **True Positives (TP)**: The model correctly predicted the positive class.
-   **True Negatives (TN)**: The model correctly predicted the negative class.
-   **False Positives (FP)**: The model incorrectly predicted the positive class (Type I Error).
-   **False Negatives (FN)**: The model incorrectly predicted the negative class (Type II Error).

### The F1-Score: Balancing Precision and Recall

While Precision and Recall are excellent metrics, you often need to balance them. The **F1-Score** is the **harmonic mean** of Precision and Recall, providing a single score that represents both.

-   **Formula**: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

An F1-Score is a better measure than accuracy for imbalanced datasets because it takes into account both false positives and false negatives. It is high only when both precision and recall are high.

## 📊 Dataset

-   **File**: `car_ownership.csv`
-   **Description**: A simple binary classification dataset.
    -   `monthly_salary`: The monthly salary of an individual.
    -   `owns_car`: The target variable (1 if they own a car, 0 if they do not).

## 🛠 Implementation Steps

1.  **Load Data**: The `car_ownership.csv` dataset is loaded.
2.  **Model Training**: A `LogisticRegression` model is trained on the data.
3.  **Generate and Visualize the Confusion Matrix**: A confusion matrix is created and plotted with clear labels to visualize the model's predictions.
4.  **Interpret the F1-Score**: The `classification_report` is used to generate the F1-score, and its meaning is explained in the context of the problem.

## 📂 Files

-   `f1_confusion_matrix.ipynb`: The Jupyter Notebook with the code and detailed explanations.
-   `car_ownership.csv`: The dataset used for the demonstration.
-   `f1_confusion_matrix.xlsx`: A spreadsheet that may contain manual calculations or explanations.
