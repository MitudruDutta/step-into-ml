# 📖 Logistic Regression: Predicting Car Ownership

## Introduction

Logistic Regression is a fundamental classification algorithm used to predict a binary outcome (e.g., yes/no, true/false, 1/0). Unlike linear regression, which predicts a continuous value, logistic regression predicts the *probability* that an instance belongs to a particular class.

This project provides a beginner-friendly guide to implementing logistic regression to predict whether a person owns a car based on their monthly salary.

## 🧠 Theory

### Key Concepts

Logistic Regression works by passing the output of a linear equation through a **Sigmoid Function** (or Logistic Function).

1.  **Linear Equation**: `z = mX + c`
    - This is the same as in linear regression.

2.  **Sigmoid Function**: `p = 1 / (1 + e^(-z))`
    - This function squashes the output of the linear equation (`z`) to a value between 0 and 1.
    - The result `p` can be interpreted as the probability of the positive class (e.g., the probability of owning a car).

A **decision boundary** (typically 0.5) is used to convert this probability into a class prediction. If `p >= 0.5`, the model predicts class 1; otherwise, it predicts class 0.

### Formula

- Linear equation: `z = β₀ + β₁x`
- Sigmoid function: `p = 1 / (1 + e^(−z))`
- Decision rule: If `p ≥ 0.5`, predict class 1; else class 0.

## 📊 Dataset

-   **File**: `car_ownership.csv`
-   **Description**: A simple dataset with two columns:
    -   `monthly_salary`: The monthly salary of an individual (Independent Variable).
    -   `owns_car`: A binary variable indicating car ownership (1 if they own a car, 0 if they do not) (Dependent Variable).

## 🛠 Implementation Steps

1.  **Load Data**: The `car_ownership.csv` dataset is loaded using pandas.
2.  **Exploratory Data Analysis (EDA)**: A scatter plot is used to visualize the relationship between salary and car ownership.
3.  **Data Preparation**: The data is split into a training set and a testing set.
4.  **Model Training**: A `LogisticRegression` model from `scikit-learn` is trained on the training data.
5.  **Model Evaluation**: The model's performance is evaluated using:
    -   **Accuracy Score**: The percentage of correct predictions.
    -   **Confusion Matrix**: A table showing the number of true positives, true negatives, false positives, and false negatives.
    -   **Classification Report**: A summary of precision, recall, and F1-score.
6.  **Prediction**: The trained model is used to predict car ownership for a new salary and to check the predicted probabilities.

## ✅ Key Takeaways

-   **Strengths**: Highly interpretable, computationally inexpensive, and provides probability scores for outcomes.
-   **Limitations**: Assumes a linear relationship between the features and the log-odds of the outcome. May not perform well if the decision boundary is non-linear.
-   **Use Cases**: Spam detection (spam vs. not spam), medical diagnosis (patient has disease vs. does not), and churn prediction (customer will churn vs. will not).

## 📂 Files

-   `logisticregression.ipynb`: The Jupyter Notebook with the Python code and explanations.
-   `car_ownership.csv`: The dataset used for the classification task.
