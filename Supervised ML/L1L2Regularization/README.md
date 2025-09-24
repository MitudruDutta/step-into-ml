# 🛡️ L1 & L2 Regularization: Preventing Overfitting

## Introduction

**Regularization** is a set of techniques used to prevent **overfitting** in machine learning models. Overfitting occurs when a model learns the training data too well, including its noise and outliers, which causes it to perform poorly on new, unseen data.

This project demonstrates the two most common types of regularization—L1 (Lasso) and L2 (Ridge)—by applying them to a linear regression problem with a large number of features.

## 🧠 Theory

### The Problem of Overfitting

In models like linear regression, overfitting often happens when there are too many features, or when the model is too complex. The model learns large coefficients for the features, making it highly sensitive to small changes in the input data.

### How Regularization Works

Regularization works by adding a **penalty term** to the model's cost function. This penalty discourages the model from learning overly complex patterns or assigning too much importance to any single feature.

#### L2 Regularization (Ridge Regression)

Ridge Regression adds a penalty equal to the **sum of the squared values of the coefficients** to the cost function.

-   **Cost Function**: `MSE + α * Σ(coefficient²)`
-   **Effect**: It forces the coefficients to be small, but not exactly zero. This reduces the model's complexity and helps it generalize better.
-   `α` (alpha) is the regularization parameter that controls the strength of the penalty.

#### L1 Regularization (Lasso Regression)

Lasso Regression adds a penalty equal to the **sum of the absolute values of the coefficients**.

-   **Cost Function**: `MSE + α * Σ(|coefficient|)`
-   **Effect**: It can shrink some coefficients to be **exactly zero**. This makes Lasso useful for **feature selection**, as it effectively removes irrelevant features from the model.

## 📊 Dataset

-   **File**: `dataset.csv`
-   **Description**: A synthetic dataset designed to demonstrate regularization.
    -   **Features**: 150 numerical features (`f1`, `f2`, ..., `f150`).
    -   **Target**: A continuous numerical target variable.

## 🛠 Implementation Steps

1.  **Load Data**: The `dataset.csv` is loaded.
2.  **Train-Test Split**: The data is split into training and testing sets.
3.  **Baseline Model**: A standard `LinearRegression` model is trained to serve as a baseline and demonstrate potential overfitting.
4.  **Ridge Regression (L2)**: A `Ridge` model is trained, and its performance is compared to the baseline.
5.  **Lasso Regression (L1)**: A `Lasso` model is trained, and its performance and feature selection capabilities are analyzed.
6.  **Coefficient Analysis**: The coefficients of all three models are plotted to visualize how Ridge shrinks coefficients and how Lasso pushes many to zero.

## ✅ Key Takeaways

-   **Regularization** is essential for building robust models that generalize well to new data.
-   **Ridge (L2)** is a good general-purpose regularizer that reduces model complexity.
-   **Lasso (L1)** is powerful for both regularization and automatic feature selection.
-   The choice between L1 and L2 depends on the problem, but both are effective at combating overfitting.

## 📂 Files

-   `l1l2regularization.ipynb`: The Jupyter Notebook with code and explanations.
-   `dataset.csv`: The high-dimensional dataset used for the demonstration.
-   `dataset_generator.ipynb`: The notebook used to generate the synthetic dataset.
