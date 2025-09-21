# 📖 Simple Linear Regression: Predicting Home Prices

## Introduction

Simple Linear Regression is a fundamental algorithm in machine learning used to model the relationship between a single independent variable (feature) and a dependent variable (target) by fitting a linear equation to the observed data.

This project serves as a beginner-friendly introduction to implementing Simple Linear Regression to predict home prices based on their square footage.

## 🧠 Theory

### Key Concepts

The goal of simple linear regression is to find the best-fitting straight line through the data points. This line is represented by the equation:

**`y = mx + c`**

- **y**: The dependent variable (what we want to predict, i.e., `price`).
- **x**: The independent variable (the feature we use for prediction, i.e., `area`).
- **m**: The slope or **coefficient** of the line. It represents the change in `y` for a one-unit change in `x`.
- **c**: The **intercept**, which is the value of `y` when `x` is 0.

### Ordinary Least Squares (OLS)

The algorithm finds the optimal values for `m` and `c` by minimizing a cost function. The most common method is **Ordinary Least Squares (OLS)**, which minimizes the **Sum of Squared Errors (SSE)** (also called Residual Sum of Squares).

The error (or residual) for each data point is the difference between the actual value (`y_i`) and the predicted value (`ŷ_i`).

**Error (Residual):** `e_i = y_i - ŷ_i`

**Cost Function (SSE):**
`SSE = Σ(y_i - ŷ_i)² = Σ(y_i - (mx_i + c))²`

OLS finds the `m` and `c` that make this sum as small as possible. The closed-form solution is:

**Slope (m):**
`m = Σ((x_i - x̄)(y_i - ȳ)) / Σ(x_i - x̄)²`

**Intercept (c):**
`c = ȳ - m * x̄`

Where:
- `x̄` is the mean of the independent variable `x`.
- `ȳ` is the mean of the dependent variable `y`.

## 📊 Dataset

- **File**: `home_prices.csv`
- **Description**: A simple dataset containing two columns:
  - `area_sqr_ft`: The area of the house in square feet (Independent Variable).
  - `price_lakhs`: The price of the house in Lakhs (Dependent Variable).

## 🛠 Implementation Steps

1.  **Load Data**: The `home_prices.csv` dataset is loaded using the pandas library.
2.  **Data Exploration**: Basic exploration is done to understand the data's structure and to check for a linear relationship.
3.  **Visualization**: A scatter plot is created to visualize the relationship between `area_sqr_ft` and `price_lakhs`.
4.  **Train-Test Split**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
5.  **Model Training**: A `LinearRegression` model from `scikit-learn` is trained on the training data.
6.  **Prediction**: The trained model is used to make predictions on the test set.
7.  **Evaluation**: The model's performance is evaluated by checking its R-squared score and visualizing the regression line.

## ✅ Key Takeaways

- **Strengths**: Simple to implement, easily interpretable, and provides a good baseline for regression problems.
- **Limitations**: Assumes a linear relationship between variables. It can be sensitive to outliers.
- **Use Cases**: Predicting stock prices, forecasting sales, analyzing the relationship between advertising spend and revenue, etc.

## 📂 Files

- `linear_regression_single_variable.ipynb`: The Jupyter Notebook with the Python code and explanations.
- `home_prices.csv`: The dataset used for training and testing.
