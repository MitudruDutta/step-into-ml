# 📈 Polynomial Regression: Modeling Non-Linear Relationships

## Introduction

Polynomial Regression is a type of regression analysis where the relationship between the independent variable `x` and the dependent variable `y` is modeled as an nth-degree polynomial. It is a special case of multiple linear regression, used when the data shows a non-linear, curvilinear pattern.

This project demonstrates how to predict a car's selling price based on its mileage by fitting a polynomial curve to the data, which a simple straight line (linear regression) cannot capture effectively.

## 🧠 Theory

### Key Concepts

While simple linear regression fits a straight line (`y = mx + c`), polynomial regression fits a curve. It does this by adding polynomial terms of the features to the model.

The equation for a polynomial regression model (e.g., degree 2) is:

**`y = β₀ + β₁x + β₂x² + ε`**

-   **y**: The dependent variable (e.g., `selling_price`).
-   **x**: The independent variable (e.g., `mileage`).
-   **x²**: The polynomial feature (the square of the mileage).
-   **β₀, β₁, β₂**: The coefficients of the model.

Even though the relationship between `x` and `y` is non-linear, the model is still treated as a **linear model** because the equation is linear in terms of the coefficients (β₀, β₁, β₂). We achieve this by transforming the input features.

## 📊 Dataset

-   **File**: `car_prices.csv`
-   **Description**: A dataset containing information about used cars.
    -   `mileage`: The total distance the car has been driven (Independent Variable).
    -   `selling_price`: The price at which the car was sold (Dependent Variable).

## 🛠 Implementation Steps

1.  **Load Data**: The `car_prices.csv` dataset is loaded using pandas.
2.  **Data Visualization**: A scatter plot is created to visualize the non-linear relationship between `mileage` and `selling_price`.
3.  **Simple Linear Regression (Baseline)**: A simple linear model is first trained to demonstrate its poor fit on this non-linear data.
4.  **Feature Transformation**: `PolynomialFeatures` from `scikit-learn` is used to create higher-degree features (e.g., `mileage²`).
5.  **Model Training**: A `LinearRegression` model is trained on these new polynomial features.
6.  **Model Evaluation**: The performance of the polynomial model is compared to the simple linear model using the R-squared (R²) score.
7.  **Visualization of the Fit**: The final polynomial regression curve is plotted against the data to visually confirm its superior fit.

## ✅ Key Takeaways

-   **Strengths**: Allows you to model non-linear relationships with the simplicity of a linear model. Very flexible and can be adapted to a wide range of curves.
-   **Limitations**: Prone to **overfitting**, especially with high-degree polynomials. The choice of the right polynomial degree is crucial and can be challenging.
-   **Use Cases**: Analyzing the growth rate of tissues, predicting the progression of diseases, and modeling population growth.

## 📂 Files

-   `poly_regression.ipynb`: The Jupyter Notebook with the Python code and detailed explanations.
-   `car_prices.csv`: The dataset used for the regression task.
