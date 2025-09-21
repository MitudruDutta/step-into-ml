# 📖 Multiple Linear Regression: Advanced Home Price Prediction

## Introduction

Multiple Linear Regression is an extension of Simple Linear Regression. It is used to model the relationship between two or more independent variables (features) and a single dependent variable (target) by fitting a linear equation to the observed data.

This project demonstrates how to predict home prices using multiple features—specifically, the area and the number of bedrooms.

## 🧠 Theory

### Key Concepts

The goal is to find the best-fitting linear equation that describes the relationship between the features and the target. The equation for multiple linear regression is:

**`Y = m1*X1 + m2*X2 + ... + mn*Xn + c`**

- **Y**: The dependent variable (e.g., `price`).
- **X1, X2, ..., Xn**: The independent variables (e.g., `area`, `bedrooms`).
- **m1, m2, ..., mn**: The coefficients for each independent variable. Each coefficient represents the change in `Y` for a one-unit change in its corresponding `X`, holding all other variables constant.
- **c**: The intercept, which is the value of `Y` when all `X` variables are 0.

The algorithm determines the optimal values for the coefficients and the intercept by minimizing the Sum of Squared Errors.

## 📊 Dataset

- **File**: `home_prices.csv`
- **Description**: This dataset contains three columns:
  - `area_sqr_ft`: The area of the house in square feet (Independent Variable).
  - `bedrooms`: The number of bedrooms in the house (Independent Variable).
  - `price_lakhs`: The price of the house in Lakhs (Dependent Variable).

## 🛠 Implementation Steps

1.  **Load Data**: The `home_prices.csv` dataset is loaded using pandas.
2.  **Exploratory Data Analysis (EDA)**: A pair plot and correlation heatmap are used to visualize the relationships between all variables.
3.  **Data Preparation**: The features (`area_sqr_ft`, `bedrooms`) are separated from the target (`price_lakhs`).
4.  **Train-Test Split**: The data is split into training and testing sets to ensure the model can be evaluated on unseen data.
5.  **Model Training**: A `LinearRegression` model from `scikit-learn` is trained using the multiple features in the training set.
6.  **Model Evaluation**: The model's performance is assessed on the test set using metrics like R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
7.  **Prediction**: The trained model is used to predict the price of a new house given its area and number of bedrooms.

## ✅ Key Takeaways

- **Strengths**: Allows for modeling more complex relationships by incorporating multiple factors. Still highly interpretable.
- **Limitations**: Assumes a linear relationship between each feature and the target. It can be affected by **multicollinearity** (high correlation between independent variables).
- **Use Cases**: Predicting house prices based on features like size, location, and age; estimating a student's exam score based on study hours and previous grades.

## 📂 Files

- `linear_regression_mul_var.ipynb`: The Jupyter Notebook with the Python code and detailed explanations.
- `home_prices.csv`: The dataset used for training and testing.
