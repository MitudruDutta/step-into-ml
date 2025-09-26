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

### 📐 Evaluation Metrics

Once the model is trained, we evaluate how well it generalizes:

```
Residual (e_i) = y_i - ŷ_i
MSE = (1/n) Σ (y_i - ŷ_i)²
RMSE = √MSE
MAE = (1/n) Σ |y_i - ŷ_i|
R² = 1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²
```

- **MSE / RMSE**: Penalize larger errors more (squared term). RMSE in original units.
- **MAE**: Robust to outliers compared to MSE.
- **R²**: Proportion of variance explained (can be negative if model performs worse than predicting the mean).

### ✅ Core Assumptions (for inference / reliability)

1. **Linearity**: Relationship between X and y is linear.
2. **Independence**: Residuals are independent (no autocorrelation).
3. **Homoscedasticity**: Constant variance of residuals across X.
4. **Normality of residuals** (mainly for confidence intervals / hypothesis tests).
5. **No influential outliers** distorting the fit.

(For pure prediction with ML mindset, mild violations may be acceptable—but diagnostics still help.)

### 🔍 Diagnostic Checks

- Scatter plot of `area` vs `price` → assesses linearity.
- Residuals vs predicted values → look for random scatter (no patterns).
- Histogram / KDE or QQ plot of residuals → normality check (optional).
- Leverage/outlier detection: Cook’s distance (advanced extension).

## 🧪 Minimal scikit-learn Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('home_prices.csv')
X = df[['area_sqr_ft']]
y = df['price_lakhs']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f} | R²: {r2:.3f}")
```

## 📊 Dataset

- **File**: `home_prices.csv`
- **Description**: A simple dataset containing two columns:
  - `area_sqr_ft`: The area of the house in square feet (Independent Variable).
  - `price_lakhs`: The price of the house in Lakhs (Dependent Variable).

## 🛠 Implementation Steps

1. **Load Data**: The `home_prices.csv` dataset is loaded using pandas.
2. **Data Exploration**: Explore basic stats and inspect for anomalies/outliers.
3. **Visualization**: Scatter plot of `area_sqr_ft` vs `price_lakhs` to confirm linear trend.
4. **Train-Test Split**: Split to evaluate on unseen data (prevents optimistic bias).
5. **Model Training**: Fit `LinearRegression` from scikit-learn.
6. **Prediction**: Generate predictions on the test set.
7. **Evaluation**: Compute RMSE, R²; plot regression line and residuals.
8. (Optional) **Diagnostics**: Residual plot, leverage points, transform variables if non-linear.

## ⚠️ Common Pitfalls

- Extrapolating beyond the observed range of `area_sqr_ft`.
- Ignoring outliers that skew slope.
- R² obsession: High R² doesn’t imply causal relationship.
- Using a single train/test split instead of cross-validation for small datasets.
- Not checking residual patterns (hidden non-linearity).

## 🚀 Running the Notebook

From the project root (after installing dependencies):

```batch
jupyter notebook "Supervised ML/Simple_linear_regression/linear_regression_single_variable.ipynb"
```

Or open via the Jupyter UI.

## 🔄 Extensions

- Add **multiple features** (bedrooms, location) → Multiple Linear Regression.
- Try **polynomial features** if curvature appears.
- Apply **regularization** (Ridge/Lasso) if expanding features.
- Use **log transformation** if variance increases with area.
- Add **confidence intervals** for predictions (via statsmodels or bootstrapping).

## ✅ Key Takeaways

- **Strengths**: Simple to implement, easily interpretable, and provides a good baseline.
- **Limitations**: Assumes linearity; sensitive to outliers; weak under complex relationships.
- **Best Practice**: Always visualize and validate assumptions before trusting inferences.

## 📂 Files

- `linear_regression_single_variable.ipynb`: The Jupyter Notebook with the Python code and explanations.
- `home_prices.csv`: The dataset used for training and testing.

---

## 🧾 Summary

This module introduces the simplest predictive model: fit a line minimizing squared errors, evaluate with RMSE and R², verify assumptions, and build intuition before moving to richer models.
