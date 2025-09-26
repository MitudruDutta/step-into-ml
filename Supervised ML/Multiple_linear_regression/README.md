# 📖 Multiple Linear Regression: Advanced Home Price Prediction

## Introduction

Multiple Linear Regression is an extension of Simple Linear Regression. It models the relationship between two or more independent variables (features) and a single dependent variable (target) by fitting a linear equation to the observed data.

This module predicts home prices using multiple features (area and number of bedrooms) and introduces core diagnostics you should perform before trusting a linear model.

## 🧠 Theory

### Model Equation
**`y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`**
- **y**: Target variable.
- **X₁..Xₚ**: Feature variables.
- **β₀**: Intercept.
- **βᵢ**: Coefficient for feature i (marginal effect holding others constant).
- **ε**: Random error (unexplained variation).

### Matrix Form & Normal Equation
**`y = Xβ + ε`** where X includes a leading column of ones for the intercept.

Ordinary Least Squares minimizes the Sum of Squared Errors (SSE):
```
SSE = Σ (yᵢ − ŷᵢ)² = (y − Xβ)ᵀ (y − Xβ)
β̂ = (Xᵀ X)⁻¹ Xᵀ y   (Normal Equation, if XᵀX invertible)
```
If XᵀX is singular (perfect multicollinearity), use the Moore–Penrose pseudo‑inverse or regularization (Ridge/Lasso).

### Evaluation Metrics
```
Residual eᵢ = yᵢ − ŷᵢ
MSE = (1/n) Σ eᵢ²
RMSE = √MSE
MAE = (1/n) Σ |eᵢ|
R² = 1 − Σ eᵢ² / Σ (yᵢ − ȳ)²
Adjusted R² = 1 − (1 − R²) * (n − 1)/(n − p − 1)
```
Adjusted R² penalizes adding irrelevant features (p = number of predictors).

### Core Assumptions (Classical Linear Model)
1. Linear relationship: each feature contributes additively & linearly.
2. Independence of errors.
3. Homoscedasticity: constant error variance.
4. No (or low) multicollinearity.
5. Errors approximately normal (mainly for inference, not pure prediction).
6. No influential outliers distorting coefficients.

### Multicollinearity
High correlation between predictors inflates variance of β estimates.
Indicators:
- High pairwise correlations.
- Variance Inflation Factor (VIF) > 5–10.
Mitigations: remove/recombine features, dimensionality reduction (PCA), or apply Ridge regression.

### When to Prefer Regularization
If p is large, features correlate, or overfitting occurs—use Ridge (L2) or Lasso (L1) instead of plain OLS.

## 📊 Dataset
- **File**: `home_prices.csv`
- **Columns**:
  - `area_sqr_ft` (feature)
  - `bedrooms` (feature)
  - `price_lakhs` (target)

## 🛠 Implementation Steps
1. Load dataset with pandas.
2. Explore structure, summary stats, missing values.
3. Visualize: scatter plots, pair plot, correlation heatmap.
4. Split into train/test sets (e.g. 80/20) for unbiased evaluation.
5. Fit `LinearRegression` model.
6. Inspect coefficients & intercept.
7. Evaluate (RMSE, MAE, R², Adjusted R²).
8. Plot residuals vs predictions (expect random noise band).
9. (Optional) Check multicollinearity (VIF) & feature importance.
10. Predict on new data.

## 🧪 Minimal Code Example (scikit-learn)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load
df = pd.read_csv('home_prices.csv')
X = df[['area_sqr_ft', 'bedrooms']]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

n, p = X_test.shape[0], X_train.shape[1]
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1)/(len(y_test) - p - 1)

print('Coefficients:', dict(zip(X.columns, model.coef_)))
print(f'Intercept: {model.intercept_:.3f}')
print(f'RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} | Adj R²: {adj_r2:.3f}')
```

### (Optional) Variance Inflation Factor (requires statsmodels)
```python
# pip install statsmodels  (if not already installed)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

X_with_const = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
for i, col in enumerate(X_with_const.columns):
    vif = variance_inflation_factor(X_with_const.values, i)
    print(col, 'VIF =', round(vif, 2))
```
Interpretation: VIF > 5 (sometimes 10) signals problematic multicollinearity.

## 🔍 Diagnostics & Plots (Recommended)
- Residuals vs fitted → randomness supports linearity & homoscedasticity.
- Histogram / KDE of residuals → approximate normality.
- Q–Q plot → heavy tails / skew detection.
- Leverage & influence (Cook’s distance) → detect points disproportionately affecting fit.

## ⚠️ Common Pitfalls
- Blindly adding correlated features → unstable coefficients.
- Interpreting coefficients without feature scaling (magnitudes misleading across scales).
- Using high R² as sole success metric (overfitting or data leakage possible).
- Ignoring residual patterns indicating non-linearity.
- Extrapolating far outside feature ranges.

## 🔄 Extensions
- Add categorical features via one-hot encoding (ensure no dummy variable trap by dropping one level).
- Polynomial / interaction terms (e.g., `area * bedrooms`).
- Regularization: Ridge / Lasso / Elastic Net.
- Feature engineering (price per square foot, bedrooms density).
- Cross-validation for more robust performance estimates.
- Pipeline with scaling (if mixing features of differing scales + regularization).

## 🚀 Running the Notebook
From project root after installing dependencies:
```batch
jupyter notebook "Supervised ML/Multiple_linear_regression/linear_regression_mul_var.ipynb"
```
Or open via Jupyter UI.

## ✅ Key Takeaways
- Multiple Linear Regression extends the simple model to incorporate richer context.
- Adjusted R² helps prevent misleading improvement from irrelevant features.
- Always inspect residuals & multicollinearity before trusting interpretations.
- Regularization is the next natural step when features are many or correlated.

## 📂 Files
- `linear_regression_mul_var.ipynb`: Notebook with code and explanations.
- `home_prices.csv`: Dataset for training/testing.

---

## 🧾 Summary
A baseline predictive model using multiple numeric features. You learned formulation, evaluation, multicollinearity checks, and where to go next (interaction terms, regularization, feature engineering).
