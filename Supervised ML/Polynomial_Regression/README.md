# 📈 Polynomial Regression: Modeling Non-Linear Relationships

## Introduction

Polynomial Regression is a type of regression analysis where the relationship between the independent variable `x` and the dependent variable `y` is modeled as an nth-degree polynomial. It is a special case of multiple linear regression, used when the data shows a non-linear, curvilinear pattern.

This project demonstrates how to predict a car's selling price based on its mileage by fitting a polynomial curve to the data, which a simple straight line (linear regression) cannot capture effectively.

## 🧠 Theory

### Key Concepts

While simple linear regression fits a straight line (`y = mx + c`), polynomial regression fits a curve. It does this by adding polynomial terms of the features to the model.

The equation for a polynomial regression model (e.g., degree 2) is:

**`y = β₀ + β₁x + β₂x² + ε`**

More generally (degree d):
```
y = β₀ + β₁ x + β₂ x² + … + β_d x^d + ε
```
Vector / design‑matrix view (still linear in parameters):
```
Φ(x) = [1, x, x², …, x^d]
y = Φ(x) · β + ε
β̂ = (Φᵀ Φ)⁻¹ Φᵀ y   (if ΦᵀΦ invertible)
```
Even though the relationship between `x` and `y` is non-linear, the model is still treated as a **linear model** because the equation is linear in terms of the coefficients (β₀ … β_d). We achieve this by transforming the input features.

### Bias–Variance & Degree Selection
- Low degree (e.g., 1–2): High bias, may underfit.
- Moderate degree: Balance fit & generalization.
- High degree: Low training error but high variance / overfitting (wiggly curve).

Use cross‑validation to select the degree that minimizes validation error, not just training error.

### Regularization With High Degrees
High-degree polynomial features can explode coefficient magnitudes and overfit. Combine with Ridge/Lasso:
```
Pipeline([
 ('poly', PolynomialFeatures(degree=d, include_bias=False)),
 ('scaler', StandardScaler()),
 ('ridge', Ridge(alpha=α))
])
```
This shrinks coefficients and stabilizes the solution.

### Numerical Stability
- Large raw feature values + high powers => huge numbers (overflow / precision loss).
- Always consider scaling (StandardScaler) before fitting high-degree polynomials.
- Alternatively, center x (x - mean) to reduce multicollinearity among powers.

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
6.  **Model Evaluation**: The performance of the polynomial model is compared to the simple linear model using metrics such as RMSE and R² (validation or test split).
7.  **Visualization of the Fit**: The final polynomial regression curve is plotted against the data to visually confirm its superior fit.
8.  **(Optional) Degree Selection**: Use cross‑validation loop to pick best degree.
9.  **(Optional) Regularized Variant**: Replace LinearRegression with Ridge for high degrees.

## 🔁 Selecting Degree via Cross-Validation
```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

X = df[['mileage']]
y = df['selling_price']

best = None
for d in range(1, 8):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])
    # Negative MSE (scikit-learn convention); take mean over folds
    scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
    mean_rmse = -scores.mean()
    print(f"Degree {d}: RMSE={mean_rmse:.3f}")
    if best is None or mean_rmse < best[1]:
        best = (d, mean_rmse)
print('Best degree:', best)
```

## 🧪 Minimal Fit & Plot Example
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load
df = pd.read_csv('car_prices.csv')
X = df[['mileage']]
y = df['selling_price']

# Polynomial transform (degree 3 for example)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
print(f'RMSE: {rmse:.2f} | R²: {r2:.3f}')

# Smooth curve for plotting
mileage_grid = np.linspace(X.mileage.min(), X.mileage.max(), 200).reshape(-1,1)
curve = model.predict(poly.transform(mileage_grid))
plt.scatter(X, y, s=18, alpha=0.7)
plt.plot(mileage_grid, curve, color='red', linewidth=2)
plt.xlabel('Mileage')
plt.ylabel('Selling Price')
plt.title('Polynomial Regression (degree=3)')
plt.show()
```

## ⚠️ Common Pitfalls
- Choosing an arbitrarily high degree (severe overfitting, oscillations at edges).
- Evaluating only on training data; always use validation/test metrics.
- Forgetting to scale when combining with regularization or very high degrees.
- Extrapolating outside observed mileage range (polynomials behave wildly).
- Interpreting individual polynomial coefficient magnitudes (not inherently meaningful without centering / scaling).

## 🔄 Extensions
- Automate degree selection with `GridSearchCV`.
- Use Ridge/Lasso with polynomial features to control overfitting.
- Try spline regression or tree-based models if local flexibility needed.
- Add interaction terms for multi-feature polynomial models.
- Log-transform target or feature if variance is multiplicative.

## 🚀 Running the Notebook
From project root:
```batch
jupyter notebook "Supervised ML/Polynomial_Regression/poly_regression.ipynb"
```
Ensure dependencies installed (see root `requirements.txt`).

## ✅ Key Takeaways
- Polynomial Regression = Linear Regression on an expanded (non-linear) feature space.
- Degree selection is crucial; rely on cross‑validation, not visual guesswork alone.
- High-degree polynomials are prone to overfitting & numerical instability—use scaling and consider regularization.
- Still a parametric approach; alternatives (splines, kernels, ensembles) may generalize better for complex shapes.

## 📂 Files

-   `poly_regression.ipynb`: The Jupyter Notebook with the Python code and detailed explanations.
-   `car_prices.csv`: The dataset used for the regression task.

---

## 🧾 Summary
You transformed a single feature into a richer polynomial basis to capture curvature, selected degree via validation, and learned how regularization and scaling help maintain generalization. This prepares you for feature engineering and more advanced non-linear models.
