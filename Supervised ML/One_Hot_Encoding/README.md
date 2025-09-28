# 🔢 One-Hot Encoding: Handling Categorical Data

## Introduction
Many ML algorithms (linear/logistic regression, SVM, k-NN, neural nets) expect numeric, fixed‑width feature vectors. Real datasets often contain **categorical features** (e.g., locality, color, brand). **One-Hot Encoding (OHE)** converts these categories into binary indicator columns without implying an ordinal relationship.

This mini‑module shows how to encode a `locality` column and train a linear regression model to predict home prices.

## 🧠 Why Not Just Label Encode?
Label encoding (e.g., `Kollur=0, Mankhal=1, Banjara=2`) injects an **artificial ordering** and distances that mislead linear / distance‑based models. One-Hot Encoding keeps categories **orthogonal**.

## 🧩 How One-Hot Encoding Works
Given `locality ∈ {Kollur, Banjara_Hills, Mankhal}` we create binary columns:
- `locality_Kollur`
- `locality_Banjara_Hills`
- `locality_Mankhal`

A row with `locality = Mankhal` becomes `(0,0,1)` for those three columns.

### Dummy Variable Trap
For linear models, the sum of all dummies = 1 causes perfect multicollinearity (design matrix not full rank). Fixes:
- Drop one column (reference category) → `pandas.get_dummies(..., drop_first=True)`
- Or use `OneHotEncoder(drop='first')`

Interpretation: coefficients of remaining dummies represent difference vs. the dropped baseline.

## ⚗️ pandas vs scikit-learn
| Aspect | pandas.get_dummies | sklearn.preprocessing.OneHotEncoder |
|--------|--------------------|-------------------------------------|
| Fit/transform separation | No | Yes |
| Unseen categories at inference | Raises misalignment (need manual alignment) | `handle_unknown='ignore'` available |
| Sparse output | Dense only | Sparse or dense |
| Pipelines / CV integration | Manual | Native |

For production/pipelines, prefer `OneHotEncoder` within a `ColumnTransformer`.

## 📊 Dataset
- **File**: `home_prices.csv`
- **Columns**:
  - `locality` (categorical)
  - `area_sqr_ft` (numeric)
  - `bedrooms` (numeric / discrete)
  - `price_lakhs` (target)

## 🛠 Implementation Steps
1. Load data with pandas.
2. Explore unique categories & frequency (spot rare categories).
3. One-hot encode `locality` (drop one dummy to avoid trap).
4. Split into train/test (ALWAYS split before fitting an encoder that learns categories in realistic setups—when using sklearn's encoder). For `get_dummies` on full data, ensure no leakage of target-derived engineered categories (not the case here but principle matters).
5. Train `LinearRegression` on encoded features.
6. Evaluate `R²` (and optionally MAE/RMSE).
7. Build a small prediction DataFrame with the same columns for inference.

## 🧪 Minimal pandas Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load
df = pd.read_csv('home_prices.csv')

# Encode (drop_first prevents dummy trap)
df_enc = pd.get_dummies(df, columns=['locality'], drop_first=True)

X = df_enc.drop('price_lakhs', axis=1)
y = df_enc['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print('R²:', model.score(X_test, y_test))
```

## 🧪 Recommended sklearn Pipeline Pattern
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

num_features = ['area_sqr_ft', 'bedrooms']
cat_features = ['locality']

X = df[num_features + cat_features]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
], remainder='passthrough')

pipe = Pipeline([
    ('prep', preprocess),
    ('linreg', LinearRegression())
])

pipe.fit(X_train, y_train)
print('R²:', pipe.score(X_test, y_test))
```
Advantages: no manual column alignment, safe with unseen categories, integrates into cross-validation.

## 🧪 Inference Example (Manual Dummy Frame)
When using pandas dummies, you must ensure the prediction DataFrame has the **exact same columns** as training (order included). A safe pattern:
```python
expected_cols = X_train.columns
new_row = pd.DataFrame([{ 'area_sqr_ft': 1600, 'bedrooms': 2, 'locality_Kollur': 0, 'locality_Mankhal': 1 }])
# Add any missing columns
for c in expected_cols:
    if c not in new_row:
        new_row[c] = 0
new_row = new_row[expected_cols]
print(model.predict(new_row))
```
(With the pipeline + OneHotEncoder you just pass the raw categorical column.)

## ⚠️ Common Pitfalls & Edge Cases
- High Cardinailty: 1000+ unique categories explode feature space → consider hashing, target encoding, frequency capping.
- Unseen Categories: `get_dummies` on train+test separately produces mismatched columns; always align or use `OneHotEncoder(handle_unknown='ignore')`.
- Sparse Data: Many dummies mostly zeros → prefer sparse matrix to save memory (set `sparse_output=True` in new sklearn versions or `sparse=True` earlier).
- Dummy Trap: Forgetting to drop one column for linear models (for tree models it’s less critical but still redundant).
- Leakage: Deriving category engineered features after seeing test fold (avoid by using pipelines in CV).

## 📏 Metrics (Optional for Regression)
Supplement R² with:
- RMSE (scale of target)
- MAE (robust to outliers)
- Adjusted R² if adding lots of features

## 🚀 Extensions
- Replace linear regression with `Ridge`/`Lasso` to handle many dummies.
- Use `OneHotEncoder` combined with numerical scaling in a single `ColumnTransformer`.
- Add interaction terms (e.g., locality × bedrooms) via `PolynomialFeatures` on selected pairs.
- Introduce target encoding or count encoding for high-cardinality categories (with CV to avoid leakage).

## 📂 Files
- `one_hot_encoding.ipynb`: Notebook implementation.
- `home_prices.csv`: Dataset.

## ✅ Summary
One-Hot Encoding safely represents nominal categories without implying order. Use pandas for quick exploration; prefer sklearn pipelines for robust, reproducible modeling—especially with unseen categories and cross-validation.

---
Return to main index: [Root README](../../README.md)
