# Step Into Machine Learning

A growing, hands-on workspace of Jupyter notebooks and small datasets to learn supervised ML step by step. The main README stays concise and links you directly to the specific topic folders/notebooks.

---

## Quick links to topics

- Simple Linear Regression
  - Notebook: [linear_regression_single_variable.ipynb](Supervised%20ML/Simple_linear_regression/linear_regression_single_variable.ipynb)
  - Folder: [Supervised ML/Simple_linear_regression](Supervised%20ML/Simple_linear_regression/)
- Multiple Linear Regression
  - Notebook: [linear_regression_mul_var.ipynb](Supervised%20ML/Multiple_linear_regression/linear_regression_mul_var.ipynb)
  - Folder: [Supervised ML/Multiple_linear_regression](Supervised%20ML/Multiple_linear_regression/)

For all other topics (e.g., train/test split, scaling, regularization, classification, ensembles), navigate via the folder tree. Detailed write-ups live inside each topic, not here.

---

## What is Machine Learning 

Machine Learning (ML) uses data to learn patterns that generalize to unseen cases. This workspace focuses on supervised learning with tabular data in Python (pandas, scikit-learn, matplotlib, seaborn, etc.).

---

## Linear Regression essentials 

- Simple Linear Regression (one feature):
  - Model: y = β₀ + β₁ x + ε
  - Closed-form fit:
    - β̂₁ = Cov(x, y) / Var(x)
    - β̂₀ = ȳ − β̂₁ x̄
  - Common metrics:
    - MSE = (1/n) Σ (yᵢ − ŷᵢ)²
    - RMSE = √MSE
    - R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²

- Multiple Linear Regression (many features):
  - Model: y = β₀ + β₁x₁ + … + β_px_p + ε (vector form: y = Xβ + ε)
  - Ordinary Least Squares (if XᵀX invertible): β̂ = (XᵀX)⁻¹ Xᵀ y
  - Practical notes: check multicollinearity, scale features when needed, evaluate with train/test split.

Use scikit-learn for practical training/evaluation:
- SLR/MLR: `from sklearn.linear_model import LinearRegression`
- Split & metrics: `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import mean_squared_error, r2_score`

---

## Daily growth plan 

1) Day 1: Simple Linear Regression
2) Day 2: Train/Test Split and Regression Metrics (MSE/RMSE, R²)
3) Day 3: Multiple Linear Regression
4) Day 4: Polynomial Regression
5) Day 5: Regularization (L1/Lasso, L2/Ridge)
6) Day 6: Logistic Regression
7) Day 7: Decision Trees
8) Day 8: Precision/Recall, F1, Confusion Matrix
9) Day 9: ROC and AUC
10) Day 10: SVM
11) Day 11: Naive Bayes
12) Day 12: Feature Scaling & One‑Hot Encoding
13) Day 13: Scikit‑Learn Pipelines
14) Day 14: Random Forest
15) Day 15: Gradient Boosting
16) Day 16: XGBoost
17) Day 17: Voting Ensembles

This schedule is flexible; details are maintained inside each topic folder.

---

## Quickstart (Windows, cmd.exe)

Prerequisites: Python 3.9+ and Git (optional).

```batch
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ml-workspace --display-name "Python (ml-workspace)"
```

Launch notebooks:

```batch
jupyter notebook
```

Tips:
- If `xgboost` has issues, ensure recent Python/pip; prebuilt wheels exist for common versions.
- `.xlsx` reading requires `openpyxl` (already listed in requirements).

---

## Contributing and updates

We push changes incrementally, focusing one topic at a time. The top-level README remains a hub with links; topic-specific explanations live in their folders/notebooks.

---

## License

No explicit license yet. Add a LICENSE (e.g., MIT) if you plan to share/reuse publicly.
