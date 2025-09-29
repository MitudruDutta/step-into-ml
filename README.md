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
- Polynomial Regression
  - Notebook: [poly_regression.ipynb](Supervised%20ML/Polynomial_Regression/poly_regression.ipynb)
  - Folder: [Supervised ML/Polynomial_Regression](Supervised%20ML/Polynomial_Regression/)
- Regularization (Ridge/Lasso)
  - Notebook: [l1l2regularization.ipynb](Supervised%20ML/L1L2Regularization/l1l2regularization.ipynb)
  - Folder: [Supervised ML/L1L2Regularization](Supervised%20ML/L1L2Regularization/)
- Train/Test Split
  - Notebook: [train_test_split.ipynb](Supervised%20ML/Train_Test_Split/train_test_split.ipynb)
  - Folder: [Supervised ML/Train_Test_Split](Supervised%20ML/Train_Test_Split/)
- Logistic Regression
  - Notebook: [logisticregression.ipynb](Supervised%20ML/LogisticRegression/logisticregression.ipynb)
  - Folder: [Supervised ML/LogisticRegression](Supervised%20ML/LogisticRegression/)
- Decision Trees
  - Notebook: [decisiontree.ipynb](Supervised%20ML/Decision_tree/decisiontree.ipynb)
  - Folder: [Supervised ML/Decision_tree](Supervised%20ML/Decision_tree/)
- Precision & Recall
  - Notebook: [precision_recall.ipynb](Supervised%20ML/Precision_Recall/precision_recall.ipynb)
  - Folder: [Supervised ML/Precision_Recall](Supervised%20ML/Precision_Recall/)
- F1 Score & Confusion Matrix
  - Notebook: [f1_confusion_matrix.ipynb](Supervised%20ML/F1_ConfusionMatrix/f1_confusion_matrix.ipynb)
  - Folder: [Supervised ML/F1_ConfusionMatrix](Supervised%20ML/F1_ConfusionMatrix/)
- ROC Curve & AUC
  - Notebook: [rocauc.ipynb](Model%20Evaluation%20and%20Fine%20Tuning/ROCAUC/rocauc.ipynb)
  - Folder: [Model Evaluation and Fine Tuning/ROCAUC](Model%20Evaluation%20and%20Fine%20Tuning/ROCAUC/)
- Support Vector Machine (SVM)
  - Notebook: [SVM.ipynb](Supervised%20ML/SVM/SVM.ipynb)
  - Folder: [Supervised ML/SVM](Supervised%20ML/SVM/)
- Naive Bayes (SMS Spam Classification)
  - Notebook: [smsclassifier.ipynb](Supervised%20ML/NaiveBayes_SMSSpamClassification_/smsclassifier.ipynb)
  - Folder: [Supervised ML/NaiveBayes_SMSSpamClassification_](Supervised%20ML/NaiveBayes_SMSSpamClassification_/)
- Feature Scaling (Min-Max & Standardization)
  - Notebook: [scaling.ipynb](Supervised%20ML/Data_Scaling(Min_Max)/scaling.ipynb)
  - Folder: [Supervised ML/Data_Scaling(Min_Max)](Supervised%20ML/Data_Scaling(Min_Max)/)
- One-Hot Encoding (Categorical Features)
  - Notebook: [one_hot_encoding.ipynb](Supervised%20ML/One_Hot_Encoding/one_hot_encoding.ipynb)
  - Folder: [Supervised ML/One_Hot_Encoding](Supervised%20ML/One_Hot_Encoding/)
- Scikit-Learn Pipeline
  - Notebook: [sklearnpipeline.ipynb](Supervised%20ML/SklearnPipeline/sklearnpipeline.ipynb)
  - Folder: [Supervised ML/SklearnPipeline](Supervised%20ML/SklearnPipeline/)
- Customer Churn (Imbalanced Classification)
  - Notebook: [churnprediction.ipynb](Supervised%20ML/CustomerChurn/churnprediction.ipynb)
  - Folder: [Supervised ML/CustomerChurn](Supervised%20ML/CustomerChurn/)
- Random Forest (Ensemble Learning)
  - Notebook: [randomforestclassification.ipynb](Ensemble%20Learning/RandomForest/randomforestclassification.ipynb)
  - Folder: [Ensemble Learning/RandomForest](Ensemble%20Learning/RandomForest/)

For all other topics (e.g., scaling, regularization, classification, ensembles), navigate via the folder tree. Detailed write-ups live inside each topic, not here.

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
    - R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²

- Multiple Linear Regression (many features):
  - Model: y = β₀ + β₁x₁ + … + β_px_p + ε (vector form: y = Xβ + ε)
  - Ordinary Least Squares (if XᵀX invertible): β̂ = (XᵀX)⁻¹ Xᵀ y
  - Practical notes: check multicollinearity, scale features when needed, evaluate with train/test split.

- Polynomial Regression (nonlinear in x, linear in parameters):
  - Model (degree k): y = β₀ + β₁ x + β₂ x² + … + β_k x^k + ε
  - View as linear regression on polynomial features Φ(x) = [1, x, x², …, x^k]
  - OLS on transformed design matrix: β̂ = (ΦᵀΦ)⁻¹ Φᵀ y
  - Practice: generate features via `sklearn.preprocessing.PolynomialFeatures`, fit with `LinearRegression`; tune degree with cross‑validation to avoid overfitting.

Use scikit-learn for practical training/evaluation:
- SLR/MLR/Poly: `from sklearn.linear_model import LinearRegression`, `from sklearn.preprocessing import PolynomialFeatures`
- Split & metrics: `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import mean_squared_error, r2_score`

---

## Regularization essentials

Regularization adds a penalty to control model complexity and reduce overfitting.

- Ridge (L2):
  - Objective: min_β (1/2n) ||y − Xβ||² + α ||β||²₂
  - Closed-form: β̂ = (XᵀX + 2nα I)⁻¹ Xᵀ y
  - Properties: shrinks coefficients smoothly; keeps all features; sensitive to feature scale.

- Lasso (L1):
  - Objective: min_β (1/2n) ||y − Xβ||² + α ||β||₁
  - Solution: no closed-form; solved by coordinate descent/ISTA; can set some β_j exactly to 0 (feature selection).

Tips:
- Standardize features first: `StandardScaler()` inside a `Pipeline` with `Ridge`/`Lasso`.
- Tune α via cross-validation: `RidgeCV`, `LassoCV`.

---

## Classification metrics essentials

Confusion Matrix layout:

|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| Actual 0   | TN          | FP          |
| Actual 1   | FN          | TP          |

Key formulas (binary):
- Precision = TP / (TP + FP)
- Recall (Sensitivity) = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall) = 2TP / (2TP + FP + FN)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

When to favor which:
- High Precision: costly false positives (e.g. spam mislabeling important email)
- High Recall: costly false negatives (e.g. disease screening)
- F1: need balance under class imbalance

(Deeper narrative lives in each topic folder; this section stays concise.)

---

## Daily growth plan (incremental)

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
14) Day 14: Customer Churn (Imbalanced Classification)
15) Day 15: Random Forest
16) Day 16: Gradient Boosting
17) Day 17: XGBoost
18) Day 18: Voting Ensembles

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
