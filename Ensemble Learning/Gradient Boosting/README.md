# 🌱 Gradient Boosting: A Step-by-Step Guide

## Introduction

**Gradient Boosting** is a powerful ensemble learning technique that builds models sequentially, each new model correcting the errors of the previous ones. It is widely used for both classification and regression tasks due to its high accuracy and flexibility.

This project demonstrates how to implement and compare different tree-based models, including **Decision Trees**, **Random Forests**, and **Gradient Boosting**, using the Titanic dataset (classification) and an Advertising Spend dataset (regression).

## 🧠 Theory

### What is Gradient Boosting?
Gradient Boosting is an ensemble method that combines the predictions of multiple weak learners (typically shallow decision trees) to create a strong predictive model. Core ideas:

1. **Additive, Stage-Wise Learning**: Start with a simple model (often predicting the mean). Add trees one at a time. Each new tree fits the *residual errors* (negative gradient) from the previous ensemble.
2. **Gradient Descent on Loss**: For a chosen loss function (e.g., log loss, squared error), each tree approximates the direction that most reduces the loss.
3. **Shrinkage (Learning Rate)**: Each tree's contribution is scaled to slow learning and reduce overfitting.
4. **Regularization**: Control complexity via tree depth, number of estimators, subsampling, and learning rate.

### Why Use Gradient Boosting?
- 🔍 Often **top performer** on structured/tabular data.
- 🧩 Handles mixed feature types after minimal preprocessing.
- 📉 Robust to moderate outliers (depending on loss).
- 🪄 Provides **feature importance** / permutation importance insights.

### When NOT to Use It
- Very large datasets where **training time** is critical and simpler linear models suffice.
- Extremely high-dimensional sparse text data (linear models or specialized methods may outperform).
- Real-time training requirements (consider pre-trained model or faster variants like **LightGBM** / **XGBoost** / **CatBoost**).

### Gradient Boosting vs. Random Forest
| Aspect | Random Forest | Gradient Boosting |
| ------ | ------------- | ----------------- |
| Tree Training | Independent (bagging) | Sequential (dependent) |
| Bias | Higher | Lower |
| Variance | Lower | Higher (needs regularization) |
| Tunable Learning Rate | No | Yes |
| Overfitting Risk | Lower | Higher if not tuned |

### Typical Loss Functions
- Regression: Squared Error, MAE, Huber
- Classification: Log Loss / Deviance

## ⚙️ Key Hyperparameters (Scikit-learn `GradientBoosting*`)
- `n_estimators`: Number of trees (too high without shrinkage ⇒ overfit).
- `learning_rate`: Shrinkage factor; lower values usually need more trees (common grid: 0.01–0.2).
- `max_depth` or `max_leaf_nodes`: Tree complexity (shallow trees generalize better).
- `subsample`: Row sampling fraction (<1.0 introduces stochasticity ⇒ regularization).
- `min_samples_split`, `min_samples_leaf`: Prevent overly specific leaves.
- `loss`: Depends on task (`squared_error`, `log_loss`, etc.).

> Rule of thumb: Lower `learning_rate` + Higher `n_estimators` often improves generalization, but watch training time.

## 📊 Datasets

1. **Titanic Dataset (`titanic.csv`)**
   - Task: Predict passenger survival (binary classification).
   - Features: Passenger class, sex, age, fare, etc.
   - Target: `Survived` (0 = No, 1 = Yes).

2. **Advertising Spend Dataset (`ad_spend.csv`)**
   - Task: Predict sales based on advertising spend across different channels.
   - Features: Budgets for TV, radio, newspaper ads.
   - Target: `Sales` (continuous).

> If you reuse these notebooks with other data, ensure proper handling of missing values, encoding, and scaling (only needed for certain models, not for tree splits).

## 🛠 Implementation Overview

### 1. Classification: Titanic Survival Prediction
- Preprocessing: Impute missing age/fare, encode categorical fields (e.g., `Sex`, `Embarked`).
- Models Compared:
  - Decision Tree
  - Random Forest
  - Gradient Boosting Classifier
- Metrics: Accuracy, Precision, Recall, F1, (optionally ROC AUC).

### 2. Regression: Advertising Sales Prediction
- Preprocessing: Basic cleaning; scaling not strictly required for tree-based models, but may be demonstrated for teaching.
- Models Compared:
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Metrics: MSE, MAE (add if not already), R².

## 🚀 Quick Start

Open the notebooks in Jupyter / VS Code:
```
# (Optional) create venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```
Then navigate to:
- `gradientboostclass.ipynb`
- `gradientboostreg.ipynb`

## 🧪 Example (Conceptual Pseudocode)
```
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## 📂 Files
- `gradientboostclass.ipynb`: Classification (Titanic)
- `gradientboostreg.ipynb`: Regression (Advertising)
- `titanic.csv`: Dataset for classification
- `ad_spend.csv`: Dataset for regression

## 🔍 Model Interpretation Ideas
- Plot feature importances: `model.feature_importances_`
- Partial Dependence or SHAP (extend later for deeper interpretability)
- Compare learning curves (vary `n_estimators`)

## 🧯 Common Pitfalls & Tips
| Issue | Symptom | Mitigation |
| ----- | ------- | ---------- |
| Overfitting | Train >> Test metrics | Lower `max_depth`, add `subsample`, reduce `n_estimators`, lower `learning_rate` |
| Slow Training | Long runtime | Reduce `n_estimators`, use higher `learning_rate`, subsample rows |
| Plateaued Performance | No gain with more trees | Reduce `learning_rate` and re-tune, engineer features |
| Class Imbalance | Poor recall on minority | Use `class_weight`, stratified splits, calibrate thresholds |

## 🧪 Suggested Experiments
- Grid search over (`learning_rate`, `n_estimators`, `max_depth`).
- Add stochasticity: `subsample=0.7`.
- Compare with `XGBoost` / `HistGradientBoosting*` for speed.

## ✅ Key Takeaways
- Gradient Boosting builds trees sequentially to reduce residual errors.
- Tuning involves a **balance** between `learning_rate` and `n_estimators`.
- Provides strong baselines for many structured data tasks.
- Consider more optimized libraries (XGBoost / LightGBM / HistGradientBoosting) for larger datasets.

## 🔗 Extending Further
- Replace with `from sklearn.ensemble import HistGradientBoostingClassifier` for faster training on medium/large data.
- Try `XGBoost` for sparse/high-dimensional tasks.

## 📚 Resources
- StatQuest: https://www.youtube.com/watch?v=3CC4N4z3GJc
- Scikit-learn Docs (Gradient Boosting): https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
- Elements of Statistical Learning (Boosting Chapters)

## 📅 Version / Status
Last reviewed: 2025-09-30

---
Feel free to propose additional experiments (e.g., early stopping via staged predictions) or add SHAP explanations in a future iteration.
