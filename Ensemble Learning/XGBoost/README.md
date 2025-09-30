# 🚀 XGBoost: eXtreme Gradient Boosting

## Introduction

**XGBoost (eXtreme Gradient Boosting)** is an optimized, regularized gradient boosting framework focused on speed and performance. It provides parallel tree boosting (GBDT / GBM) and is widely adopted in Kaggle competitions and production ML for structured/tabular data.

## 🧠 Theory Recap
XGBoost extends classical Gradient Boosting with:
1. **Second-Order Optimization**: Uses both first and second derivatives (gradient + hessian) of the loss for more precise updates.
2. **Explicit Regularization**: Penalizes number of leaves and leaf weights (L1 & L2) in the objective to reduce overfitting.
3. **Sparsity Awareness**: Efficient handling of missing values & sparse inputs with learned default directions.
4. **Weighted Quantile Sketch**: Efficient tree split finding for weighted data.
5. **Out-of-core & Parallelization**: Can scale to larger-than-memory datasets (not demonstrated here but supported).
6. **Early Stopping**: Monitor eval metric on validation set to stop before overfitting.

## 🔍 When to Use / Not Use
Use when:
- Data is structured/tabular with mixed feature types.
- You need a strong baseline quickly.
- Interpretability via feature importance / SHAP is acceptable.

Avoid / reassess when:
- Data is mostly unstructured (images, audio, raw text embeddings) — deep learning may fit better.
- Real‑time, very low-latency training updates required (consider simpler linear models).
- Extremely high feature dimensionality with heavy sparsity (try linear booster mode or alternative algorithms).

## ⚙️ Key Hyperparameters (Common Ones)
| Parameter | Purpose | Typical Values | Notes |
|-----------|---------|----------------|-------|
| `n_estimators` / `num_boost_round` | Number of trees | 100–1000 | Tune with `early_stopping_rounds`. |
| `learning_rate` (`eta`) | Shrinkage factor | 0.01–0.3 | Lower ⇒ need more trees. |
| `max_depth` | Tree depth | 3–8 | Higher depth ⇒ risk of overfit. |
| `subsample` | Row sampling | 0.6–1.0 | <1 adds stochastic regularization. |
| `colsample_bytree` | Feature subsampling | 0.6–1.0 | Helps reduce correlation between trees. |
| `gamma` | Min loss reduction to split | 0–5+ | Larger ⇒ more conservative splitting. |
| `min_child_weight` | Min hessian sum per leaf | 1–10 | Higher ⇒ fewer, more general leaves. |
| `reg_alpha` | L1 regularization | 0–1+ | Can induce sparsity. |
| `reg_lambda` | L2 regularization | 0–10 | Stabilizes weights. |
| `objective` | Loss function | `binary:logistic`, `reg:squarederror`, etc. | Must match task. |
| `eval_metric` | Monitoring metric | e.g. `logloss`, `auc`, `rmse` | Use domain-relevant metric. |

> Start with moderate depth (4–6), a learning rate of 0.05–0.1, and enable `early_stopping_rounds` with a validation set.

## 📊 Datasets
1. **Synthetic Classification Dataset**
   - Generated via `make_classification` (multi-class example).
2. **Regression Dataset**
   - Historically the Boston Housing dataset was popular, but `load_boston` has been **removed** from scikit-learn (ethical/data quality concerns).
   - If the notebook still uses Boston, consider replacing with:
     - California Housing: `fetch_california_housing()`
     - Synthetic regression: `make_regression()`
     - Kaggle / open housing datasets.

> Recommendation: Update future examples to avoid deprecated Boston dataset usage.

## 🛠 Implementation Outline
### 1. Classification
- Generate / load data, train/test split
- Baseline: Logistic Regression
- Train: XGBoost Classifier (`XGBClassifier`)
- Evaluate: Accuracy, Precision, Recall, F1, (add ROC AUC if class imbalance exists)
- Tune: Grid / Random search over (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`)

### 2. Regression
- Load / generate dataset
- Baseline: Linear Regression
- Train: XGBoost Regressor (`XGBRegressor`)
- Evaluate: RMSE / MAE / R²
- Feature importance visualization (gain / weight / cover)

## 🚀 Quick Start
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```
Open:
- `xgboostclassifier.ipynb`
- `xgboostregressor.ipynb`

## 🧪 Minimal Usage Example
```python
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=False)
print(model.best_iteration)
print(model.score(X_test, y_test))
```

## 🔍 Interpretation & Diagnostics
- `model.feature_importances_` (default: gain) — use SHAP for more robust interpretability.
- SHAP (optional):
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
```
- Learning curves via evaluation history (`model.evals_result()`).

## 🧯 Common Pitfalls
| Problem | Symptom | Mitigation |
|---------|---------|-----------|
| Overfitting | Train metric improves but validation stalls | Use early stopping, increase `min_child_weight`, add regularization, lower depth |
| Slow training | Long runs | Reduce trees (`n_estimators`), increase `learning_rate`, reduce features/rows, enable hardware acceleration (GPU) |
| Class imbalance | Poor recall on minority | Use `scale_pos_weight = (neg/pos)`, or re-weight classes |
| Memory usage | High RAM usage | Use smaller `max_depth`, enable `hist` tree method (if using native API) |
| Not converging | Flat metrics | Lower learning rate & tune, verify data preprocessing |

## 🧪 Suggested Experiments
- Compare `learning_rate` grid: [0.01, 0.05, 0.1, 0.2]
- Depth vs. regularization: vary `max_depth` with `min_child_weight`
- Add column sampling synergy: tune `colsample_bytree` & `subsample`
- Evaluate SHAP interactions for top 5 features
- Replace dataset with California Housing to modernize regression example

## 📂 Files
- `xgboostclassifier.ipynb`: Classification workflow
- `xgboostregressor.ipynb`: Regression workflow

## ✅ Key Takeaways
- XGBoost combines gradient boosting with additional system & algorithmic optimizations.
- Regularization + early stopping are central to generalization.
- Proper hyperparameter tuning materially impacts performance.
- Consider dataset update if still relying on deprecated Boston housing data.

## 🔗 Resources
- Docs: https://xgboost.readthedocs.io/
- Parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
- Python API: https://xgboost.readthedocs.io/en/latest/python/python_api.html
- Original Paper: https://arxiv.org/abs/1603.02754
- SHAP: https://shap.readthedocs.io/

## 📅 Version / Status
Last reviewed: 2025-09-30

---
Future Improvements:
- Add GPU training example (`tree_method='gpu_hist'`).
- Integrate Optuna / Hyperopt for smarter parameter search.
- Replace Boston dataset fully if still present in notebook code.
