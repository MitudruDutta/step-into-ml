# Ensemble Learning: Majority, Average, and Weighted Average Voting

This directory contains examples and implementations of ensemble learning techniques using voting methods for both classification and regression tasks.

## Table of Contents
- [Introduction to Ensemble Voting](#introduction-to-ensemble-voting)
- [Why Voting Works](#why-voting-works)
- [Types of Voting](#types-of-voting)
- [Quick Implementation Examples](#quick-implementation-examples)
- [Weighted Voting Details](#weighted-voting-details)
- [Classification Example: Raisin Dataset](#classification-example-raisin-dataset)
- [Regression Example: House Price Prediction](#regression-example-house-price-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Best Practices & Tips](#best-practices--tips)
- [Key Takeaways](#key-takeaways)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Changelog](#changelog)

## Introduction to Ensemble Voting

Ensemble voting combines predictions from multiple base estimators to produce a final decision that is (ideally) more accurate, robust, and stable than any single model. This leverages the wisdom-of-crowds effect: if individual models make uncorrelated errors, aggregating them reduces variance (and sometimes bias).

## Why Voting Works
- Error Reduction: Independent (or weakly correlated) errors cancel out.
- Stability: Less sensitive to noise in the training data.
- Model Complementarity: Different algorithms capture different structure.
- Simplicity: Minimal tuning required compared to more complex stacking methods.

## Types of Voting

### 1. Hard Voting (Majority Voting)
- Task: Classification
- Mechanism: Each classifier outputs a class label; majority class wins.
- Sklearn: `VotingClassifier(voting="hard")`
- Use when class probability calibration is poor or heterogeneous.

### 2. Soft Voting (Probability Averaging)
- Task: Classification
- Mechanism: Average (optionally weighted) class probabilities; choose argmax.
- Requirement: All models must implement `predict_proba` (e.g., set `probability=True` for `SVC`).
- Sklearn: `VotingClassifier(voting="soft")`
- Often outperforms hard voting when probabilities are well-calibrated.

### 3. Average Voting
- Task: Regression
- Mechanism: Arithmetic mean of predictions.
- Sklearn: `VotingRegressor(estimators=[...])`

### 4. Weighted Voting / Weighted Averaging
- Task: Classification & Regression
- Mechanism: Weighted mean instead of simple mean. Higher weights for stronger models.
- Sklearn: `VotingClassifier(..., weights=[...])` or `VotingRegressor(..., weights=[...])`

## Quick Implementation Examples

### Hard vs Soft Voting (Classification)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Base estimators (with probability support for soft voting)
log_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
svc_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, random_state=42))
])
tr_clf = DecisionTreeClassifier(max_depth=6, random_state=42)

hard_vote = VotingClassifier(
    estimators=[('lr', log_clf), ('svc', svc_clf), ('dt', tr_clf)],
    voting='hard'
)
soft_vote = VotingClassifier(
    estimators=[('lr', log_clf), ('svc', svc_clf), ('dt', tr_clf)],
    voting='soft',
    weights=[2, 3, 1]  # emphasize stronger models
)
```

### Regression Voting (Weighted)
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor

lin = LinearRegression()
ridge = Ridge(alpha=1.0, random_state=42)
cart = DecisionTreeRegressor(max_depth=5, random_state=42)

weighted_reg = VotingRegressor(
    estimators=[('lin', lin), ('ridge', ridge), ('tree', cart)],
    weights=[2, 3, 1]
)
```

## Weighted Voting Details
Given predictions (classification probabilities or regression outputs) from models M₁..M_k and weights w₁..w_k:

Weighted average prediction:  ŷ = (Σ wᵢ · yᵢ) / (Σ wᵢ)

Choose weights based on:
- Cross-validation performance (e.g., inverse validation error)
- Model reliability / probability calibration
- Domain knowledge or business cost structure

Sklearn automatically normalizes by Σw for probability averaging.

## Classification Example: Raisin Dataset
The `ensemble_voting_classifier.ipynb` notebook demonstrates ensemble voting on the Raisin Dataset (two classes: Kecimen, Besni). Features include:
- Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter

Models explored:
1. Logistic Regression
2. SVC (probability enabled)
3. Decision Tree

Example results:
- Hard voting accuracy ≈ 0.89 (varies slightly by seed)
- Soft voting similar or slightly improved when probabilities are good

## Regression Example: House Price Prediction
The `emsemble_voting_regressor.ipynb` notebook (filename retains a spelling typo for backward compatibility) predicts `price_lakhs` from:
- `area_sqr_ft`
- `bedrooms`

Models:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor

The ensemble offers reduced variance compared to the tree and greater stability compared to individual linear models.

## Evaluation Metrics

Classification:
- Accuracy
- (Recommended) Precision, Recall, F1
- Confusion Matrix for per-class error distribution

Regression:
- Mean Squared Error (MSE) / RMSE
- R² Score
- Mean Absolute Error (MAE) (robustness to outliers)

## Best Practices & Tips
- Encourage model diversity (different algorithmic biases) for maximum gain.
- Standardize features for scale-sensitive models (Logistic Regression, SVC, Ridge) using `Pipeline`.
- Calibrate probabilities (`CalibratedClassifierCV`) if using soft voting and base models are miscalibrated.
- Avoid including multiple highly correlated estimators; minimal marginal benefit.
- Use cross-validation to derive weights: weightᵢ = 1 / (validation_errorᵢ + ε).
- Start with equal weights; only tune if plateaued performance.
- Set `random_state` for reproducibility.
- Evaluate out-of-fold (cross-val) performance; don't rely only on training splits.

## Key Takeaways
1. Voting is a fast, low-complexity ensemble baseline.
2. Soft voting typically outperforms hard voting when probabilities are well-calibrated.
3. Weighted voting embeds performance knowledge directly into aggregation.
4. Diversity of model errors drives ensemble improvement.
5. Begin simple; iterate with weights/calibration only if justified by metrics.

## File Structure
```
Majority, Average, Weighted Average/
├── README.md                         # This file
├── ensemble_voting_classifier.ipynb  # Classification example (Raisin dataset)
├── emsemble_voting_regressor.ipynb   # Regression example (name has spelling typo)
├── Raisin_Dataset.xlsx               # Classification dataset
└── regression_home_prices.csv        # Regression dataset
```

## Dependencies
- Python 3.9+ (aligns with project root)
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (optional, for visualizations)

Install (already covered at project root):
```bash
pip install -r requirements.txt
```
Minimal subset:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Changelog
- 2025-10-01: Expanded README with examples, weighting math, best practices, metrics, and clarified file naming.
- (Earlier): Initial description of voting approaches and datasets.

---
_Last reviewed: 2025-10-01_
