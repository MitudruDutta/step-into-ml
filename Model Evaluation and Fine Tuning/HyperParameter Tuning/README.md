# 🔧 Hyperparameter Tuning: GridSearchCV & RandomizedSearchCV

## Why tune?
Model performance often hinges on a few key hyperparameters (e.g., tree depth, SVM kernel, regularization strength). Tuning finds a good combination via cross‑validation without leaking test data.

- Grid Search: exhaustively evaluates all combinations in a parameter grid.
- Randomized Search: samples a fixed number of combinations from given ranges; faster on large spaces.

Use cross‑validation (cv) to estimate generalization; refit the best model on the full training set.

## Files in this folder
- `gridsearchCV.ipynb`: End‑to‑end examples using GridSearchCV (e.g., DecisionTree, SVM).
- `randomisedsearchCV.ipynb`: Hyperparameter search with RandomizedSearchCV on the same/similar models.
- `README.md`: This guide.

## Quick start
1) Open the notebook you want (`gridsearchCV.ipynb` or `randomisedsearchCV.ipynb`).
2) Run cells to generate data, split, define parameter grids, and search.
3) Inspect `best_params_`, `best_score_`, and evaluate the `best_estimator_` on a hold‑out test set.

Tip: Keep preprocessing inside a scikit‑learn `Pipeline` to avoid data leakage.

## Minimal examples

Grid search for a Decision Tree:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15]
}

clf = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',        # change per task (e.g., 'f1', 'roc_auc')
    n_jobs=1,                  # Windows tip: keep 1 if you hit freezing issues
    return_train_score=False,
    refit=True
)
clf.fit(X_train, y_train)
print('Best params:', clf.best_params_)
print('CV best score:', clf.best_score_)
print('Test accuracy:', clf.best_estimator_.score(X_test, y_test))
```

Randomized search for an SVM (no SciPy required — use discrete lists):

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm

param_distributions = {
    'C': [0.1, 1, 3, 10, 30, 100],
    'kernel': ['linear', 'rbf'],
    # for 'rbf' only; ignored for 'linear'. Include a few values to try.
    'gamma': ['scale', 'auto', 0.01, 0.03, 0.1, 0.3, 1.0]
}

svm_model = svm.SVC()
rs = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=param_distributions,
    n_iter=15,                # how many samples from the space
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=1,                 # Windows tip
    refit=True
)
rs.fit(X_train, y_train)
print('Best params:', rs.best_params_)
print('CV best score:', rs.best_score_)
print('Test accuracy:', rs.best_estimator_.score(X_test, y_test))
```

Note: If you prefer continuous ranges, you can use SciPy distributions (e.g., `scipy.stats.loguniform`), but SciPy isn’t required for the included notebooks.

## Good practices
- Always tune on training data with cross‑validation; reserve a separate test set for final evaluation.
- Use `Pipeline` to include scaling/encoding so CV reflects full preprocessing.
- Pick `scoring` aligned with your goal (e.g., `roc_auc` for ranking, `f1` under imbalance).
- Set `random_state` where available for reproducible searches.
- Start with RandomizedSearchCV for large spaces; finalize with a narrow GridSearchCV if needed.

## Common pitfalls
- Data leakage: fitting scalers/encoders before CV. Fix with `Pipeline`.
- Overfitting to CV: keep a final test set; consider nested CV for rigorous comparisons.
- Long runtimes: reduce `cv`, `n_iter`, or parameter ranges; leverage `n_jobs` but note Windows may require `n_jobs=1` in notebooks.

## Next steps
- Try different `scoring` metrics and compare.
- Add stratified CV for classification: `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Extend to other models: Logistic Regression (`C`), Random Forest (`n_estimators`, `max_depth`), Gradient Boosting (`learning_rate`, `max_depth`).

