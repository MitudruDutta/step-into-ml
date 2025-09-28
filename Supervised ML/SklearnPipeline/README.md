# 🔗 Scikit-Learn Pipelines: Streamlining Your Workflow

## Introduction

A **scikit-learn Pipeline** chains preprocessing steps (scaling, encoding, feature generation, selection) with a final estimator into one object. This enforces a clean separation between training and inference logic, prevents data leakage, and makes experimentation (e.g. hyperparameter tuning) reproducible.

This module builds a pipeline that scales numeric raisin morphology features then trains an SVM classifier. We also cover more advanced patterns you will reuse across projects.

## 🧠 Why Use a Pipeline?

Typical manual workflow (fit scaler → transform train → transform test → fit model) is verbose and error-prone. A pipeline:

- Guarantees transforms are fit only on training folds (in CV) → prevents leakage.
- Allows hyperparameter search over preprocessing + model jointly.
- Simplifies persistence (`joblib.dump(pipeline, 'model.joblib')`).
- Provides a unified interface: `fit`, `predict`, `transform`, `score`.

## ⚙️ Basic Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Parameter access pattern for tuning: `<stepname>__<param>` e.g. `svm__C`, `svm__gamma`.

## 🧱 ColumnTransformer for Mixed Data

If you have both categorical and numeric columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

num_features = ['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']
cat_features = ['ShapeType']  # example categorical column

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

clf = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=500))
])
clf.fit(X_train, y_train)
```

## 🔍 Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.001]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
best_model = grid.best_estimator_
```

All cross-validation splits automatically refit the scaler only on the training fold → leakage avoided.

## ⚡ Performance: Caching Transformers

If heavy transformations (e.g., text vectorization, polynomial features):

```python
from tempfile import mkdtemp
from shutil import rmtree
cachedir = mkdtemp()
pipe_cached = Pipeline([...], memory=cachedir)
# After experimentation, remove cache
rmtree(cachedir)
```

## 🧩 make_pipeline vs Pipeline

```python
from sklearn.pipeline import make_pipeline
make_pipeline(StandardScaler(), SVC())  # auto step names: standardscaler, svc
```

Use `Pipeline([...])` when you want explicit, stable step names (important for param grids).

## 🛠 Custom Transformer Skeleton

```python
from sklearn.base import BaseEstimator, TransformerMixin
class LogFeature(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c + '_log'] = (X[c].clip(min=1)).map(float).ravel().__class__(X[c])  # placeholder; simpler below
        return X
```

Simpler practical version:

```python
class LogFeature(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c + '_log'] = np.log1p(X[c])
        return X
```

Add it as a step *before* scaling within a numeric sub-pipeline.

## 🧪 End-to-End Example (Raisin Dataset)

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

cols = ['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']
df = pd.read_excel('Raisin_Dataset.xlsx')
X, y = df[cols], df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro')
print('CV f1_macro mean:', cv_scores.mean())
pipe.fit(X_train, y_train)
print('Test accuracy:', pipe.score(X_test, y_test))
```

## 🧪 Probability & Threshold Adjustment

After fitting a probability-capable classifier in a pipeline:

```python
proba = pipe.predict_proba(X_test)[:,1]
# custom threshold
import numpy as np
threshold = 0.4
pred_custom = (proba >= threshold).astype(int)
```

## 🧪 Inspecting Steps & Changing Params

```python
pipe.named_steps['svm']  # access estimator
pipe.set_params(svm__C=10).fit(X_train, y_train)
```

## 📉 Avoiding Data Leakage: Anti-Pattern

Bad (leaks test distribution info):

```python
scaler = StandardScaler()
scaler.fit(X)          # uses full dataset
X_train_s = scaler.transform(X_train)
```

Good (pipeline inside CV): scaling refit per split.

## ⚠️ Common Pitfalls

- Fitting scalers / encoders outside CV loop (leakage).
- Using inconsistent column ordering at prediction time (pipeline solves this).
- Forgetting `handle_unknown='ignore'` for OHE with unseen categories.
- Passing dense high-dimensional matrices → memory bloat (use sparse if possible).
- Tuning only model hyperparameters; ignoring preprocessing hyperparameters (e.g., polynomial degree, feature selector thresholds).

## 🚀 Extensions

- Add `PolynomialFeatures` + regularization inside pipeline for controlled feature expansion.
- Integrate `SelectKBest` / `RFE` for feature selection.
- Use `Pipeline` + `CalibratedClassifierCV` if calibrated probabilities needed.
- Combine multiple feature extraction branches with `FeatureUnion` / `ColumnTransformer`.
- Persist and load pipeline with `joblib`.

## ✅ Summary

Pipelines unify preprocessing and modeling, eliminate leakage, and simplify tuning & deployment. Mastering them is foundational for reliable ML experimentation.

## 📂 Files

- `sklearnpipeline.ipynb`: Notebook implementation & examples.
- `Raisin_Dataset.xlsx`: Sample dataset.

---

Return to main index: [Root README](../../README.md)
