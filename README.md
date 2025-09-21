# Machine Learning Workspace

A curated collection of Jupyter notebooks, datasets, and small utilities exploring core supervised machine learning, model evaluation, and ensemble techniques. Each subfolder focuses on a concept or algorithm with a runnable example and starter dataset.

If you're new to this workspace, start with the Quickstart section to set up your environment on Windows.

## Simple Linear Regression

Kick off with the most fundamental model: a straight-line fit between one feature and a target.

- Notebook: `Supervised ML/Simple_linear_regression/linear_regression_single_variable.ipynb`
- Goal: Understand the relationship y ≈ a·x + b, interpret slope/intercept, and assess fit with RMSE/R².
- Steps:
  1) Open the notebook in Jupyter
  2) Run cells top-to-bottom
  3) Experiment by changing the train/test split and plotting residuals

As we progress day by day, we’ll add one concept at a time (see Daily growth plan) and keep notes in `CHANGELOG.md`.

## What is Machine Learning

- Machine Learning (ML) is about teaching computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every rule.
- Supervised learning (the focus here) means we provide examples with inputs (features) and known answers (labels) so models can learn the mapping.
- Typical workflow:
  1) Load and explore data
  2) Split into train/test sets
  3) Preprocess features (encode categories, scale numbers)
  4) Train a model
  5) Evaluate with appropriate metrics
  6) Tune/improve (regularization, better features, ensembles)

## How this repository will grow

We’ll expand from fundamentals to advanced methods incrementally:
- Data handling, splitting, and evaluation metrics
- Feature engineering (encoding, scaling) and pipelines
- Core models for regression and classification
- Regularization and optimization
- Ensemble methods (bagging, boosting, voting)

The detailed repository map will be added later as topics are introduced.

---

## Daily growth plan (incremental path)

1) Day 1: Simple Linear Regression (today)
2) Day 2: Train/Test Split and Evaluation Metrics (MSE/RMSE, R²)
3) Day 3: Multiple Linear Regression
4) Day 4: Polynomial Regression (non-linear relationships)
5) Day 5: Regularization (L1/Lasso, L2/Ridge)
6) Day 6: Logistic Regression (binary classification)
7) Day 7: Decision Trees (classification/regression)
8) Day 8: Precision/Recall, F1, Confusion Matrix
9) Day 9: ROC and AUC
10) Day 10: SVM (Support Vector Machines)
11) Day 11: Naive Bayes (text classification example)
12) Day 12: Feature Scaling & One-Hot Encoding (deep dive)
13) Day 13: Scikit-Learn Pipelines
14) Day 14: Random Forest (bagging)
15) Day 15: Gradient Boosting
16) Day 16: XGBoost
17) Day 17: Voting Ensembles (majority/weighted)

We’ll keep this flexible—topics may shift as needed. See `CHANGELOG.md` for what got added each day.

---

## Quickstart (Windows, cmd.exe)

Prerequisites:
- Python 3.9+ installed and added to PATH (3.8+ should also work)
- Git (optional)

Create and activate a virtual environment, then install common packages:

```batch
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn xgboost openpyxl jupyter ipykernel
python -m ipykernel install --user --name ml-workspace --display-name "Python (ml-workspace)"
```

Run notebooks:

```batch
jupyter notebook
```

Tips:
- On Windows, if `xgboost` fails to build, ensure you are on a recent Python and pip. Prebuilt wheels are provided for supported versions.
- Excel files (xlsx) are read with `openpyxl`.

---

## How to push updates daily (Git, Windows cmd)

- First time: create a GitHub repo and push only Simple Linear Regression (optional)

```batch
git status --no-pager
git add "Supervised ML/Simple_linear_regression"
git commit -m "Day 1: Add Simple Linear Regression"
REM Set your GitHub repo URL below
REM git remote add origin https://github.com/<your-username>/machine-learning-workspace.git
git branch -M main
git push -u origin main
```

- Daily update afterwards (example):

```batch
git pull
REM Add changed notebooks/data
git add -A
git commit -m "Day %DATE%: Update with <topic>"
git push
```

If you prefer to push the entire repository from the start, replace the selective `git add` with `git add -A` in the first-time block.

---

## How to use this workspace

1) Pick a topic folder and open the notebook (.ipynb) in Jupyter.
2) Ensure your kernel is the `Python (ml-workspace)` you created.
3) Execute cells top-to-bottom. Each notebook includes code comments guiding you through preprocessing, training, and evaluation.

Common libraries used throughout:
- pandas, numpy for data handling
- scikit-learn for models, preprocessing, pipelines, and metrics
- matplotlib, seaborn for plots
- xgboost for gradient boosted trees (in XGBoost/ and some ensemble exercises)

If a notebook refers to a dataset by relative path, run Jupyter from the repository root so paths resolve correctly.

---

## Reproducibility & environment tips

- Set random seeds for scikit-learn models where available (e.g., `random_state=42`).
- Save your environment for sharing:
  - Freeze pip packages: `pip freeze > requirements.txt`
  - Or export conda env: `conda env export > environment.yml`
- Add a new kernel with `ipykernel` if you create multiple virtual environments.

---

## Suggested learning path

1) Supervised ML basics: simple/ multiple linear regression, logistic regression
2) Preprocessing: one-hot encoding, scaling, train/test split, pipelines
3) Model evaluation: precision/recall, F1, ROC/AUC
4) Algorithms: decision trees, SVM, Naive Bayes
5) Advanced/Ensembles: gradient boosting, random forests, voting ensembles, XGBoost

---

## Troubleshooting

- Package install issues (Windows):
  - Upgrade build tooling: `pip install --upgrade pip setuptools wheel`
  - If a package fails to build, try a compatible Python (e.g., 3.10 or 3.11) and ensure you use the latest pip.
- Jupyter cannot find the kernel:
  - Re-run: `python -m ipykernel install --user --name ml-workspace --display-name "Python (ml-workspace)"`
- Excel read errors:
  - Ensure `openpyxl` is installed.

---

## License

This repository currently has no explicit license file. If you intend to share or reuse the content publicly, consider adding a `LICENSE` file (e.g., MIT) to define permissions.

---

## Acknowledgements

Datasets included here are for educational purposes. Notebook code generally uses standard Python data science libraries.

Happy learning and experimenting!
