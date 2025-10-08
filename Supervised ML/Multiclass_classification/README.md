# 🌸 Multiclass Classification: Classifying Iris Flowers

## Introduction

**Multiclass Classification** is a type of classification task where the goal is to categorize instances into one of **three or more** distinct classes. This is an extension of binary classification, where there are only two possible outcomes.

This project provides a clear and practical guide to multiclass classification by building a model to classify different species of Iris flowers based on their sepal and petal measurements.

## 🧠 Theory

### How Do Binary Classifiers Handle Multiple Classes?

Many powerful classification algorithms, like Logistic Regression and Support Vector Machines, are inherently binary. To use them for multiclass problems, they employ strategies like:

-   **One-vs-Rest (OvR)** or **One-vs-All (OvA)**: This is the most common strategy. It works by training a separate binary classifier for each class. For a problem with N classes, N classifiers are trained. For example, to classify Iris flowers (Setosa, Versicolor, Virginica), it would train:
    1.  A classifier for `Setosa` vs. `(Versicolor + Virginica)`.
    2.  A classifier for `Versicolor` vs. `(Setosa + Virginica)`.
    3.  A classifier for `Virginica` vs. `(Setosa + Versicolor)`.
    When a new flower needs to be classified, it's run through all three classifiers, and the one that outputs the highest confidence score wins.

-   **One-vs-One (OvO)**: This strategy involves training a binary classifier for every pair of classes. For N classes, it trains `N * (N-1) / 2` classifiers. This can be more computationally expensive but can be more accurate for some algorithms like SVM.

Algorithms like Decision Trees, Random Forest, and Naive Bayes are naturally capable of handling multiclass problems without these strategies.

## 📊 Dataset

-   **The Iris Flower Dataset**: This is a classic and famous dataset in machine learning.
    -   **Features**: It contains four features (all in centimeters):
        -   `sepal length`
        -   `sepal width`
        -   `petal length`
        -   `petal width`
    -   **Target**: The species of the Iris flower, which falls into one of three classes:
        -   `Setosa`
        -   `Versicolor`
        -   `Virginica`

## 🛠 Implementation Steps

1.  **Load Data**: The Iris dataset is loaded directly from `scikit-learn`.
2.  **Exploratory Data Analysis (EDA)**: A pair plot is used to visualize the relationships between the features and the separability of the classes.
3.  **Train-Test Split**: The data is split into training and testing sets.
4.  **Model Training**: A `LogisticRegression` model is trained. By default, scikit-learn will use OvR for binary problems and multinomial for some solvers; you can force a choice via `multi_class`.
5.  **Model Evaluation**: The model's performance is evaluated using:
    -   An **accuracy score**.
    -   A **multiclass confusion matrix** to see where the model is making errors.
    -   A **classification report** to review precision, recall, and F1-score for each class.

## ✅ Minimal Example (Logistic Regression)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)

# Split (stratify preserves class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Model: try multinomial LR (works well for softmax-like multiclass)
clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=500,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred, digits=3))

# Confusion matrix plot
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Iris – Multiclass Confusion Matrix')
plt.tight_layout(); plt.show()
```

### Notes
- `multi_class='multinomial'` with `solver='lbfgs'` often performs well; `multi_class='ovr'` is also valid and may be faster for some models.
- If convergence warnings appear, increase `max_iter`.
- For non-linearly separable data, try SVM with RBF kernel and compare OvR vs OvO.

### Metrics averaging (beyond accuracy)
- **Macro F1**: unweighted mean across classes (treats all classes equally).
- **Weighted F1**: accounts for class support (useful under class imbalance).
- **Micro F1**: aggregates contributions over all classes (equivalent to accuracy for multiclass).

```python
from sklearn.metrics import f1_score
print('Macro F1:', f1_score(y_test, y_pred, average='macro'))
print('Weighted F1:', f1_score(y_test, y_pred, average='weighted'))
```

## 🔄 Comparing OvR vs OvO with SVM

For algorithms like SVM, you can explicitly choose between One-vs-Rest and One-vs-One strategies:

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-vs-Rest (OvR)
ovr_clf = OneVsRestClassifier(SVC(kernel='rbf', gamma='auto', random_state=42))
ovr_clf.fit(X_train_scaled, y_train)
ovr_pred = ovr_clf.predict(X_test_scaled)
print('OvR Accuracy:', accuracy_score(y_test, ovr_pred))

# One-vs-One (OvO)
ovo_clf = OneVsOneClassifier(SVC(kernel='rbf', gamma='auto', random_state=42))
ovo_clf.fit(X_train_scaled, y_train)
ovo_pred = ovo_clf.predict(X_test_scaled)
print('OvO Accuracy:', accuracy_score(y_test, ovo_pred))

# Note: By default, SVC uses OvO for multiclass
default_svc = SVC(kernel='rbf', gamma='auto', random_state=42)
default_svc.fit(X_train_scaled, y_train)
default_pred = default_svc.predict(X_test_scaled)
print('Default SVC (OvO) Accuracy:', accuracy_score(y_test, default_pred))
```

## ⚠️ Common Pitfalls

- **Forgetting to stratify**: Use `stratify=y` in `train_test_split` to preserve class proportions, especially important when classes are imbalanced.
- **Not scaling features**: Algorithms like SVM and Logistic Regression are sensitive to feature scale. Always use `StandardScaler` or `MinMaxScaler`.
- **Using accuracy for imbalanced data**: If one class dominates, accuracy can be misleading. Use macro/weighted F1 or confusion matrix instead.
- **Ignoring convergence warnings**: Logistic Regression may not converge with default `max_iter=100`. Increase it or scale your features.
- **Mixing up averaging methods**: 
  - Use `macro` when all classes are equally important.
  - Use `weighted` when you want to account for class imbalance.
  - `micro` is just accuracy for multiclass.

## 🚀 Next Steps

- **Try other algorithms**: Decision Trees, Random Forest, Naive Bayes (all handle multiclass naturally).
- **Feature engineering**: Create interaction features or polynomial features.
- **Cross-validation**: Use K-Fold or Stratified K-Fold for more robust evaluation.
- **Hyperparameter tuning**: Use GridSearchCV or RandomizedSearchCV to optimize `C`, `gamma` (SVM), or `max_iter` (Logistic Regression).
- **Multiclass ROC curves**: Plot One-vs-Rest ROC curves for each class.
- **Handle class imbalance**: Try techniques like SMOTE or class weighting if your dataset has imbalanced classes.

## 📂 Files

-   Notebook: [`multiclass_class.ipynb`](multiclass_class.ipynb)
-   README: This file
