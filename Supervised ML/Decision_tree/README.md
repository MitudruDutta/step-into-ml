# 🌳 Decision Tree Classifier: Predicting Salaries

## Introduction

A Decision Tree is a versatile and intuitive supervised learning algorithm that can be used for both classification and regression tasks. It works by creating a tree-like model of decisions, where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.

This project demonstrates how to build a decision tree to predict whether a person's salary is more than $100k based on their company, job role, and degree.

## 🧠 Theory

### Key Concepts

A decision tree makes predictions by learning simple decision rules inferred from the data features. It splits the data into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

-   **Root Node**: The top-most node, representing the entire dataset.
-   **Internal Nodes**: Nodes that represent a test on a feature.
-   **Branches**: The outcomes of the test (e.g., 'company is google' or 'company is not google').
-   **Leaf Nodes**: The terminal nodes that represent the final class label (e.g., `salary > 100k` or `salary <= 100k`).

The splits are chosen to maximize the "purity" of the resulting nodes. Common criteria for measuring purity are:
-   **Gini Impurity**: Measures the frequency at which any element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
-   **Entropy**: A measure of information disorder. The goal is to choose splits that reduce entropy.

### Formulas

- Gini impurity: Gini = 1 − Σ pᵢ² (where pᵢ is the fraction of samples of class i)
- Entropy: Entropy = −Σ pᵢ log₂(pᵢ)

## 📊 Dataset

-   **File**: `salaries.csv`
-   **Description**: A dataset containing categorical features about job positions.
    -   `company`: The company the person works for (e.g., google, facebook).
    -   `job`: The person's job title (e.g., sales executive, business manager).
    -   `degree`: The person's educational degree (bachelors, masters).
    -   `salary_more_then_100k`: The target variable (1 if salary > $100k, 0 otherwise).

## 🛠 Implementation Steps

1.  **Load Data**: The `salaries.csv` dataset is loaded using pandas.
2.  **Data Preprocessing**: Since decision trees require numerical input, the categorical features (`company`, `job`, `degree`) are converted into a numerical format using **One-Hot Encoding** and **Label Encoding**.
3.  **Train-Test Split**: The dataset is split into training and testing sets.
4.  **Model Training**: A `DecisionTreeClassifier` from `scikit-learn` is trained on the training data.
5.  **Model Evaluation**: The model's accuracy is evaluated on the test set.
6.  **Tree Visualization**: The trained decision tree is plotted to visualize the decision rules the model has learned.
7.  **Prediction**: The model is used to predict the salary category for a new candidate.

## ✅ Key Takeaways

-   **Strengths**: Easy to understand and interpret. The visual representation is very intuitive. Can handle both numerical and categorical data.
-   **Limitations**: Prone to **overfitting**, especially with deep trees. Can be unstable, as small variations in the data might result in a completely different tree being generated.
-   **Use Cases**: Customer churn prediction, loan default prediction, and medical diagnosis.

## 📂 Files

-   `decisiontree.ipynb`: The Jupyter Notebook with the Python code and explanations.
-   `salaries.csv`: The dataset used for the classification task.
