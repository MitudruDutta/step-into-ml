# 📧 Naive Bayes Classifier: SMS Spam Detection

## Introduction

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**. It is particularly well‑suited for text classification tasks like spam detection due to its efficiency and effectiveness with high‑dimensional sparse data.

This project builds an SMS spam classifier distinguishing between legitimate messages (*ham*) and unwanted messages (*spam*).

## 🧠 Theory Refresher

### Bayes' Theorem
`P(A|B) = P(B|A) P(A) / P(B)`

For spam filtering:
`P(spam | message) ∝ P(message | spam) * P(spam)`

Naive Bayes assumes **conditional independence** of tokens (words) given the class:
`P(message | class) = Π P(word_i | class)` (in practice using counts and smoothing).

### Multinomial Naive Bayes (Used Here)
Suitable when features are discrete term counts (Bag‑of‑Words or TF‑IDF). The model estimates class priors `P(class)` and per‑class conditional likelihoods for each token with **Laplace (add‑α) smoothing**:
`P(word|class) = (count(word,class) + α) / (Σ counts(class) + α * |V|)`
Where α = `alpha` parameter (default 1.0 in scikit‑learn).

### Other Variants
- **BernoulliNB** – binary word presence/absence (can work well for short texts).
- **ComplementNB** – often better on imbalanced text datasets.
- **GaussianNB** – continuous features (not typical for raw text).

## 🧹 Text Vectorization & Preprocessing
This example uses a simple **Bag‑of‑Words** representation via `CountVectorizer`.
Consider upgrading to `TfidfVectorizer`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2)
```
Typical optional cleaning steps (not always required):
- Lowercasing (handled by vectorizer).
- Removing punctuation / numbers (vectorizer token pattern controls this).
- Stopword removal (`stop_words='english'`).
- Handling emojis / URLs (custom preprocessor if needed).

Avoid aggressive stemming unless evaluated (may harm precision on short SMS tokens).

## 📊 Dataset

- **File**: `spam.csv`
- **Columns**:
  - `Category`: label ('ham' or 'spam')
  - `Message`: raw SMS text

(Ensure encoding is UTF‑8; some public spam datasets ship with extraneous unnamed columns—drop unused columns as needed.)

## 🛠 Implementation Steps
1. Load and inspect dataset; drop duplicate rows; check class balance.
2. Split using `train_test_split(..., stratify=Category)` to preserve class ratios.
3. Vectorize text (`CountVectorizer` or `TfidfVectorizer`).
4. Train `MultinomialNB` (optionally tune `alpha`).
5. Evaluate with precision, recall, F1, confusion matrix, **and** PR AUC (spam often minority class).
6. Package steps in a `Pipeline` for cleanliness & reuse.
7. (Optional) Persist model and vectorizer with `joblib`.

## 🚀 Minimal Pipeline Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('spam.csv')
# Keep only needed columns (example dataset may have extras)
df = df[['Category','Message']].drop_duplicates()

X = df['Message']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2)),
    ('nb', MultinomialNB(alpha=0.7))  # tune alpha via grid/CV
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Confusion Matrix'); plt.show()

# Precision-Recall Curve (treat spam as positive class)
import numpy as np
proba = pipe.predict_proba(X_test)[:, list(pipe.classes_).index('spam')]
prec, rec, thr = precision_recall_curve((y_test=='spam').astype(int), proba)
ap = average_precision_score((y_test=='spam').astype(int), proba)
plt.plot(rec, prec, label=f'AP={ap:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.show()

# Quick threshold optimization (F1)
from sklearn.metrics import f1_score
scores = [(t, f1_score((y_test=='spam').astype(int), proba >= t)) for t in np.linspace(0.1,0.9,17)]
print('Best threshold (by F1):', max(scores, key=lambda x: x[1]))
```

## 📏 Key Evaluation Metrics
Because spam datasets are usually imbalanced:
- **Precision (spam)**: Of predicted spam, how many are actually spam? (Avoid flagging ham.)
- **Recall (spam)**: Of actual spam, how many did we catch? (Avoid missing spam.)
- **F1 (spam)**: Harmonic mean; good balance.
- **Average Precision / PR AUC**: Threshold‑independent summary for minority positive class.
- **Confusion Matrix**: Concrete error counts.

## ⚙️ Hyperparameter Ideas
| Component | Parameter | Rationale |
|----------|-----------|-----------|
| Vectorizer | ngram_range | Include bigrams for phrases ("free entry") |
| Vectorizer | min_df / max_df | Remove very rare or very common noise tokens |
| Vectorizer | stop_words | Remove uninformative words |
| Vectorizer | sublinear_tf=True | Dampens overly frequent tokens |
| NB | alpha | Laplace smoothing strength (try 0.1–2.0) |

Search via `GridSearchCV`:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
  'tfidf__ngram_range': [(1,1),(1,2)],
  'tfidf__min_df': [1,2,3],
  'nb__alpha': [0.3,0.7,1.0,1.5]
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)
```

## ⚠️ Common Pitfalls
- Treating **accuracy** as sufficient (may be inflated by majority ham class).
- Not stratifying on split → distorted class ratios in small test sets.
- Using raw counts when TF‑IDF improves separation (test both).
- Over-cleaning (removing tokens that signal spam like numbers or special offers).
- High false positives degrading user trust—threshold tuning essential.
- Forgetting `alpha` tuning—default 1.0 may not be optimal.

## 🔄 Extensions
- Switch to `ComplementNB` for imbalanced data.
- Add character n‑grams (helps with obfuscation like "fr33 w1n").
- Apply **SMOTE** or class weighting (usually NB handles imbalance decently via priors, but explore).
- Ensemble with linear SVM or Logistic Regression for comparison.
- Deploy: persist pipeline (`joblib.dump(pipe, 'spam_nb.joblib')`).
- Add fast inference script / REST API stub.

## 🔐 Handling HTML / URLs / Emojis
Add a custom preprocessor:
```python
import re
url_re = re.compile(r'https?://\S+|www\.\S+')
html_re = re.compile(r'<.*?>')

def clean(text):
    text = url_re.sub(' URL ', text)
    text = html_re.sub(' ', text)
    return text
TfidfVectorizer(preprocessor=clean, ...)
```

## ✅ Key Takeaways
- Naive Bayes is a strong baseline for text: fast, low memory, good with sparse features.
- Proper evaluation (precision/recall/F1/AP) matters more than raw accuracy under imbalance.
- Vectorization & smoothing choices (TF‑IDF + tuned alpha) materially affect performance.
- Use a Pipeline to avoid leakage and simplify deployment.

## 📂 Files
- `smsclassifier.ipynb`: Notebook.
- `spam.csv`: Dataset.
- `README.md`: This guide.

---
### 🧾 Summary
You implemented a spam detector with Multinomial Naive Bayes, explored vectorization & smoothing, evaluated with classification & PR metrics, and outlined extensions (ComplementNB, character n‑grams, calibration, deployment). This establishes a reliable NLP baseline before moving to heavier models.
