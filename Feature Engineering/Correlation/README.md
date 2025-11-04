# Feature Engineering Using Correlation

## Overview
Feature engineering using correlation is a technique to select the most relevant features for your machine learning model by analyzing the correlation between features and the target variable. This approach helps improve model performance, reduce overfitting, and decrease training time by eliminating irrelevant or weakly correlated features.

## What is Correlation?
Correlation measures the strength and direction of the relationship between two variables. The correlation coefficient ranges from -1 to 1:
- **1**: Perfect positive correlation (as one variable increases, the other increases proportionally)
- **0**: No correlation (no linear relationship)
- **-1**: Perfect negative correlation (as one variable increases, the other decreases proportionally)

## Why Use Correlation for Feature Selection?
1. **Dimensionality Reduction**: Removes irrelevant features, making the model simpler and faster
2. **Improved Performance**: Focuses on features that have strong relationships with the target
3. **Reduced Overfitting**: Fewer features means less chance of the model learning noise
4. **Better Interpretability**: Easier to understand which features drive predictions

## Dataset
This example uses a house price prediction dataset (`home_prices.csv`) with the following features:
- `area_sqr_ft`: Area of the house in square feet
- `bedrooms`: Number of bedrooms
- `color`: Color of the house (categorical)
- `price_lakhs`: House price in lakhs (target variable)

## Implementation Steps

### 1. Load and Explore Data
```python
import pandas as pd

df = pd.read_csv("home_prices.csv")
df.head()
```

### 2. Handle Categorical Variables
Since correlation works with numerical data, we need to encode categorical variables using one-hot encoding:
```python
df_encoded = pd.get_dummies(df, columns=["color"], drop_first=True)
```

**Note**: `drop_first=True` is used to avoid multicollinearity by dropping one of the dummy variables.

### 3. Calculate Correlation Matrix
```python
cm = df_encoded.corr()
```
This creates a correlation matrix showing the relationship between all pairs of features.

### 4. Extract Correlations with Target Variable
```python
cm_price = abs(cm["price_lakhs"])
```
We use the absolute value to capture both positive and negative correlations, as both indicate a strong relationship.

### 5. Select Features Based on Threshold
```python
selected_features = cm_price[cm_price > 0.2].index.drop("price_lakhs")
```
This selects features with correlation > 0.2 with the target variable, excluding the target itself.

**Common Correlation Thresholds:**
- **> 0.7**: Strong correlation
- **0.4 - 0.7**: Moderate correlation
- **0.2 - 0.4**: Weak correlation
- **< 0.2**: Very weak/no correlation

The threshold should be chosen based on your specific use case and domain knowledge.

### 6. Train Model with Selected Features
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

X = df[selected_features]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("R2:", r2, "MSE:", mse)
```

## Key Concepts

### Correlation vs Causation
- **Correlation** indicates that two variables move together
- **Causation** means one variable directly causes changes in another
- High correlation doesn't imply causation!

### Multicollinearity
When features are highly correlated with each other (not just the target), it can cause problems:
- Makes coefficient interpretation difficult
- Increases variance of coefficient estimates
- Solution: Remove one of the highly correlated features

### Limitations of Correlation-Based Feature Selection
1. **Only captures linear relationships**: May miss non-linear relationships
2. **Ignores feature interactions**: Doesn't consider how features work together
3. **Sensitive to outliers**: Outliers can skew correlation values
4. **Not suitable for all problems**: Works best with numerical data and linear relationships

## Best Practices
1. **Always visualize**: Use heatmaps to visualize the correlation matrix
2. **Domain knowledge**: Combine correlation analysis with domain expertise
3. **Test different thresholds**: Experiment with different correlation thresholds
4. **Consider alternatives**: For non-linear relationships, consider mutual information or tree-based feature importance
5. **Validate results**: Always evaluate model performance with selected features

## Alternative Feature Selection Methods
- **Mutual Information**: Captures non-linear relationships
- **Chi-Square Test**: For categorical features
- **Recursive Feature Elimination (RFE)**: Model-based selection
- **Feature Importance from Tree Models**: Random Forest, XGBoost feature importance
- **L1 Regularization (Lasso)**: Automatic feature selection through regularization

## Files in This Directory
- `featureusingcorr.ipynb`: Jupyter notebook with complete implementation
- `home_prices.csv`: Dataset for house price prediction
- `README.md`: This documentation file

## Conclusion
Correlation-based feature selection is a simple yet effective technique for identifying relevant features in your dataset. It works best for linear relationships and should be combined with other techniques for comprehensive feature engineering.

## Next Steps
- Experiment with different correlation thresholds
- Try visualizing the correlation matrix using seaborn heatmap
- Compare model performance with all features vs selected features
- Explore other feature selection techniques for comparison
