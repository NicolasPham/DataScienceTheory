## Statistic:
### Decomposition of Variability
- Sum of squares total (SST / TSS)
  - measures the total variability of the datasets
  - $SUM((yi - mean(y))^2)$
- Sum of squares regression (SSR / ESS - explained sum of squares)
  - measure the explained variability by your line
  - $SUM((yhat - mean(y))^2)$
- Sum of squares error (SSE / RSS - residual sum of squares)
  - measure the unexplained variability by the regression
  - $SUM(e^2)$
- Connection: SST = SSR + SSE
- R-Squared = SSR /SST
  - Rsquared = 0: explain **NONE** of the variability
  - Rsquared = 1: model explain the entire variability of the data
- Adjusted R-Squared: always smaller than R-squared
  - penalize the excessive uses of variables
  - $R^2_{adj.} = 1 - (1-R^2)\frac{n-1}{n-p-1}$
- F-statistic: is used for testing the overal significance of the model
  - The lower F-statistic, the closer to non-significant model
  - Prob(F-statistic): p-value for F
- OLS Assumptions:
    1. Linearity
    2. No Endogeneity
    3. Normality and homoscedasticity (normal distributed)
    4. No autocorrelation
    5. NO multicollinearity (2 or more variables have a highe observed correlation) 
### 1. Linear Regression: $yhat = b0 + bi * xi + e$ (simple regression equation)
  - y: dependent value (for population)
  - yhat: estimated / predicted value
  - b0: intercept, constant
  - bi: slope
  -  P>|t|: p-value of hypothesis H0: b = 0
    - if > 0.05: b=0 means we should exclude that variable
#### Python:
``` python
  import statsmodel.api as sm
  from sklearn.linear_model import LinearRegression
# Simple linear regression
  x1 = df[<column>]
  y = df[<column>]
  x = sm.add_constant(x)
  result = sm.OLS(y,x).fit()

# Get the summary of result
  result.summary()

# Make predictions:
  predictions = result.predict(new_df)

# Using Sklearn
  x_matrix = x.values.reshape(-1,1) #reshape to 2D array for "Single Linear Regression" only
  reg = LinearRegression()
  reg.fit(x_matrix,y) # x, y has to be in this order as inputs, target

  reg.score(x_matrix,y) #return R-Squared
  reg.coef_ #return coefficient
  reg.intercept_ #return the intercept
  reg.predict(new_df) #predict the output
```
#### Feature Selection: F_regression:
- Create a simple linear regressions of each feature and dependent variable

```python
from sklearn.feature_selection import f_regression
f_regression(x,y) #return arrays with first value is F_statistic, second value is p-value
p-values = f_regression(x,y)[1].round(3)
```
#### Standardization:
- Scale the features so the model treat them equally
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

#For prediction:
new_data_scaled = scaler.transform(new_data)
reg.predict(new_data_scaled)
```
#### Train Test Split
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state = 163)
```

