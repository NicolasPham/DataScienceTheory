

### 1. Linear Regression: yhat = b0 + bi * xi + e (simple regression equation)
  - y: dependent value (for population)
  - yhat: estimated / predicted value
  - b0: intercept, constant
  - bi: slope
#### Python:
``` python
  import statsmodel.api as sm
# Simple linear regression
  x1 = df[<column>]
  y = df[<column>]
  x = sm.add_constant(x)
  result = sm.OLS(y,x).fit()

# Get the summary of result
  result.summary()
```
