## Statistic:
### 1. Decomposition of Variability
- Sum of squares total (SST / TSS)
  - measures the total variability of the datasets
  - SUM((yi - mean(y))^2)
- Sum of squares regression (SSR / ESS - explained sum of squares)
  - measure the explained variability by your line
  - SUM((yhat - mean(y))^2)
- Sum of squares error (SSE / RSS - residual sum of squares)
  - measure the unexplained variability by the regression
  - SUM(e^2)
- Connection: SST = SSR + SSE

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
