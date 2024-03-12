## Statistic:
### Decomposition of Variability
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
- R-Squared = SSR /SST
  - Rsquared = 0: explain **NONE** of the variability
  - Rsquared = 1: model explain the entire variability of the data
- Adjusted R-Squared: always smaller than R-squared
  - penalize the excessive uses of variables
- F-statistic: is used for testing the overal significance of the model
  - The lower F-statistic, the closer to non-significant model
  - Prob(F-statistic): p-value for F
- OLS Assumptions:
    1. Linearity
    2. No Endogeneity
    3. Normality and homoscedasticity (normal distributed)
    4. No autocorrelation
    5. NO multicollinearity (2 or more variables have a highe observed correlation) 

### 1. Linear Regression: yhat = b0 + bi * xi + e (simple regression equation)
  - y: dependent value (for population)
  - yhat: estimated / predicted value
  - b0: intercept, constant
  - bi: slope
  -  P>|t|: p-value of hypothesis H0: b = 0
    - if > 0.05: b=0 means we should exclude that variable
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
