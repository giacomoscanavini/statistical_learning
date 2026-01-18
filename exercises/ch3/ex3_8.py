"""
Exercises for 
An Introdution to Statistical Learning 
Chapter 3: Linear Regression
Exercise: 8


Auto Data Set - description:
    Engine related and other information for 392 vehicles
    mpg          : Miles per gallon
    cylinders    : Number of cylinders between 4 and 8
    displacement : Engine displacement (cu. inches)
    horsepower   : Engine horsepower
    weight       : Vehicle weight (lbs.)
    acceleration : Time to accelerate from 0 to 60 mph (sec.)
    year         : Model year (modulo 100)
    origin       : Origin of car (1. American, 2. European, 3. Japanese)
    name         : Vehicle name
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

path_to_data = "/mnt/c/Users/scana/Desktop/datasets_AISL"
auto = pd.read_csv(os.path.join(path_to_data, 'Auto.csv'))

# Convert to numeric and drop missing values
for col in ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']:
    auto[col]=pd.to_numeric(auto[col], errors='coerce')
auto_clean=auto.dropna()

#(a) Use the sm.OLS() function to perform a simple linear regression
#with mpg as the response and horsepower as the predictor. 
#Comment on the output. For example:
#i. Is there a relationship between the predictor and the response?

X = auto_clean['horsepower'] # independent variable
y = auto_clean['mpg']        # dependent variable

X     = sm.add_constant(X)   # add vector of 1s to X for intercept
model = sm.OLS(y, X)

results = model.fit()
print(results.summary())

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.606
Model:                            OLS   Adj. R-squared:                  0.605
Method:                 Least Squares   F-statistic:                     599.7
Date:                Sun, 18 Jan 2026   Prob (F-statistic):           7.03e-81
Time:                        21:31:44   Log-Likelihood:                -1178.7
No. Observations:                 392   AIC:                             2361.
Df Residuals:                     390   BIC:                             2369.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         39.9359      0.717     55.660      0.000      38.525      41.347
horsepower    -0.1578      0.006    -24.489      0.000      -0.171      -0.145
==============================================================================
Omnibus:                       16.432   Durbin-Watson:                   0.920
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               17.305
Skew:                           0.492   Prob(JB):                     0.000175
Kurtosis:                       3.299   Cond. No.                         322.
==============================================================================

The null hypothesis states the there is no relationship between predictor and response
The null hypothesis is rejected as we notice a p-value associated to "horsepower" that is much lower than 0.05
"""

#ii. How strong is the relationship between the predictor and the response?
"""
The strength of the effect is given by the following factors: 
- t-score which is how many standard errors the regression coefficient differs from 0
    t ~ 24.5 which is high, implying a strong effect

- R-squared which is the amout of total variance in the response that is explained by the model
    R-squared ~ 0.6 which corresponds to 60% of total variance expalained
"""

#iii. Is the relationship between the predictor and the response positive or negative?
"""
The relationship between the predictor and the response is negative as the estimated coefficient is negative
That translates to more mpg given lower values of engine horsepower and vice versa
"""

#iv. What is the predicted mpg associated with a horsepower of 98? 
#What are the associated 95 % confidence and prediction intervals?

# Prediction
def predict_y(x, params):
    return params[0] + x * params[1]

params = [results.params['const'], results.params['horsepower']]
x0 = 98
y0 = predict_y(x0, params)
print(f"Predicted value for: horsepower = {x0}, y = {y0}")

# For the intervasl we need to consider:
x_mean = np.sum(auto_clean['horsepower']) / results.nobs
y_mean = np.sum(auto_clean['mpg'])        / results.nobs

# tss = np.sum([(y - y_mean)**2 for y in auto_clean['mpg']])
# ess = np.sum([(predict_y(x, params) - y_mean)**2 for x in auto_clean['horsepower']])
# rss = np.sum([(y - predict_y(x, params))**2      for x,y in zip(auto_clean['horsepower'], auto_clean['mpg'])])
ess = results.ess # ESS
rss = results.ssr # RSS
tss = ess + rss   # TSS
s2 = rss / results.df_resid # estimator for sigma-squared

Sxx    = np.sum([(x - x_mean)**2 for x in auto_clean['horsepower']])
Sxy    = np.sum([(x - x_mean)*(y - y_mean) for x,y in zip(auto_clean['horsepower'], auto_clean['mpg'])])
Syy    = np.sum([(y - y_mean)**2 for x in auto_clean['mpg']])
var_y  = s2 * (1/results.nobs + (x_mean**2 + x0**2 - 2*x0*x_mean)/Sxx    ) # Variance for mean prediction
var_y0 = s2 * (1/results.nobs + (x_mean**2 + x0**2 - 2*x0*x_mean)/Sxx + 1) # Variance for single prediction
# Confidence interval 95%
z_025 = 1.96
ci_95 = [float(y0 - z_025 * np.sqrt(var_y)),  float(y0 + z_025 * np.sqrt(var_y))]
# Prediction Interval 95%
pi_95 = [float(y0 - z_025 * np.sqrt(var_y0)), float(y0 + z_025 * np.sqrt(var_y0))]
print(f"95% Confidence interval: {[round(x, 2) for x in ci_95]}")
print(f"95% Prediction interval: {[round(x, 2) for x in pi_95]}")


#(b) Plot the response and the predictor in a new set of axes ax. Use
#the ax.axline() method or the abline() function defined in the
#lab to display the least squares regression line.
r2   = 1 - rss / tss
corr = round(np.sqrt(r2), 2) # only valid for simple linear regression 
if results.params['horsepower'] > 0: pass
else: corr = -corr

fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.scatter(auto_clean['horsepower'], auto_clean['mpg'], color='black', label=f'N = {int(results.nobs)}')
x_grid = np.linspace(np.min(auto_clean['horsepower']), np.max(auto_clean['horsepower']), 100)
ax.plot(x_grid, [predict_y(x, params) for x in x_grid], color='red', label=f'OLS: r = {corr}')
ax.set_xlabel('horsepower')
ax.set_ylabel('mpg')
ax.legend()
plt.show()


#(c) Produce some of diagnostic plots of the least squares regression
#fit as described in the lab. Comment on any problems you see with the fit.

# Residuals vs Fitted values to spot non-linearity
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.scatter([predict_y(x, params) for x in auto_clean['horsepower']], results.resid, color='black', label=f'N = {int(results.nobs)}')
ax.axhline(0, lw=1, ls=':', color='black', alpha=0.5)
ax.set_xlabel('fitted value')
ax.set_ylabel('residual')
ax.legend()
plt.show()

"""
The plot shows a lot of points that are not distributed around 0 with clear non-flat pattern in the residuals plot
This finding suggests a non-linear relationship might be present between X and Y
Fitting a new model with logY or a concave up transformation of Y might help in this context
"""

# Standardized residuals vs Leverage to spot high-leverage xi
leverages = [1/results.nobs + (x - x_mean)**2 / Sxx for x in auto_clean['horsepower']]

fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.scatter(leverages, results.resid / np.sqrt(s2), color='black', label=f'N = {int(results.nobs)}')
ax.axhline(0, lw=1, ls=':', color='black', alpha=0.5)
ax.set_xlabel('leverage')
ax.set_ylabel('standardized residual')
ax.legend()
plt.show()

"""
The plot shows a lot of points with high leverage and high standardized residuals
This finding suggests this points might impact the result of the linear regression model
Fitting a new model excluding these points might help in this context, or changing model shape
"""