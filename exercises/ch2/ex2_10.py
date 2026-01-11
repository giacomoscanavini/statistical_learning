"""
Exercises for 
An Introdution to Statistical Learning 
Chapter 2: Statistical Learning
Exercise: 10

Boston Housing Data Set - description: 
    A data set containing housing values in 506 suburbs of Boston
    CRIM    : Per capita crime rate by town
    ZN      : Proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS   : Proportion of non-retail business acres per town
    CHAS    : Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX     : Nitric oxides concentration (parts per 10 million)
    RM      : Average number of rooms per dwelling
    AGE     : Proportion of owner-occupied units built prior to 1940
    DIS     : Weighted distances to five Boston employment centres
    RAD     : Index of accessibility to radial highways
    TAX     : Full-value property-tax rate per $10,000
    PTRATIO : Pupil-teacher ratio by town
    LSTAT   : % lower status of the population
    MEDV    : Median value of owner-occupied homes in $1000's
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_data = "/mnt/c/Users/scana/Desktop/datasets_AISL"

boston = pd.read_csv(os.path.join(path_to_data, 'Boston.csv'))
boston = boston.drop(columns='Unnamed: 0')

# (b) How many rows are in this data set? How many columns? What do the rows and columns represent?
nrows, ncols = boston.shape
print(f"rows: {nrows}, cols: {ncols}")
# Rows represent data sample size, Boston area suburbs investigated
# Cols represent number of features / predictors

# (c) Make some pairwise scatterplots of the predictors (columns) in this data set. Describe your findings
plt.figure()
pd.plotting.scatter_matrix(boston[['crim', 'indus', 'age', 'dis']])
plt.show()
# e.g. some linear relationship is visible between age and dis, between age and indus

# (d) Are any of the predictors associated with per capita crime rate? If so, explain the relationship
print(boston.corr()['crim'].sort_values(key=lambda s: s.abs(), ascending=False))
# Evaluate correlation matrix between preditors, select `crim` column and sort it by absolute value
# e.g. `rad` and `tax` have the highest correlation with `crim`

# (e) Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios? 
# Comment on the range of each predictor
for col in ['crim', 'tax', 'ptratio']:
    print(f"{col}: from {boston[col].min()} to {boston[col].max()}")
    print(f"  Max is found at {boston[col].idxmax()}\n")

# (f) How many of the suburbs in this data set bound the Charles river?
print(boston['chas'].value_counts())
print(f"Suburbs bound to Charles river: {boston['chas'].sum()}")

# (g) What is the median pupil-teacher ratio among the towns in this data set?
print(f"Median ptratio: {boston['ptratio'].median()}")

# (h) Which suburb of Boston has lowest median value of owner occupied homes? 
# What are the values of the other predictors for that suburb, and how do those values compare to the overall ranges for those predictors? Comment on your findings
print(f"Suburb with lowest median value: {boston['medv'].idxmin()} with value {boston['medv'].min()}")
for col in boston.columns:
    print(f"{col:<8} {boston.iloc[boston['medv'].idxmin()][col]:>8.2f} with range: {boston[col].min():>8.2f} {boston[col].max():>8.2f}")

# (i) In this data set, how many of the suburbs average more than seven rooms per dwelling? More than eight rooms per dwelling? 
# Comment on the suburbs that average more than eight rooms per dwelling
print(f"Suburbs with >7 rooms per dwelling: {boston[boston["rm"] > 7].shape[0]}")
print(f"Suburbs with >8 rooms per dwelling: {boston[boston["rm"] > 8].shape[0]}")

print(boston[boston['rm'] > 8])
