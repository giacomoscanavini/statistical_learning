"""
Exercises for 
An Introdution to Statistical Learning 
Chapter 2: Statistical Learning
Exercise: 9

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

path_to_data = "/mnt/c/Users/scana/Desktop/datasets_AISL"

auto = pd.read_csv(os.path.join(path_to_data, 'Auto.csv'))

# Check for missing values
print('Missing values per column')
for col in auto.columns:
    print(col, auto[col].isna().sum())

# (a) Which of the predictors are quantitative, and which are qualitative?
"""
Response: mpg

Quantitative predictors: 
    dispalcement
    horsepower
    weight
    acceleration
    year

Qualitative predictors : 
    cylinders codes as [3, 4, 5, 6, 8]
    origin coded as [1, 2, 3]
    name is str type
"""
# (b) What is the range of each quantitative predictor? You can answer this using the min() and max() methods in numpy
# (c) What is the mean and standard deviation of each quantitative predictor?
auto['horsepower']=pd.to_numeric(auto['horsepower'], errors='coerce')
auto_clean=auto.dropna()
# The DataFrame contains some weird values for 'horsepower', convert the col to numerical or NaN and drop those rows

cols = ['displacement', 'horsepower', 'weight', 'acceleration', 'year']
print(f"{'NAME':<14}{'MIN':>10}{'MAX':>10}{'MEAN':>10}{'STD':>10}")

for col in cols:
    mn = auto_clean[col].min()
    mx = auto_clean[col].max()
    mean = auto_clean[col].mean()
    std = auto_clean[col].std()

    print(f"{col:<14}{mn:>10.1f}{mx:>10.1f}{mean:>10.1f}{std:>10.1f}")

"""
NAME                 MIN       MAX      MEAN       STD
displacement        68.0     455.0     194.4     104.6
horsepower          46.0     230.0     104.5      38.5
weight            1613.0    5140.0    2977.6     849.4
acceleration         8.0      24.8      15.5       2.8
year                70.0      82.0      76.0       3.7
"""

# (d) Now remove the 10th through 85th observations. What is the range, mean, and standard deviation of each predictor in the subset of the data that remains?
auto_filt = auto_clean.drop(auto_clean.index[9:85])

cols = ['displacement', 'horsepower', 'weight', 'acceleration', 'year']
print(f"{'NAME':<14}{'MIN':>10}{'MAX':>10}{'MEAN':>10}{'STD':>10}")

for col in cols:
    mn = auto_filt[col].min()
    mx = auto_filt[col].max()
    mean = auto_filt[col].mean()
    std = auto_filt[col].std()

    print(f"{col:<14}{mn:>10.1f}{mx:>10.1f}{mean:>10.1f}{std:>10.1f}")

"""
NAME                 MIN       MAX      MEAN       STD
displacement        68.0     455.0     187.2      99.7
horsepower          46.0     230.0     100.7      35.7
weight            1649.0    4997.0    2936.0     811.3
acceleration         8.5      24.8      15.7       2.7
year                70.0      82.0      77.1       3.1
"""

# (e) Using the full data set, investigate the predictors graphically, using scatterplots or other tools of your choice
# Create some plots highlighting the relationships among the predictors. Comment on your findings
plt.figure()
pd.plotting.scatter_matrix(auto_clean[['displacement', 'horsepower', 'weight', 'acceleration', 'year']])
plt.show()

plt.figure()
auto_clean.boxplot(['horsepower'], by='origin', grid=False)
plt.show()

plt.figure()
auto_clean.boxplot(['acceleration'], by='cylinders', grid=False)
plt.show()

# (f) Suppose that we wish to predict gas mileage (mpg) on the basis of the other variables. Do your plots suggest that any of the other variables might be useful in predicting mpg? Justify your answer
plt.figure()
pd.plotting.scatter_matrix(auto_clean[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']])
plt.show()

plt.figure()
auto_clean.boxplot(['mpg'], by='origin', grid=False)
plt.show()

plt.figure()
auto_clean.boxplot(['mpg'], by='cylinders', grid=False)
plt.show()
