"""
Exercises for 
An Introdution to Statistical Learning 
Chapter 2: Statistical Learning
Exercise: 8

College Data Set - description:
    It contains a number of variables for 777 different universities and colleges in the US
    Private     : Public/private indicator
    Apps        : Number of applications received
    Accept      : Number of applicants accepted
    Enroll      : Number of new students enrolled
    Top10perc   : New students from top 10 % of high school class
    Top25perc   : New students from top 25 % of high school class
    F.Undergrad : Number of full-time undergraduates
    P.Undergrad : Number of part-time undergraduates
    Outstate    : Out-of-state tuition
    Room.Board  : Room and board costs
    Books       : Estimated book costs
    Personal    : Estimated personal spending
    PhD         : Percent of faculty with Ph.D.s
    Terminal    : Percent of faculty with terminal degree
    S.F.Ratio   : Student/faculty ratio
    perc.alumni : Percent of alumni who donate
    Expend      : Instructional expenditure per student
    Grad.Rate   : Graduation rate
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

path_to_data = "/mnt/c/Users/scana/Desktop/datasets_AISL"

# (a) Use the pd.read_csv() function to read the data into Python
college = pd.read_csv(os.path.join(path_to_data, "College.csv"), index_col=0)

# (c) Use the describe() method of to produce a numerical summary
print(college.describe())

# (d) Use the pd.plotting.scatter_matrix() function to produce a scatterplot matrix of the first columns [Top10perc, Apps, Enroll]
plt.figure()
pd.plotting.scatter_matrix(college[['Top10perc', 'Apps','Enroll']])
plt.show()

# (e) Use the boxplot() method of college to produce side-by-side boxplots of Outstate versus Private
plt.figure()
college.boxplot(column=['Outstate'], by='Private', grid=False)
plt.show()

# (f) Create a new qualitative variable, called Elite, by binning the Top10perc variable into two groups based on percentage of elite students
college['Elite'] = pd.cut(college['Top10perc'], [0, 50, 100], labels=['No', 'Yes'])
# Use the value_counts() method of college['Elite'] to see how many elite universities there are
print(college['Elite'].value_counts())
# Use the boxplot() method again to produce side-by-side boxplots of Outstate versus Elite
plt.figure()
college.boxplot(column=['Outstate'], by='Elite', grid=False)
plt.show()

# (g) Use the plot.hist() method of college to produce some histograms with differing bins for a few of the quantitative variables
plt.figure()
college.plot.hist(by='Private', bins=10)
plt.show()















