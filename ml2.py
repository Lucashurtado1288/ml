
################################################ SIMPLE LINEAR REGRESSION ####################################################

import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')


print(nyc.head(3))                          # head(3) displays the first 3 rows of the DataFrame.

print(nyc.Date.values)                      # nyc.Date selects the Date column. .values converts that column into a NumPy array.

print(nyc.Date.values.reshape(-1,1))        # These are the input values (X) (features) converts from one-dimensional to two dimensional arrary

############################################ TRAINING AND TESTING DATE ############################################

# nyc.Temperature.values (targets). These are the target values (y) the model is trying to predict.


X_train, X_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11)

print(X_train.shape)

print(X_test.shape)


# Creating the linear model, imports Skikit-Learns linear regression algorithm
from sklearn.linear_model import LinearRegression

# Linear Regression object 
lr = LinearRegression()          

# This is the learning step, The model examines and finds the line that best fits all data points
lr.fit(X=X_train, y=y_train)        # Where all of the learning takes place/occurs, calculates the slope and intercept

    # similar to: y = mx + b

# The coefficient is the slope (m).
print(lr.coef_)

# The intercept is b in:
print(lr.intercept_)

predicted = lr.predict(X=X_test)

expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

############################################ EXAMPLE USING REGRESSION FORMULA ############################################

# y = mx + b

predict = lambda x: lr.coef_ * x + lr.intercept_

print(predict(2019))

print(predict(1890))


############################################ VISUALIZE ON GRAPH USING SEABORN ############################################

# Create a scatterplot with regression line

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False
)


axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values),max(nyc.Date.values)])
print(x)

y = predict(x)
print(y)


import matplotlib.pyplot as plt

line = plt.plot(x,y)

plt.show()




