####################################### MUTIPLE LINEAR REGRESSION #######################################

from sklearn.datasets import fetch_california_housing

import pandas as pd

california = fetch_california_housing()

# print(california.DESCR)

# print(california.feature_names)

# create a dataframe from california data

california_df = pd.DataFrame(california.data, columns=california.feature_names)

california_df["MedHouseValue"] = pd.Series(california.target)

print(california_df.head())

print(california_df.describe())

# Creating a sample because our data is so large
sample_df = california_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8,4.5))
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend=False
    )

# plt.show()

####################################### TRAINING AND TESTING #######################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, random_state=11)

# y = m1x1 + m2x2 + m3x3 ......mnxm + b

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

for i, name in enumerate(california.feature_names):
    print(f"{name}: {lr.coef_[i]}")    

predicted = lr.predict(X=X_test)
print(predicted[:5])


expected = y_test
print(expected[:5])


####################################### VISUALIZE ON A GRAPH #######################################

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)

df["Predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9,9))

axes = sns.scatterplot(
    data=df,
    x="Expected",
    y="Predicted",
    hue="Predicted",
    palette="cool",
    legend=False
)

start = min(expected.min(), predicted.min())

end = max(expected.max(), predicted.max())

print(start)
print(end)

axes.set_xlim(start,end)
axes.set_ylim(start,end)

line = plt2.plot([start,end], [start,end], "k--")

plt2.show()















