# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1 - Understanding & Planning

# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** December, 2025

# %% [markdown]
# ## The Problem

# %% [markdown]
# Regardless of the workflow a data scientist chooses for a project, it should
# start with a question. Of course, there'll be many questions asked, and
# answered along the way; however, the first question should spark the
# curiosity within and guide us through the project: the ultimate question
# we're trying to answer with this project. Thus, we state our question:
#
# > Did people survive the Titanic incident out of pure luck, or were social
# constructs resulted in certain groups having more chances at survival?
#
# Remember, one of the reasons why so many people died in this incident was
# because of the lack of lifeboats. Therefore, people on board had to make
# certain choices; some sacrifices had to be made, and some people were saved.
# But we wonder whether a person's traits, such as age, wealth, sex, etc.
# played a role in their survival, and if so, which groups were more likely to
# survive.

# %% [markdown]
# ## The Data

# %% [markdown]
# With this project, we've been handed out three datasets:
#
# 1. `train.csv`: Contains the passenger information for training the ML model
# 2. `test.csv`: Used for evaluating the model performance for submission
# 3. `gender_submission.csv`: Example submission data

# %% [markdown]
# ### Data Dictionary <a name="data-dictionary"></a>

# %% [markdown]
# `train.csv` contains
#
# - **891 rows**
# - **11 columns**
#
# Variable | Dtype | Definition | Key
# ---------|-------|------------|-----
# `Survived` | `int64` | Survival | 0 = No, 1 = Yes
# `Pclass` | `int64` | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd
# `Name` | `object` | Name | 
# `Sex` | `object` | Sex |
# `Age` | `float64` | Age in years |
# `SibSp` | `int64` | The number of siblings / spouses aboard the Titanic |
# `Parch` | `int64` | The number of parents / children aboard the Titanic |
# `Ticket` | `object` | Ticket number |
# `Fare` | `float64` | Passenger fare |
# `Cabin` | `object` | Cabin number |
# `Embarked` | `object` | Port of Embarkation | C = Cherbourg, </br>Q = Queenstown, </br>S = Southampton
#
# **Notes:**
#
# - `Pclass`: A proxy for socio-economic status (SES), 1st = Upper,
# 2nd = Middle, 3rd = Lower
# - `Age`: Age is fractional if less than 1. If the age is estimated, is it in
# the form of xx.5
# - `SibSp`: The dataset defines family relations in this way...
#     - Sibling = brother, sister, stepbrother, stepsister
#     - Spouse = husband, wife (mistresses and fiancés were ignored)
# - `Parch`: The dataset defines family relations in this way...
#     - Parent = mother, father
#     - Child = daughter, son, stepdaughter, stepson
#     - Some children travelled only with a nanny, therefore parch=0 for them.

# %% [markdown]
# ## Inspecting the Data

# %% [markdown]
# ### Importing the Data

# %% [markdown]
# Let us start by importing the required packages.

# %%
import pandas as pd

# %% [markdown]
# Let's now import the data.

# %%
df = pd.read_csv("../data/raw/train.csv", encoding="utf-8")
df.head(10)

# %% [markdown]
# ### Initial Data Wrangling

# %% [markdown]
# We have the column `PassengerId`. This column is used in the `test.csv` data
# for submission purposes. We will not need this column for model training.
#
# Now, let's check for missing values and data types.

# %%
df.info()

# %% [markdown]
# Only columns to contain missing values are `Age`, `Cabin`, and `Embarked`. In
# the data wrangling section of the project, we'll deal with these missing
# values. For now, having an educated view on the data is beneficiary for our
# planning purposes.

# %% [markdown]
# Let's also check for duplicates.

# %%
df.duplicated(keep="first").sum()

# %% [markdown]
# We don't have any duplicates we need to deal with.

# %% [markdown]
# **Takeaways**
#
# - We'll drop the column `PassengerId` as it's not needed for training
# - We've found no duplicates in the data
# - The data contains some missing values, especially in the `Cabin` column.

# %% [markdown]
# ### Initial EDA

# %% [markdown]
# To make reasonable plans, we'll briefly inspect the data.

# %%
df.describe()

# %% [markdown]
# We can deduce that
#
# - Majority did not survive (`Survived` is a binary variable with mean 0.38)
# - Upper class (`Pclass`) was the minority
# - Majority were younger than midle-aged (75th percentile is 38)
# - More than half the people had no siblings/spouses
# - Majority had no children
#
# Let's also take a look at the `Sex` variable.
#
# <center><div><img src="../assets/survival_by_sex.png" width="500"/></div></center>

# %%
df['Sex'].value_counts()

# %%
df.groupby('Sex')['Survived'].agg('sum')

# %% [markdown]
# We see that although the majority of the passengers were males, females were
# the majority among survivors. We'll also take other variables like `Age` into
# consideration during the EDA, but for now, this information, together with
# our intuition is enough to suspect that `Sex` is probably correlated with
# the target variable.

# %% [markdown]
# ### Performance Targets
#
# Lastly, we need to determine a performance benchmark. Let's create a baseline
# prediction: since the majority of females survived (233 survivors out of
# 314 total), a model that predicts the passenger to survive if the passenger
# is female will be accurate most of the time. Let's check.
#
# Let's say the model predicts survival if the passenger is female and did not
# survive if the passenger is male. In this case,

# %%
from sklearn.metrics import accuracy_score

def predict(X: pd.DataFrame) -> pd.Series:
    return X['Sex'] == 'female'

print(accuracy_score(df['Survived'], predict(df)))

# %% [markdown]
# A model that predicts all females as survived and all males as did not
# survive has an accuracy of 79%. Let's also see what a baseline decision tree
# classifier is able to achieve.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, precision_score, f1_score)

df = pd.read_csv("../data/raw/train.csv", encoding="utf-8")

df.drop(['Cabin', 'Embarked', 'Name', 'PassengerId', 'Ticket'],
        axis=1, inplace=True)

# Filling missing values; only `Age` column contains NaNs after dropping the
# columns above
df.fillna(value=df['Age'].mean(), axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)

df = pd.get_dummies(data=df)

y = df['Survived']
X = df.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 40
)

dt = DecisionTreeClassifier(random_state = 40)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, pred)}")
print(f"Recall score: {recall_score(y_test, pred)}")
print(f"Precision score: {precision_score(y_test, pred)}")
print(f"F1 score: {f1_score(y_test, pred)}")

# %% [markdown]
# A baseline model without any feature engineering, by dropping the columns
# `Cabin`, `Embarked`, `Name`, and `Ticket`, and by filling in missing `Age`
# values with the mean can achieve an accuracy score of 79.9%, and an F1-score
# of 75.7%. Thus, we'll set our target as around 85% accuracy and F1-score for
# this project, performing better than an all-female survivor model and a
# baseline decision tree model.
#
# Let's also `pickle` this model for future reference.

# %%
import pickle

with open("../models/base_dt.pkl", "wb") as f:
    pickle.dump(dt, f)

# test
with open("../models/base_dt.pkl", "rb") as f:
    base_dt = pickle.load(f)

print(accuracy_score(y_test, base_dt.predict(X_test)))

# %% [markdown]
# ## Next Steps

# %% [markdown]
# Recall that the problem we're trying to solve is to be able to infer whether
# a passenger survived on the basis of their features present in the dataset.
# These features are stated in the [data dictionary](#data-dictionary).
# The target variable is the binary variable `Survived`.
#
# We can make some initial, educated plans. First of all, for the data
# wrangling part, we'll need to deal with the missing values. We've discovered
# that `Age`, `Cabin`, and `Embarked` variables contain missing values. Based
# on the data dictionary and our purpose, we may choose to
#
# - Impute missing `Age` values because, as per intuition, this column is
# probably correlated with the target variable `Survived`
# - Discuss the relevance of `Cabin` column and potentially drop it
# - Discuss the relevance of `Embarked` column and potentially drop it
#
# Other features that are probably correlated with the target variable are
#
# - `Pclass`
# - `Sex`
# - `Fare`
#
# Of course, we'll conduct a detailed analysis for each feature on whether
# they're correlated or not with the target variable. We'll also conduct
# analysis to reveal relationships between feature variables. We'll discuss
# this topic more in the EDA phase.
#
# Additionally, we may choose to drop columns such as `Name` or `Sibsp`
# after EDA, if we fail to find any relation to the target variable.
# Intuitively, we'll at least drop the `Name` column after feature extraction.
# The `Name` itself shouldn't be significant except for determining sex,
# family members, or title (Mr., Mrs., etc.).
#
# Lastly, the model choice will be clearer after data wrangling and EDA;
# however, an educated guess would be that a tree-based classifier is probably
# the best fit for the case. We'll uncover more in the upcoming sections. We've
# set a performance target of around 85% in both accuracy and F1 scores for
# the model.
