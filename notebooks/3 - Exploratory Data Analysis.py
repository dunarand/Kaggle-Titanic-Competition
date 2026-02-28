# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: kaggle-titanic
#     language: python
#     name: kaggle-titanic
# ---

# %% [markdown]
# # 3 - Exploratory Data Analysis


# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** February, 2026


# %% [markdown]
# ## Introduction


# %% [markdown]
# Previously, we've imported, cleaned, and validated the data, then imputed the
# missing values. Here's a quick reminder: We've
#
# - Converted column names to lowercase
# - Validated data types of columns
# - Validated the data except for columns such as `fare` and `name`, where it is
# not possible to do so
# - Extracted a new column `title` from the passenger names, denoting the
# honorifics such as Mr. and Mrs.
# - Extracted a new column `deck` from the cabin numbers, such as deck "A" from
# cabin "A67"
# - Extracted a new column `fare_per_passenger` using `ticket`, `sibsp`,
# `parch` columns
# - Using the `title` column, we've validated the `sex` column
# - Using the `title` column and basic statistical tools, we've imputed missing
# `age` values
# - Imputed 2 missing `embarked` entries using mode
# - Imputed 687 missing `cabin` values (77.1% of the whole data) using a random
# forest model, trained on `fare_per_passenger` and `pclass` columns
#
# In this section, our goal is to answer some questions, discover insights, and
# prepare ourselves for the model-building phase. Firstly, let's recall our goal.
#
# **Goal**: We want to understand what factors, if there's any, contributes to
# a passenger's survival.
#
# Questions we'll be answering throughout this notebook can be summarized as:
#
# - What variables are correlated with `survived`?
# - Are there hierarchies within features affecting survivorship? For example,
# children versus middle-aged survivorship.
# - Are there correlated features?
#
# As usual, we'll start with the imports.


# %% [markdown]
# ## Imports


# %% [markdown]
# We'll import the raw data as well. The purpose is to obtain the passenger IDs
# for certain cases. For example, we want to get the IDs of passengers with
# missing age values so that our deductions are minimally affected by
# possible noise we introduced with data imputation. We will explore the
# modified data as well, but it's a good idea to keep the original and even
# compare the two.


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# %%
na_age_idx = pd.read_csv("../data/raw/na_ids.csv", index_col=0)

na_age_idx.head(10)


# %%
df = pd.read_csv("../data/modified/cleaned.csv")

df.head(10)


# %% [markdown]
# Let's remind ourselves of the data structure.


# %%
df.info()


# %% [markdown]
# ## Univariate Data Analysis


# %% [markdown]
# We'll first setup the Seaborn configuration.


# %%
sns.set_theme(
    style="whitegrid",
    palette="muted",
    rc={
        "axes.spines.left": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)


# %% [markdown]
# We'll also define some helper functions.


# %%
def describe_numeric(data: pd.DataFrame, column: str | list[str]):
    """Custom defined descriptive statistics generator for numeric data."""
    return data[column].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        q25=lambda x: x.quantile(0.25),
        q50=lambda x: x.quantile(0.50),
        q75=lambda x: x.quantile(0.75),
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
        max="max",
        skew="skew",
    )

def describe_categorical(data: pd.DataFrame, column: str | list[str]):
    """Custom define descriptive statistics generator for categorical data."""
    return pd.concat(
        {
            "Count": data[column].value_counts(),
            "Percentage": data[column].value_counts(normalize=True)*100
        },
        axis = 1
    )

def first_nonzero_percentile(data: pd.DataFrame, column: str):
    """Returns the first nonzero percentile for the specified column"""
    return (data[column] == 0).mean()


# %% [markdown]
# ### Survived


# %% [markdown]
# `survived` is the binary target variable.


# %%
describe_categorical(df, "survived")


# %%
ax = sns.countplot(
    data=df, x="survived", hue="survived", alpha=0.9, legend=False
)

ax.set_xlabel("Survived")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship")

plt.savefig("../assets/survived_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - 342 passengers survived
# - 549 passengers did not survive
#
# Titanic's sinking resulted in approximately 1500 deaths among roughly 2200
# passengers, which makes the survival rate about 31.8%. In our training
# dataset, this rate is about 38.4%.


# %% [markdown]
# ### Numeric Features


# %% [markdown]
# Numeric features in the dataset are as follows:
#
# - `sibsp`: The number of siblings/spouses aboard the Titanic
# - `parch`: The number of parents/children aboard the Titanic
# - `fare`: Passenger fare
# - `age`: Age in years
#
# We also feature extracted the `fare_per_person` column.


# %% [markdown]
# #### Sibsp


# %%
describe_numeric(df, "sibsp")


# %%
first_nonzero_percentile(df, "sibsp")


# %%
ax = sns.histplot(data=df, x="sibsp", bins=list(range(0, 9)), binwidth=0.75)

ax.set_xlabel("Siblings/Spouses")
ax.set_title("Titanic Sibling/Spouse Distribution")

plt.savefig("../assets/sibsp_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - 8 siblings is the maximum in our training data
# - Mean sibling/spouse count is 0.52 with median 0
# - A skew value of 3.7 indicates a strong positive skew
# - 68.2% of the passengers have no siblings/spouses aboard
# - In summary, `sibsp` is dominated by passengers with no siblings or spouses,
# resulting in a highly right-skewed distribution


# %% [markdown]
# #### Parch


# %%
describe_numeric(df, "parch")


# %%
first_nonzero_percentile(df, "parch")


# %%
ax = sns.histplot(data=df, x="parch", bins=list(range(0, 7)), binwidth=0.75)

ax.set_xlabel("Parents/Children")
ax.set_title("Titanic Parent/Children Distribution")

plt.savefig("../assets/parch_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - 6 children is the maximum in our training data
# - Mean parents/children count is 0.38 with median 0
# - A skew value of 2.7 indicates a strong positive skew
# - 76.1% of the passengers have no parents/children aboard
# - In summary, `parch` is dominated by passengers with no parents/children,
# resulting in a highly right-skewed distribution
#
# **Takeaways**
#
# We'll investigate these two features more in-depth in the multivariate data
# analysis section. However, we can draw some conclusions by combining our
# results.
#
# - Since both `sibsp` and `parch` are highly right-skewed, and most passengers
# had no siblings/spouses or parents/children, 68% and 76% respectively, we can
# safely conclude that most passengers traveled without close family members.
# - The mean of `sibsp` being greater than `parch`, 0.52 versus 0.38,
# suggesting that there were siblings traveling without their parents or
# couples without any children. The latter is more plausible in this context.
#
# Note that we do not declare a passenger with `sibsp == 0 & parch == 0` as
# someone who traveled alone. It simply means they did not travel with close
# family members. In fact, it was quite common among first-class passengers to
# travel with maids or assistants.


# %% [markdown]
# #### Age


# %% [markdown]
# Before we start our analysis, let's define the age groups we'll use. We will
# stick to these definitions throughout the project.
#
# Age Interval | Class
# -------------|------
# <1 Years     | Infant
# 1-12 Years   | Child
# 13-18 Years  | Youth
# 19-24 Years  | Young Adult
# 25-44 Years  | Adult
# 45-64 Years  | Middle Aged
# 65+          | Aged
#
# **NOTE:** We did not have any cultural, socioeconomic, or other
# considerations to make this classification. It's purely for interpretation
# purposes. If such a grouping is needed for statistical inference or model
# building, we'll make the distinctions in a more concrete way. For our
# purposes, these should serve as a decent guideline. Additionally, the fact
# that we're set to develop a tree based model, making such virtual
# distinctions could affect model performance in a negative way.
#
# Recall that we've imputed missing values in this column. We'll first analyze
# the raw age data.


# %%
raw_age = df.loc[~df.passengerid.isin(na_age_idx)]

describe_numeric(raw_age, "age")


# %% [markdown]
# **Remarks**
#
#  - Youngest passenger is 0.42 years old while the oldest passenger is 80.
#      - Recall from the data dictionary that if a passenger's `age < 1`,
#      then the age is expressed as a fraction. In this case, the said
#      passenger is roughly 22 weeks old.
#  - Median age of 28 is very close to the mean age of 29.7.
#  - A standard deviation of 14.5 indicates that the age feature is quite
#  spread.
#  - About half the passengers are young adults and adults.
#      - We'll analyze the age groups separately.


# %%
fig, axes = plt.subplots(
    nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios": [0.4, 0.6]}
)
fig.set_figheight(5)
fig.set_figwidth(10)

sns.boxplot(data=raw_age, y="age", ax=axes[0])
sns.histplot(data=raw_age, y="age", binwidth=5, kde=True, ax=axes[1])

axes[0].set(ylabel=None)
fig.supylabel("Age")
fig.suptitle("Titanic Age Distribution")

plt.tight_layout()
plt.savefig("../assets/age_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Data is approximately symmetric about the median with a slight positive
# skew.
# - Data is unimodal.
# - There are several high-age outliers above age ~65.
#     - These statistical outliers are most likely genuine data rather than
#     error based on historical facts.
# - There are no outliers on the lower-end, left tail is less extreme
# relative to the IQR.
# - IQR is roughly between 20 and 40 years, which falls into the young
# adult & adult age groups.
#
# Let's also compare imputed age values with raw age data.


# %%
pd.concat(
    {
        "raw": describe_numeric(raw_age, "age"),
        "modified": describe_numeric(df, "age"),
    },
    axis=1,
)


# %% [markdown]
# Recall that we've imputed the age feature as follows:
#
# - We used `title` feature we've extracted from the `name` column for age
# imputation
# - For missing age entries with titles Dr. and Master., we've used their group
# median  ages
# - For the title Miss., we've used `parch` column to differentiate young girls
# from unmarried adult women and filled the missing values by taking the median
# age
# - For the rest of the missing data, we've assigned medians based on `pclass`
# and `title`
#
# By looking at the table, we can understand the following:
#
# - Entry count increased from 714 to 891
# - Mean age decreased about 2.36% while the median age decreased about 7.14%
# - Standard deviation also decreased
# - Since we've used median ages for imputation, the minimum and maximum ages
# remained the same
# - Skewness increased by 15.7%
#
# One important note here is that none of these changes are extreme. For one
# thing, imputed values did not drastically change any descriptive statistic.
# Given we've imputed about 20% of the whole data, the noise introduced seems
# minimal from what we can tell. Let's also compare the distributions.


# %%
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    sharex="col",
    sharey="row",
    gridspec_kw={"height_ratios": [0.6, 0.4]},
)
fig.set_figheight(10)
fig.set_figwidth(10)

axes[1, 0].set(xlabel="Raw Data")
axes[1, 1].set(xlabel="Modified Data")
fig.supxlabel("Age")
fig.suptitle("Raw vs. Imputed Age Distributions")

sns.histplot(
    data=raw_age, x="age", binwidth=5, kde=True, ax=axes[0, 0], stat="percent"
)
sns.histplot(
    data=df, x="age", binwidth=5, kde=True, ax=axes[0, 1], stat="percent"
)
sns.boxplot(data=raw_age, x="age", ax=axes[1, 0])
sns.boxplot(data=df, x="age", ax=axes[1, 1])

plt.tight_layout()
plt.savefig("../assets/raw_vs_imputed_age_comparison.png")
plt.show()


# %% [markdown]
# Distributions look similar except around the median. Since we've imputed the
# majority of the missing values with the median, imputed age distribution has
# steeper median value. On the other hand, rest of the distribution seems
# similar and looks as expected from an age distribution in our case. It still
# captures  the characteristic that the age distribution is positively skewed.
# One other change we can observe is the number of statistical outliers, which
# is as expected since the mean and the standard deviation are lower in
# modified data.
#
# Now, let's also plot the age groups we defined earlier.


# %%
age_groups = pd.cut(
    df["age"],
    bins = [0, 1, 12, 18, 24, 44, 64, 100],
    labels = ["Infant", "Child", "Youth", "Young Adult", "Adult",
              "Middle Aged", "Aged"],
    right = False,
    include_lowest = True
)
if isinstance(age_groups, pd.Series):
    counts = age_groups.value_counts()
    percentages = age_groups.value_counts(normalize=True) * 100

    age_groups = pd.DataFrame({
        "age group": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    })

    age_groups = (
        age_groups.sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

else:
    raise ValueError()

print(age_groups)

ax = sns.barplot(
    data=age_groups, x="age group", y="count", alpha=0.9, legend=False
)

plt.xticks(rotation = 45)

ax.set_xlabel("Age Group")
ax.set_ylabel("Count")
ax.set_title("Titanic Age Group Distribution")

plt.tight_layout()
plt.savefig("../assets/age_group_dist.png")
plt.show()


# %% [markdown]
# #### Fare


# %%
print(df.fare.mean())
df.sort_values(by="fare", ascending=False)["fare"].head(21)


# %% [markdown]
# `fare` column's mean is 32.2 while the maximum value is 512.3292. Therefore,
# plotting without filtering make the visual not easily readable. Hence, we'll
# filter the maximum values for the graph.


# %%
fare_desc = describe_numeric(df, "fare")
print(fare_desc)


# %% [markdown]
# **Remarks**
#
# - The minimum fare is GBP 0 while the highest fare is GBP 512.
#    - Highest fare is historically accurate in this case.
# - Median fare is GBP 14.5 while the mean is 32.2.
# - The standard deviation is GBP 49.7, which is quite high, especially
# compareed to the mean. Thus, the data is quite spread out.
# - About half the passengers paid GBP 14.5 or less, 25% of which paid about
# GBP 7.9 or less.
# - About 25% of the passengers paid more than GBP 31.

# %%
fare_filtered = df.loc[df.fare <= 75]
fig, axes = plt.subplots(
    nrows=1, ncols=2, sharey=True, gridspec_kw={"width_ratios": [0.4, 0.6]}
)
fig.set_figheight(5)
fig.set_figwidth(10)

sns.boxplot(data=fare_filtered, y="fare", ax=axes[0])
sns.histplot(data=fare_filtered, y="fare", binwidth=5, kde=True, ax=axes[1])

axes[0].set(ylabel=None)
fig.supylabel("Fare")
fig.suptitle("Titanic Fare Distribution")

plt.savefig("../assets/fare_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Data is extremely skewed. A skew value of 4.79 indicates an extreme
# positive skew.
# - There are a lot of statistical outliers above ~GBP 60.
#    - In fact, there are 20 outliers detected by the z-test, or 116
#    outliers detected using IQR.
#    - However, note that these statistical outliers are highly likely to be
#    legitimate values. In reality, the Titanic had luxurious suites with
#    very high markups. For example, the highest value of GBP 512 is
#    historically accurate.
#  - There are no outliers on the lower-end, left tail is less extreme
#  relative to the IQR.
#  - IQR is about GBP 23.1
#
# Below is the IQR and z-scores tests for outliers.


# %%
iqr = fare_desc["iqr"]
q3 = fare_desc["q75"]
upper_lim = q3 + 1.5 * iqr

print(f"Outlier count using IQR: {df[df.fare > upper_lim].shape[0]}")

z_scores = np.asarray(stats.zscore(df["fare"]))

print(f"Outlier count using z-scores: {np.count_nonzero(z_scores > 3)}")


# %% [markdown]
# Let's also plot log-scaled `fare`.


# %%
ax = sns.histplot(
    data=df, x="fare", alpha=0.9, legend=False, log_scale=True
)

ax.set_xlabel("Fare")
ax.set_ylabel("Count")
ax.set_title("Titanic Fare Distribution (Log Scaled)")

plt.savefig("../assets/fare_logscaled_dist.png")
plt.show()


# %% [markdown]
# ### Categorical Features


# %% [markdown]
# Categorical features in the dataset are as follows:
#
# - `pclass`
# - `name`
# - `sex`
# - `ticket`
# - `cabin`
# - `embarked`: Port of embarkation
#
# We've also feature extracted
#
# - `deck`
# - `title`
#
# We will not analyze the `name`, `ticket`, and `cabin` columns.


# %% [markdown]
# #### PClass


# %%
describe_categorical(df, "pclass")


# %%
ax = sns.countplot(
    data=df, x="pclass", alpha=0.9, legend=False
)

ax.set_xlabel("Passenger Class")
ax.set_ylabel("Count")
ax.set_title("Titanic Passenger Class Distribution")

plt.savefig("../assets/pclass_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - More than half of the passengers were third-class
# - There are more first-class passengers than second-class


# %% [markdown]
# #### Sex


# %%
describe_categorical(df, "sex")

# %%
ax = sns.countplot(
    data=df, x="sex", hue="sex", alpha=0.9, legend=False
)

ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_title("Titanic Sex Distribution")

plt.savefig("../assets/sex_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Out of 891 passengers, 577 of the are male, 64.8% of the passengers, while
# 314 are female


# %% [markdown]
# #### Embarked


# %%
describe_categorical(df, "embarked")


# %%
ax = sns.countplot(
    data=df, x="embarked", alpha=0.9, legend=False
)

plt.xticks([0, 1, 2], ["Southampton", "Cherbourg", "Queenstown"])

ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_title("Titanic Port of Embarkation Distribution")

plt.savefig("../assets/embarked_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - 72.5% of the passengers embarked from Southampton, England
# - 18.9% of the passengers embarked from Cherbourg, France
# - 8.64% of the passengers embarked from Queenstown, Ireland


# %% [markdown]
# #### Deck


# %% [markdown]
# Recall that we've extracted the deck feature from the `cabin` column, then
# imputed the missing values. About 77% of the `deck` data was missing prior to
# imputation.


# %%
describe_categorical(df, "deck")


# %%
ax = sns.countplot(
    data=df,
    x="deck",
    alpha=0.9,
    legend=False,
    order = ["T", "A", "B", "C", "D", "E", "F", "G"]
)

ax.set_xlabel("Deck")
ax.set_ylabel("Count")
ax.set_title("Titanic Deck Distribution")

plt.savefig("../assets/deck_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - The majority of the passengers occupied E & F decks
# - Deck T had only 1 passenger
#   - Recall that we've validated the data already, there's no mistake here
# - Data is negatively skewed
# - Mode of deck is deck F


# %% [markdown]
# #### Title


# %% [markdown]
# Recall that we've extracted the title feature from the `name` column.


# %%
describe_categorical(df, "title")


# %%
ax = sns.countplot(
    data=df,
    x="title",
    alpha=0.9,
    legend=False,
    order = df["title"].value_counts().sort_values(ascending=False).index
)

plt.xticks(rotation=45)

ax.set_xlabel("Honorific Title")
ax.set_ylabel("Count")
ax.set_title("Titanic Honorific Title Distribution")

plt.show()


# %% [markdown]
# This plot isn't exactly easy to read. Let's group titles with 1 or 2
# occurrences into a combined group called *"Others"* and plot again.


# %%
titles_df = df["title"].value_counts().reset_index()
titles_df.columns = ["title", "count"]
titles_df.loc[-1] = ["Others", sum(titles_df["count"]<=2)]
titles_df = (
    titles_df.drop(titles_df[titles_df["count"] <= 2].index)
    .sort_values(by="count", ascending=False)
    .reset_index(drop=True)
)

ax = sns.barplot(
    data=titles_df, x="title", y="count", alpha=0.9, legend=False,
)

ax.set_xlabel("Honorific Title")
ax.set_ylabel("Count")
ax.set_title("Titanic Honorific Title Distribution")

plt.savefig("../assets/title_dist.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - More than half the passengers, 58% to be precise, were Mr.
# - Occupation related titles, such as Dr. or Col., are rare
# - Majority of the passengers had English honorifics but there were also some
# honorifics in French, Italian, etc.
