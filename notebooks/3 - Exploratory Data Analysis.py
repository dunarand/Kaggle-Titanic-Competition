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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3 - Exploratory Data Analysis


# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** March, 2026


# %% [markdown]
# ## Introduction


# %% [markdown]
# Previously, we've imported, cleaned, and validated the data, then imputed the
# missing values. Here's a quick reminder: We've
#
# - Converted column names to lowercase
# - Validated data types of columns
# - Validated the data except for columns such as `fare` and `name`, where it
# is not possible to do so
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
# In this section, our goal is to answer questions, uncover insights, and
# prepare for the model-building phase. Firstly, let's recall our goal.
#
# **Goal**: We want to understand what factors, if any, contribute to a
# passenger's survival.
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
# missing age values so that our deductions are minimally affected by possible
# noise we introduced with data imputation. We will explore the modified data
# as well, but it's a good idea to keep the original and even compare the two.


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import Table

# %%
na_age_idx = pd.read_csv("../data/raw/na_ids.csv", index_col=0)
na_age_idx = na_age_idx["passengerid"]

na_age_idx.head(10)


# %%
df = pd.read_csv("../data/modified/cleaned.csv")

df.head(10)


# %% [markdown]
# Let's remind ourselves of the data structure.


# %%
df.info()


# %% [markdown]
# ## Univariate Analysis


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

ax.set_xticks([0, 1], ["No", "Yes"])

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
# The sinking of the Titanic resulted in approximately 1500 deaths among
# roughly 2200 passengers, which makes the survival rate about 31.8%. In our
# training dataset, this rate is about 38.4%.


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
# - Mean sibling/spouse count is 0.52 with a median of 0
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
# - Mean parents/children count is 0.38 with a median of 0
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
# that we're set to develop a tree-based model means that making such virtual
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
#  - The youngest passenger is 0.42 years old, while the oldest passenger is
#  80.
#      - Recall from the data dictionary that if a passenger's `age < 1`, then
#      the age is expressed as a fraction. In this case, the said passenger is
#      roughly 22 weeks old.
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
#     errors based on historical facts.
# - There are no outliers on the lower-end, the left tail is less extreme
# relative to the IQR.
# - IQR is roughly between 20 and 40 years, which falls into the young adult &
# adult age groups.
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
# - We used the `title` feature we've extracted from the `name` column for age
# imputation
# - For missing age entries with titles Dr. and Master., we've used their group
# median  ages
# - For the title Miss., we've used the `parch` column to differentiate young
# girls from unmarried adult women, and filled the missing values by taking the
# median age
# - For the rest of the missing data, we've assigned medians based on `pclass`
# and `title`
#
# By looking at the table, we can understand the following:
#
# - Entry count increased from 714 to 891
# - Mean age decreased about 2.36%, while the median age decreased about 7.14%
# - Standard deviation also decreased
# - Since we've used median ages for imputation, the minimum and maximum ages
# remained the same
# - Skewness increased by 15.7%
#
# One important note here is that none of these changes is extreme. For one
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
# majority of the missing values with the median, the imputed age distribution
# has a steeper median value. On the other hand, the rest of the distribution
# seems similar and looks as expected from an age distribution in our case. It
# still captures  the characteristic that the age distribution is positively
# skewed. One other change we can observe is the number of statistical
# outliers, which is, as expected, since the mean and the standard deviation
# are lower in the modified data.
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
#    - The highest fare is historically accurate in this case.
# - Median fare is GBP 14.5 while the mean is 32.2.
# - The standard deviation is GBP 49.7, which is quite high, especially
# compared to the mean. Thus, the data is quite spread out.
# - About half the passengers paid GBP 14.5 or less, 25% of whom paid about GBP
# 7.9 or less.
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
#    - In fact, there are 20 outliers detected by the z-test, or 116 outliers
#    detected using IQR.
#    - However, note that these statistical outliers are highly likely to be
#    legitimate values. In reality, the Titanic had luxurious suites with very
#    high markups. For example, the highest value of GBP 512 is historically
#    accurate.
#  - There are no outliers on the lower-end, the left tail is less extreme
#  relative to the IQR.
#  - IQR is about GBP 23.1
#
# Below are the IQR and z-score tests for outliers.


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
# We've also extracted features
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

ax.set_xticks([0, 1], ["Male", "Female"])

ax.set_xlabel("Sex")
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

ax.set_xlabel("Port of Embarkation")
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
#   - Recall that we've validated the data already; there's no mistake here
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
# - Occupation-related titles, such as Dr. or Col., are rare
# - The majority of the passengers had English honorifics, but there were also
# some honorifics in French, Italian, etc.


# %% [markdown]
# ## Bivariate Analysis


# %% [markdown]
# We'll investigate data relations between our target variable and features.
# We'll start by defining some helper functions.


# %%
def desc_survived_vs_cont(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Describes a continuous variable versus survived."""
    result = data.groupby(by="survived")[col].agg([
        "count", "mean", "median", "std", "min", "max",
        ("25%", lambda x: x.quantile(0.25)),
        ("75%", lambda x: x.quantile(0.75)),
        ("IQR", lambda x: x.quantile(0.75) - x.quantile(0.25))
    ])
    if isinstance(result, pd.DataFrame):
        return result
    else:
        raise ValueError("Aggregated object is not a pd.DataFrame instance.")

def desc_survived_vs_cat(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Describes a categorical variable versus survived."""
    counts = data.groupby(by=col)["survived"].value_counts().unstack()
    percents = (
        data.groupby(by=col)["survived"]
        .value_counts(normalize=True)
        .unstack() * 100
    ).round(1).astype("str")

    return pd.concat(
        {"Count": counts, "Percentage": percents}, axis=1
    )


# %% [markdown]
# ### Passenger Class versus Survived


# %% [markdown]
# We'll investigate whether `pclass` had any effect on a passenger's
# survivorship or not.


# %%
desc_survived_vs_cat(df, "pclass")


# %%
ax = sns.countplot(data=df, x="pclass", hue="survived", alpha=0.9)

ax.set_xlabel("Passenger Class")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Passenger Class")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_pclass.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - First-class passengers had the highest survival rate, with 63.0% of the
# passengers surviving
# - Survivorship among second-class passengers was roughly 50-50
# - Third-class passengers had the lowest survival rate, with only 24.2% of the
# passengers surviving
#
# Visually, it seems that there's an association between passenger class and
# survivorship. Let's test this claim statistically. We'll use the chi-square
# test of independence.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `pclass`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `pclass`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
obs_val = pd.crosstab(df["pclass"], df["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Since the computed p-value is smaller than our determined $\alpha$ value, we
# reject the null hypothesis. Thus, we have found statistical evidence that
# `survived` is dependent on `pclass`.
#
# We should also interpret residuals to see how each passenger class is
# associated with `survived`.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("Passenger Class")
ax.set_title("pclass vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_pclass_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - We observe significantly more survivors in first-class passengers than
# expected
# - We observe more survivors in second-class passengers than expected
# - We observe significantly fewer survivors in third-class than expected
#
# From the initial plot "Passenger Class versus Survived" and the chi-square
# test of independence, we observe that `survived` is dependent on `pclass`:
#     # first- and second-class passengers had higher survivorship rates than
#     # expected. In comparison, third-class passengers had lower survivorship
#     # rates.


# %% [markdown]
# ### Sex versus Survived


# %% [markdown]
# We'll investigate whether `sex` had any effect on a passenger's survivorship
# or not.


# %%
desc_survived_vs_cat(df, "sex")


# %%
ax = sns.countplot(data=df, x="sex", hue="survived", alpha=0.9)

ax.set_xlabel("Sex")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Sex")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_sex.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - The majority of the males did not survive, while the majority of the
# females did
# - Most of the survivors are female, 74.2% to be exact
#
# Visually, it seems that there's an association between sex and survivorship.
# We'll use the chi-square test of independence to prove this statistically.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `sex`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `sex`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
obs_val = pd.crosstab(df["sex"], df["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Observed p-value is significantly smaller than the determined $\alpha$. Thus,
# we reject the null hypothesis and conclude that `survived` is dependent on
# the `sex` variable.
#
# What we can tell with confidence by looking at the plot and the result of the
# chi-square test is that `sex` is a good predictor for the `survived` target
# variable. In fact, recall that at the start of the project, we used an "all
# female survived" type of model where the model predicted a passenger as
# survived for females and did not survive for males. The model had a 78.7%
# accuracy.
#
# This evidence shows how important the `sex` feature is. What we historically
# know from the Titanic incident, the famous line *"Women and children
# first!"*, seems to be true for the time being. We'll test for age next and
# verify what we know and back it up with statistics.


# %% [markdown]
# ### Age versus Survived


# %% [markdown]
# We'll investigate whether `age` had any effect on a passenger's survivorship
# or not.


# %%
desc_survived_vs_cont(df, "age")


# %%
ax = sns.boxplot(data=df, x="survived", y="age", hue="survived", legend=False)

ax.set_xticks([0, 1], ["No", "Yes"])

ax.set_xlabel("Survived")
ax.set_ylabel("Age")
ax.set_title("Titanic Survivorship by Age")

plt.savefig("../assets/surv_vs_age_box.png")
plt.show()


# %%
ax = sns.violinplot(
    data=df, x="survived", y="age", hue="survived", legend=False
)

ax.set_xticks([0, 1], ["No", "Yes"])

ax.set_xlabel("Survived")
ax.set_ylabel("Age")
ax.set_title("Titanic Survivorship by Age")

plt.savefig("../assets/surv_vs_age_vio.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - From the descriptive statistics table, we see no huge differences:
#   - There's roughly a 1.5 years of age difference in both means and medians
#   between survivors and victims
#   - Percentiles are roughly the same
# - Visually, box plots are very similar. However, violin plots show a
# different picture
#   - Distributions look completely different
#   - The sharper median could be the result of the age imputation. Recall that
#   we've imputed many missing age values with medians, which can explain the
#   sharp edge
#
# Before we proceed with statistical tests, let's confirm our suspicion. We'll
# check whether we've imputed the age data of mostly survived passengers.


# %%
ax = sns.kdeplot(
    data=df.loc[~df.passengerid.isin(na_age_idx)],
    x="age",
    hue="survived",
    alpha=0.5,
    fill=True,
    common_norm=False
)

ax.set_xlabel("Age")
ax.set_ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.show()


# %% [markdown]
# As we've suspected, the sharp edge around the median comes from the age
# imputation. Victims' distributions look roughly the same, which means that
# most of the imputed age values were from survivors.
#
# Recall that we've imputed missing age values using `pclass` and `title`
# columns. Since passenger class is a feature that can explain `survived`, we
# could have contaminated the `age` feature with the `survived` signal. A
# tree-based model shouldn't be sensitive to this issue; however, should we
# decide to implement a different classifier in the future, we may need to
# improve the age imputation methods.
#
# Now, we should go back to our first claim from the `sex` section, Titanic's
# famous *"Women and children first"* strategy. We should conduct a statistical
# test just to be sure. For this, we'll compare the population mean ages
# between two groups, survivors and victims.
#
# We'll use the Mann-Whitney U test since the data is not normally distributed.
# We meet the assumptions:
#
# 1. Observations are independent of each other
# 2. Groups are independent (survivors and victims) from each other
# 3. The variable (age) is continuous
#
# Since the survivor and victim age distributions are **not** similarly shaped,
# our hypothesis will be as follows:
#
# - **$\text{H}_0$**: There is no difference between age distributions of
# survivors and non-survivors:
# $$
#     P(X_\text{survivor} > X_\text{victim}) = 0.5
# $$
# - **$\text{H}_\text{a}$**: There is a difference between the age
# distributions of survivors and non-survivors:
# $$
#     P(X_\text{survivor} > X_\text{victim}) \neq 0.5
# $$
#
# Essentially, we test for the probability of a randomly chosen survivor being
# older (or younger) than a randomly chosen victim.
#
# We'll set the significance level $\alpha = 0.05$.


# %%
survivors = df.loc[df.survived == 1, "age"].values
victims = df.loc[df.survived == 0, "age"].values

stat, pval = stats.mannwhitneyu(survivors, victims)

print(f"Stat: {stat}\np-value: {pval}")


# %% [markdown]
# The computed p-value of 0.23 is greater than our determined $\alpha$ value.
# Thus, we fail to reject the null hypothesis to conclude that there is no
# difference between the age distributions of survivors and non-survivors.
# Essentially, there is no statistical evidence to support the claim that a
# randomly chosen survivor would necessarily be younger (or older) than a
# randomly chosen victim, showing that there's no pattern.
#
# But this result does not necessarily explain our question to the full extent.
# The Mann-Whitney U test explains that the overall age distributions are not
# statistically different. However, it is possible that some specific age
# groups, such as infants, could have higher survival chances. Let's also test
# for this fact. For that, we'll use the chi-square test of independence. We'll
# use the same age groups we've defined earlier in the univariate analysis
# section.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `age_group`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `age_group`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
df["age_group"] = pd.cut(
    df["age"],
    bins = [0, 1, 12, 18, 24, 44, 64, 100],
    labels = ["Infant", "Child", "Youth", "Young Adult", "Adult",
              "Middle Aged", "Aged"],
    right = False,
    include_lowest = True
)

desc_survived_vs_cat(df, "age_group")


# %% [markdown]
# The infant row violates the data assumptions of the chi-square test.
# Therefore, we'll combine infants and children into a single group and proceed
# that way.


# %%
df["age_group"] = pd.cut(
    df["age"],
    bins = [0, 12, 18, 24, 44, 64, 100],
    labels = ["Child", "Youth", "Young Adult", "Adult",
              "Middle Aged", "Aged"],
    right = False,
    include_lowest = True
)

desc_survived_vs_cat(df, "age_group")


# %%
obs_val = pd.crosstab(df["age_group"], df["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# It appears that the computed p-value is significantly less than our
# determined significance level of $\alpha = 0.05$. Therefore, we reject the
# null hypothesis to conclude that survivorship is indeed dependent on the age
# group. In other words, certain age groups had higher chances of survival than
# others. Let's also see which groups survived more than expected using
# standardized residuals.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("Age Group")
ax.set_title("age_group vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_age_group_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Among children and youth, survival rates are higher than expected
# - The survival rate of aged passengers is less than expected
# - The rest of the age groups had the expected survival rates
#
# **Takeaways**
#
# To sum up all the knowledge we've obtained by analyzing the `age` feature
# against the target variable `survived`, specifically the groups "survived"
# and "did not survive", we can state:
#
# 1. Descriptive statistics of the survived and not-survived groups are
# similar; we don't observe huge differences
# 2. These groups have different-looking distributions. However, the
# Mann-Whitney U test revealed that the distributions are not statistically
# significantly different
# 3. Despite the previous results, we've observed that certain age groups had
# higher chances of survival than expected
#
# Now that we have both "Sex versus Survived" and "Age versus Survived"
# covered, we can state that *"Women and children first"* was indeed the
# reality of what happened in the tragedy.


# %% [markdown]
# ### SibSp versus Survived


# %% [markdown]
# We'll investigate whether `sibsp` had any effect on a passenger's
# survivorship or not.


# %%
desc_survived_vs_cat(df, "sibsp")


# %%
ax = sns.countplot(data=df, x="sibsp", hue="survived", alpha=0.9)

ax.set_xlabel("Siblings/Spouses")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Sibling/Spouse Count")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_sibsp.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Groups `sibsp == 1` and `sibsp == 2` had balanced survivors and
# non-survivors
# - For `sibsp` equals 0, 3, and 4, there are more non-survivors
# - Groups 5 and 8 had no survivors at all
#
# We suspect that `survived` might be dependent based on what we've observed
# from the plot. Let's conduct a chi-square test of independence to be sure.
# We'll drop the 5 and 8 columns since they have no survivors, and that would
# violate data assumptions for the test otherwise.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `sibsp`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `sibsp`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
df_sibsp = df.loc[df.sibsp <= 4]

obs_val = pd.crosstab(df_sibsp["sibsp"], df_sibsp["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# The resulting p-value is significantly smaller than our significance level of
# $\alpha = 0.05$. Thus, we reject the null hypothesis to conclude that the
# target variable `survived` is dependent on `sibsp`. Let's also look  at the
# residuals to figure out this dependence.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("SibSp")
ax.set_title("sibsp vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_sibsp_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - The passengers without siblings and spouses on board had significantly
# fewer survivors than expected
# - The passengers with a sibling or a spouse had significantly higher survival
# rates than expected
# - Other groups were roughly as expected in terms of survival ratios
#
# The `sibsp == 0` case could be explained by the fact that the Titanic had
# numerous workers traveling to the United States. This idea is in line with
# our previous results from the `pclass` section, where we observed a higher
# number of non-survivors among third-class passengers, many of whom were the
# said workers.
#
# We will dive deeper into this suspicion in the multivariate analysis section.


# %% [markdown]
# ### ParCh versus Survived


# %% [markdown]
# We'll investigate whether `parch` had any effect on a passenger's
# survivorship or not.


# %%
desc_survived_vs_cat(df, "parch")


# %%
ax = sns.countplot(data=df, x="parch", hue="survived", alpha=0.9)

ax.set_xlabel("Parents/Childrens")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Parent/Children Count")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_parch.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Groups 0 and 5 had significantly more non-survivors than survivors
# - `parch == 1` group had more survivors
# - `parch == 2` group had exactly as many survivors as victims
# - Groups 4 and 6 had no survivors
#
# Same with the last case, we suspect that the target variable `survived` is
# dependent on the feature `parch`; thus, we'll conduct a chi-square test of
# independence. We'll drop columns 4 and 6 to satisfy the data assumptions.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `parch`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `parch`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
df_parch = df.loc[(df.parch != 4) & (df.parch != 6)]

obs_val = pd.crosstab(df_parch["parch"], df_parch["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Since the observed p-value is significantly less than the determined $\alpha$
# value of 0.05, we reject the null hypothesis and conclude that `survived` is
# dependent on the `parch` feature. Let's also analyze the residuals to
# understand this relationship.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("ParCh")
ax.set_title("parch vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_parch_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - The passengers with no parents or children had significantly lower survival
# rates than expected
# - The passengers with a parent or a child had significantly higher survival
# rates than expected
# - The passengers with 2 parents and/or children had slightly higher survival
# rates than expected
# - The rest of the groups are just as expected


# %% [markdown]
# ### Fare versus Survived


# %% [markdown]
# We'll investigate whether `fare` had any effect on a passenger's survivorship
# or not.


# %%
desc_survived_vs_cont(df, "fare")


# %%
ax = sns.boxplot(
    data=df.loc[df.fare <= 100], x="survived", y="fare", hue="survived",
    legend=False
)

ax.set_xticks([0, 1], ["No", "Yes"])

ax.set_xlabel("Survived")
ax.set_ylabel("Fare")
ax.set_title("Titanic Survivorship by Fare")

plt.savefig("../assets/surv_vs_fare_box.png")
plt.show()


# %%
ax = sns.violinplot(
    data=df, x="survived", y="fare", hue="survived", legend=False
)

ax.set_xticks([0, 1], ["No", "Yes"])

ax.set_xlabel("Survived")
ax.set_ylabel("Fare")
ax.set_title("Titanic Survivorship by Fare")

plt.savefig("../assets/surv_vs_fare_vio.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Survived passengers paid higher fares across the statistics: mean, median,
# etc.
# - Survivors had more extreme fare outliers relative to the IQR compared to
# non-survivors
# - The fare difference between `survived` groups is so extreme that the 25th
# percentile of the survived passengers' fare, GBP 12.5, is more than the
# median non-survived passenger fare
# - Both the mean and the median survived passenger fares are more than double
# that of the non-survived passenger fare
#
# We'll still conduct a statistical test regardless of how obvious the
# relationship may look. We're testing two independent groups with non-normal
# distributions of continuous, independent variables against a dependent
# variable, `fare` versus `survived`. Therefore, we'll use the Mann-Whitney U
# test since it is non-parametric and we also satisfy the other conditions.
# However, before we state our hypothesis, we should determine whether the
# distributions of survived and non-survived passenger fares are similar or
# not.


# %%
stat, pval  = stats.levene(
    df[df.survived == 0]["fare"],
    df[df.survived == 1]["fare"]
)

print(f"Levene Test p-value: {pval}")
stat, pval = stats.ks_2samp(
    df[df.survived == 0]["fare"],
    df[df.survived == 1]["fare"]
)

print(f"Kolmogorov-Smirnov Test p-value: {pval}")


# %% [markdown]
# Because the Levene and K-S tests show the groups have unequal variances and
# distribution shapes, we cannot use the Mann-Whitney U test to evaluate a
# difference in medians. Then, we can state our hypothesis for the Mann-Whitney
# test as follows:
#
# - **$\text{H}_0$**: There is no difference between the fare distributions of
# survivors and non-survivors:
# $$
#     P(X_\text{survivor} > X_\text{victim}) = 0.5
# $$
# - **$\text{H}_\text{a}$**: There is a difference between the fare
# distributions of survivors and non-survivors:
# $$
#     P(X_\text{survivor} > X_\text{victim}) \neq 0.5
# $$
#
# We'll set the significance level $\alpha = 0.05$.


# %%
survivors = df.loc[df.survived == 1, "fare"].values
victims = df.loc[df.survived == 0, "fare"].values

stat, pval = stats.mannwhitneyu(survivors, victims)

print(f"Stat: {stat}\np-value: {pval}")


# %% [markdown]
# Since the resulting p-value is less than our significance level $\alpha$, we
# reject the null hypothesis to conclude that fare distributions of survivors
# and non-survivors differ. Therefore, `survived` is dependent on the `fare`
# feature.


# %% [markdown]
# ### Embarked versus Survived


# %% [markdown]
# We'll investigate whether `embarked` had any effect on a passenger's
# survivorship or not.


# %%
desc_survived_vs_cat(df, "embarked")


# %%
ax = sns.countplot(data=df, x="embarked", hue="survived", alpha=0.9)

ax.set_xticks([0, 1, 2], ["Southampton", "Cherbourg", "Queenstown"])

ax.set_xlabel("Port of Embarkation")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Port of Embarkation")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_embarked.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - There are more victims than survivors among passengers who embarked from
# Queenstown and Southampton
# - There are more survivors among passengers who embarked from Cherbourg
#
# Let's conduct a statistical test to see if there's a relationship between
# survivorship and port of embarkation. For this, we'll use the chi-square test
# of independence.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `embarked`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `embarked`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
obs_val = pd.crosstab(df["embarked"], df["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Since the resulting p-value is smaller than our significance level, we reject
# the null hypothesis to conclude that `survived` is dependent on the
# `embarked` feature. Let's also look at the individual relationships.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("Embarked")
ax.set_title("embarked vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_embarked_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Passengers who embarked from Cherbourg survived at a rate significantly
# higher than expected
# - Passengers who embarked from Southampton survived at a rate significantly
# less than expected
# - Survival rates among passengers who embarked from Queenstown are as expected
#
# Although we've found a statistically significant relationship between
# `survived` and `embarked`, logically it wouldn't make sense for the port of
# embarkation to directly influence a passenger's survival. Therefore, it is
# possible that `embarked` is a proxy for `survived` where this feature embeds
# other features, such as passenger class. We'll test for this in the next
# section, which is multivariate analysis.


# %% [markdown]
# ### Deck versus Survived


# %% [markdown]
# We'll investigate whether `deck` had any effect on a passenger's survivorship
# or not.


# %%
desc_survived_vs_cat(df, "deck")


# %%
ax = sns.countplot(
    data=df,
    x="deck",
    hue="survived",
    alpha=0.9,
    order = ["A", "B", "C", "D", "E", "F", "G", "T"]
)

ax.set_xlabel("Deck")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Deck")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_deck.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - There are more survivors than victims among passengers who traveled in
# decks B and C, with deck B having the highest survival rate
# - Decks E, F, and G had significantly fewer survivors than victims
# - The only passenger to travel in the T deck did not survive
# - Decks A and D have a roughly 50-50 survival rate
#
# There seems to be a relationship between `deck` and `survived`. Let's conduct
# a chi-square test of independence just to be sure. Since deck T includes an
# expected value of 0, it violates the assumptions of the chi-square test.
# Thus, we'll drop the T deck before performing the test.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `deck`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `deck`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
df_deck = df.loc[df.deck != "T"]

obs_val = pd.crosstab(df_deck["deck"], df_deck["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Since the resulting p-value is smaller than our significance level, we reject
# the null hypothesis to conclude that `survived` is dependent on the `deck`
# feature. Let's also look at the individual relationships.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("Deck")
ax.set_title("deck vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_deck_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - For decks B and C, survival rates were significantly higher than expected
# - Deck D had higher-than-expected survival rates
# - Deck E and G had lower-than-expected survival rates
# - Deck F had significantly fewer survivors than expected
# - Deck A is as expected
#
# There's statistical evidence that `deck` is related to `survived`. The
# reasons could vary. For starters, some decks could have easier access to the
# boat deck where lifeboats were situated, which could explain why a passenger
# accommodated on the deck could explain his/her survival chance to some
# extent. On the other hand, the deck is directly related to passenger class
# and fare, which are also related to the survival rates. Hence, we can't say
# for certain whether the deck feature directly contributes to survival chance
# or not, which we'll test in multivariate analysis.


# %% [markdown]
# ### Title versus Survived


# %% [markdown]
# We'll investigate whether `title` had any effect on a passenger's
# survivorship or not. We'll combine every title with fewer than 7 occurrences
# into a single group so that it's easier to plot and conduct a statistical
# test.


# %%
title_counts = df["title"].value_counts()

def group_title(title):
    if pd.isna(title):
        return "Other"
    elif title_counts.get(title, 0) >= 7: # < 7 occurences gets filtered
        return title
    else:
        return "Other"

df["title_grouped"] = df["title"].apply(group_title)

desc_survived_vs_cat(df, "title_grouped")


# %%
ax = sns.countplot(data=df, x="title_grouped", hue="survived", alpha=0.9)

ax.set_xlabel("Honorific Title")
ax.set_ylabel("Count")
ax.set_title("Titanic Survivorship by Honorific Title")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.savefig("../assets/surv_vs_title.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Male-only titles except Master. more victims than survivors
# - Female-only titles had more survivors than victims
#
# The results are in line with the section where we analyzed `survived` versus
# `sex`. Recall that the `title` column was extracted to verify the `sex`
# column. Thus, the results are as expected. Let's conduct a chi-square test of
# independence.
#
# We state the hypothesis:
#
# - **$\text{H}_0$**: `survived` is independent of `title`
# - **$\text{H}_\text{a}$**: `survived` is dependent on `title`
#
# We'll set the significance level $\alpha = 0.05$.


# %%
obs_val = pd.crosstab(df["title_grouped"], df["survived"])
contingency_table = obs_val
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print(f"chi2: {chi2}\np-value: {pval}\ndof: {dof}")


# %% [markdown]
# Since the resulting p-value is smaller than our significance level, we reject
# the null hypothesis to conclude that `survived` is dependent on the `title`
# feature. Let's also look at the individual relationships.


# %%
table = Table(obs_val)
table.test_nominal_association()
std_resids = table.standardized_resids

ax = sns.heatmap(std_resids, annot=True, cmap="flare", center=0, fmt=".2f")

ax.set_xlabel("Survived")
ax.set_ylabel("Title")
ax.set_title("title vs survived Standardized Residuals")

plt.savefig("../assets/surv_vs_title_resids.png")
plt.show()


# %% [markdown]
# **Remarks**
#
# - Mr. title had extremely lower survival rates than expected
# - Mrs. and Miss. titles had extremely higher survival rates than expected
# - Master. title had higher survival rates than expected
# - The rest of the groups are as expected
#
# These are the most extreme residuals we've encountered so far. A good reason
# for this could be the fact that titles can reflect both age and sex. We'll
# investigate this assumption in greater detail at later stages.
#
# In addition, as with the `embarked` and `deck` features, we can't be sure
# whether a passenger's title directly contributes to their survival chances.
# An honorific title is directly related to sex, which explains survival
# chances to a great extent. Therefore, we'll test the title feature in a
# multivariate analysis and determine whether it's a proxy for or directly
# related to `survived`.


# %% [markdown]
# ### Summary


# %% [markdown]
# - We have found that the `survived` target variable is dependent on every
# single feature we've tested in this section
# - We'll check whether `embarked`, `deck`, and `title` are proxies or not
# - Although the relationship isn't immediately visible in some cases, for
# example, the statistical tests we've conducted did not provide information on
# dependence in the case of the age feature, we've still found that there exist
# naturally occurring groups whose survival chances were higher than expected.
