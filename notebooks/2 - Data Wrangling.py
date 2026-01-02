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
#     name: .venv
# ---

# %% [markdown]
# # 2 - Data Wrangling

# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** January, 2026

# %% [markdown]
# ## Introduction

# %% [markdown]
# In the previous section of the project, we conducted initial data wrangling.
# We've found that the data contains no duplicates; however, there are missing
# values that we need to address.
#
# Let's start by importing the modules and the data.

# %% [markdown]
# ## Imports

# %%
import sqlite3

import numpy as np
import pandas as pd

conn = sqlite3.connect(":memory:")

df = pd.read_csv("../data/raw/train.csv", encoding="utf-8")
df.head()

# %% [markdown]
# ## Data Cleaning & Validation

# %% [markdown]
# Recall that we've already checked for duplicates in the understanding and
# planning phase, and we've found no duplicates. Let's start by converting the
# column names to lowercase.

# %%
df.columns = df.columns.str.lower()

# %% [markdown]
# Let's also validate the data types.

# %%
df.dtypes

# %% [markdown]
# Data types are correct. Then, we can check for invalid values. For example,
# we expect `survived` to be only 0 or 1.

# %%
df["sex"].value_counts(dropna=False)

# %%
df["embarked"].value_counts(dropna=False)

# %%
df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].agg(["min", "max"])

# %% [markdown]
# We see no invalid values among these columns except for `fare`, where there
# exist rows with a `fare` amount of 0. We shall validate the `fare` column.

# %%
df["fare"].describe()

# %% [markdown]
# Now, there isn't much we can do to validate the `fare` column. We can verify
# that the most expensive ticket was in fact GBP 512.00, and some tickets were
# handed out for free, as we see with the minimum. The mean ticket price also
# makes sense, but besides these, there isn't much more we can do for the time
# being.
#
# Titanic accommodated luxurious cabins with utmost comfort, especially for the
# first-class passengers. Below is a cutaway diagram depicting these
# facilities.
#
# <center><div><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Olympic_%26_Titanic_cutaway_diagram.png/960px-Olympic_%26_Titanic_cutaway_diagram.png" alt="Titanic cutaway diagram" width="400"/></div></center>
#
# > "The accomodation for first-class passengers is placed amidships and
# > extends over five decks, the promenade (A), bridge (B), shelter (C), saloon
# > (D), and upper (E) decks. *(Titanica, 2025a)*"
#
# First-class accommodations were placed in cabins A through E, second and
# third classes in D through G.
#
# In our dataset, we see the following cases:


# %%
df.to_sql("titanic", conn, if_exists="replace", index=False)

query = """
SELECT DISTINCT(cabin) FROM titanic
WHERE cabin IS NOT NULL
LIMIT 30
"""

result = pd.read_sql_query(query, conn)
print(result)

# %% [markdown]
# - Cabin numbers start with a capital English letter
# - Some passengers accommodated multiple cabins
#     - This is likely due to cabin numbers being shared between passenger
#     groups, such as families (similar to the ticket number)
# - Some passengers have cabin numbers starting with F, followed by another
# cabin string, eg, F G73
# - There exists a "T" cabin
# - Some passengers don't have full cabin numbers but just the deck letter
#
# Firstly, let's investigate the cabin T.

# %%
df.loc[df.cabin.str.contains("T") == True, "name"]

# %% [markdown]
# There's only 1 passenger with this cabin number, Mr. Stephen Weart Blackwell.
# According to Titanica (2025b), Mr. Blackwell was the only passenger with the
# [cabin number T](https://www.encyclopedia-titanica.org/cabins.html), and this
# number referred to the [boat
# deck](https://www.encyclopedia-titanica.org/titanic-deckplans/boat-deck.html).
# Thus, this is a legitimate entry.
#
# Now, the entries like "F G73" are not mistakes either. The reason is that
# this cabin number denotes the deck F, section G, cabin 73. We infer this
# information from the [Titanic's
# plans](https://www.encyclopedia-titanica.org/titanic-deckplans/f-deck.html)
# themselves.
#
# We're not yet sure whether cabin numbers themselves are correlated with the
# survival of a passenger, though the deck information of the cabin number
# could be. Therefore, we'll create a new column called `deck` that we'll fill
# in by extracting it from the cabin number itself. This approach has 3 main
# benefits:
#
# 1. We can fill in missing cabin numbers more reliably
# 2. Reduces model complexity by focusing on a broader feature
# 3. Reduces the risk of overfitting


# %%
def extract_deck(cabin: str) -> str | list:
    """
    Extracts the deck information (A, B, C, etc.) from the cabin number.

    Parameters
    ----------
    cabin : str
        Cabin number

    Returns
    -------
    str
        Cabin's deck
    """
    lst = cabin.split(" ")
    if len(lst) == 1:  # cabin number is of type letter + number
        return lst[0][0]
    if all([lst[0][0] == x[0] for x in lst]):
        return lst[0][0]
    if lst[0][0] == "F" and lst[1][0] in [
        "G",
        "E",
    ]:
        return f"F-{lst[1][0]}"
    return lst


# %%
df.loc[df["cabin"].notna(), "cabin"].map(extract_deck).unique()

# %% [markdown]
# We see that only F G000 and F E000 types of cabin numbers exist, while there
# are other such possibilities, such as F R000. Although some sections, such as
# F-J, did not contain any rooms, some sections that had rooms are still not
# present in the dataset. Therefore, instead of distinguishing between a deck's
# sections, we'll just use the main deck string and exclude the section.


# %%
def extract_deck_sectionless(cabin: str) -> str | list:
    """
    Extracts the deck information (A, B, C, etc.) from the cabin number.
    Ignores the sections such as F G73, F R171, etc. and only returns the deck
    letter.

    Parameters
    ----------
    cabin : str
        Cabin number

    Returns
    -------
    str
        Cabin's deck
    """
    lst = cabin.split(" ")
    if len(lst) == 1:
        return lst[0][0]
    if all([lst[0][0] == x[0] for x in lst]):
        return lst[0][0]
    if lst[0][0] == "F" and lst[1][0] in ["G", "E"]:
        return f"F"  # Only this line is modified
    return lst


df["deck"] = df.loc[df["cabin"].notna(), "cabin"].map(extract_deck_sectionless)
df.loc[df["cabin"].notna(), ["cabin", "deck"]].head(10)

# %% [markdown]
# Notice that our assumption that the group tickets' cabin numbers were all in
# the same deck was true for the non-missing data. We'll keep this assumption
# in mind when filling in the missing values for the cabin column.
#
# Lastly, let's check if the assumption that the first-class accommodations are
# in decks A through E, and second and third classes in decks D through G.

# %%
invalid = (
    "(pclass == 1 and deck in ['F', 'G']) or "
    "(pclass == 2 and deck in ['A', 'B', 'C']) or "
    "(pclass == 3 and deck in ['A', 'B', 'C'])"
)

mismatched = df.dropna(subset=["cabin"]).query(invalid)
mismatched[["passengerid", "pclass", "deck", "cabin"]]

# %% [markdown]
# There are no mismatches between passenger classes and their decks. Thus, the
# `cabin` column contains no invalid entries.
#
# Lastly, we can validate the `sex` column using passenger names. Firstly, we
# can extract passenger titles from the `name` column. To do that, we need to
# extract what titles are present in the dataset. What makes it easy in our
# case is that titles end with a dot at the end of the word, so we can filter
# by using that. Also, the names are consistent in formatting. Every name
# string starts with the surname, followed by the title with a dot at the end,
# and then the first and middle names follow. For example, Ward, Miss. Anna.

# %%
titles = [
    df["name"]
    .dropna()
    .str.split()
    .explode()
    .loc[lambda s: s.str.contains(".", regex=False)]
    .unique()
    .tolist()
]

print(titles)

# %% [markdown]
# Apart from one mistake in the titles ("L."), the rest of them are valid
# titles in English, French, Italian, etc. The "L." comes from a shortened name
# in the dataset. Below is an SQL query for the string "L."


# %%
df.to_sql("titanic", conn, if_exists="replace", index=False)
query = """
SELECT NAME FROM TITANIC
WHERE NAME LIKE "% L. %"
"""
result = pd.read_sql_query(query, conn)
print(result)

# %% [markdown]
# These titles can be explained as follows:
# - ["Mr."](https://en.wikipedia.org/wiki/English_honorifics): for men,
# regardless of marital status, who do not have another professional or
# academic title
# - ["Mrs."](https://en.wikipedia.org/wiki/English_honorifics): for married
# women who do not have another professional or academic title
# - ["Miss."](https://en.wikipedia.org/wiki/English_honorifics): for girls,
# unmarried women, and (in the United Kingdom) married women who continue to
# use their maiden name
# - ["Master."](https://en.wikipedia.org/wiki/English_honorifics): for boys and
# young men, or as a style for the heir to a Scottish peerage
# - ["Don."](https://en.wikipedia.org/wiki/Don_(honorific)): commonly used in
# Spain, Portugal, and Italy, it is an honorific prefix derived from the Latin
# Dominus, meaning "lord" or "owner"
# - ["Rev."](https://en.wikipedia.org/wiki/English_honorifics): used generally
# for members of the Christian clergy, regardless of affiliation, but
# especially in Catholic and Protestant denominations, for both men and women
# - ["Dr."](https://en.wikipedia.org/wiki/English_honorifics): for the holder
# of a doctoral degree in many countries, and for medical practitioners,
# dentists, and veterinary surgeons
# - ["Mme."](https://en.wikipedia.org/wiki/Title): the French abbreviation for
# Madame, for women, a term of general respect or flattery, originally used
# only for a woman of rank or authority
# - ["Ms."](https://en.wikipedia.org/wiki/English_honorifics): for women,
# regardless of marital status or when marital status is unknown
# - ["Major."](https://en.wikipedia.org/wiki/Title): a military title
# - ["Lady."](https://en.wikipedia.org/wiki/English_honorifics): for female
# peers with the rank of baroness, viscountess, countess, and marchioness, or
# the wives of men who hold the equivalent titles
# - ["Sir."](https://en.wikipedia.org/wiki/English_honorifics): for men,
# formally, if they have a British knighthood or if they are a baronet
# - ["Mlle."](https://en.wikipedia.org/wiki/Mademoiselle_(title)): is a French
# courtesy title traditionally given to an unmarried woman
# - ["Col."](https://en.wikipedia.org/wiki/Title): a military title
# - ["Capt."](https://en.wikipedia.org/wiki/Title): a military title or a
# ship's highest responsible officer acting on behalf of the ship's owner
# - ["Countess."
# ](https://dictionary.cambridge.org/dictionary/english/countess): a woman of
# high social rank, or the wife of a count or earl
# - ["Jonkheer."](https://en.wikipedia.org/wiki/Jonkheer): is literally
# translated as 'young lord' in Dutch
#
# (Paraphrased or directly quoted from the references)
#
# Let's now create a column `title` from the names.

# %%
titles = [
    "Mr.",
    "Mrs.",
    "Miss.",
    "Master.",
    "Don.",
    "Rev.",
    "Dr.",
    "Mme.",
    "Ms.",
    "Major.",
    "Lady.",
    "Sir.",
    "Mlle.",
    "Col.",
    "Capt.",
    "Countess.",
    "Jonkheer.",
]


def extract_title(name: str) -> str | None:
    """
    Extracts the honorific title from the name string

    Parameters
    ----------
    name : str
        Passenger name

    Returns
    -------
    str | None
        Honorific title
    """
    str_list = name.split(" ")
    for _str in str_list:
        if _str in titles:
            return _str
    return None


# %%
df["title"] = df["name"].apply(extract_title)

print(f"Missing title values count: {df.title.isna().sum()}")
df[["name", "title"]].head(10)

# %% [markdown]
# Let's also do a sanity check.

# %%
df["title"].isna().sum()

# %% [markdown]
# We understand that every name entry in the database contains a title and that
# we had no errors extracting these titles.
#
# Now, we can cross-reference titles with their respective genders. Note that
# some of the titles in our dataset are gender-neutral. Thus, we'll only check
# for the ones that are gender-specific initially.

# %%
pd.crosstab(df["sex"], df["title"])

# %% [markdown]
# We see that the gender-specific titles all match a passenger's sex. On the
# other hand, we cannot methodically check for gender-neutral honorifics, and
# the only way to verify those would be to individually research the passengers
# with these titles. This verification is out of the scope of this project and
# will not be needed for our purposes. Therefore, we deem that we've verified
# the `sex` column using the passenger titles we've extracted from their names.


# %% [markdown]
# ## Missing Values

# %% [markdown]
# Now that we've validated our data, let's proceed with filling the missing
# values in.

# %%
na_counts = df.isna().sum()
na_counts[na_counts > 0]

# %% [markdown]
# ### Embarked column


# %% [markdown]
# We'll start with the `embarked` column. Referring back to the data dictionary
# we've provided in the notebook `Understanding & Planning`, `embarked` column
# refers to the port of embarkation, meaning where they've boarded the Titanic.
# The three possible values this column takes are `C` for Cherbourg, France,
# `Q` for Queenstown, Ireland, and `S` for Southampton, England.
#
# > Titanic first departed from Southampton, England. It made two port calls,
# > the first being Queenstown, Ireland, and the second being Cherbourg,
# > France. This route was popular among British ocean liners for the
# > Southampton-New York route.
# >
# > *"Titanic's maiden voyage was intended to be the first of many
# > trans-Atlantic crossings between Southampton and New York via Cherbourg and
# > Queenstown on westbound runs."*
# >
# > (*Wikipedia contributors, 2025*).
#
# Let's first see the passengers with missing boarding locations and then try
# to make some connections to fill these missing values.

# %%
df[df["embarked"].isna()]

# %% [markdown]
# Both passengers have first-class tickets for cabin "B28", and they share the
# same ticket number. Therefore, they've traveled together. In this case, it
# makes sense that they also boarded the liner together.

# %%
df[df.pclass == 1].groupby(by="embarked")["embarked"].value_counts()

# %% [markdown]
# We see that the vast majority of the passengers with `pclass` as 1 embarked
# from Southampton, followed by Cherbourg.
#
# Considering the vast majority of 1st class passengers boarded the Titanic
# from Southampton, we'll assign the `Embarked` as `S` for these passengers.

# %%
df["embarked"] = df["embarked"].fillna(value="S")
df["embarked"].isna().sum()


# %% [markdown]
# ### Age

# %% [markdown]
# The dataset has 177 missing age values. Let's first investigate missing
# values in the `age` column per title.

# %%
na_age_titles = df.loc[df.age.isna(), "title"].unique()
print(f"Missing age values per title: {na_age_titles}")

# %%
df.loc[df.title.isin(na_age_titles), "title"].value_counts()

# %% [markdown]
# We observe that the titles with missing age values contain nonempty entries
# as well. Therefore, we can use groupings to impute the missing age entries.
# For the majority of the cases, we can just use the medians to fill the
# missing values. However, we should handle some cases manually. For example,
# Miss. title includes girls and unmarried woman and first-class passengers
# often traveled with the family maids, who were often unmarried women.
# Therefore, we'll determine some cases to handle first.
#
# For titles Dr. and Master., we can impute using title medians. Let's start
# with that first.

# %%
dr_median = df.loc[df.title == "Dr.", "age"].median()
df.loc[(df.title == "Dr.") & (df.age.isna()), "age"] = dr_median

master_median = df.loc[df.title == "Master.", "age"].median()
df.loc[(df.title == "Master.") & (df.age.isna()), "age"] = master_median

# %% [markdown]
# Next up, let's manually fill missing the young girl ages. We can filter by
# title Miss. and `parch`. Recall that `parch` denotes the number of parents or
# children onboard. Since Miss. title is exclusive to unmarried women and
# having children without marriage was not common at all at the time, we can
# assume that `parch` > 0 implies they have parents, which means that they are
# girls or young women.

# %%
mask = (df.title == "Miss.") & (df.parch > 0)
young_girl_median = df.loc[mask, "age"].median()
df.loc[mask & (df.age.isna()), "age"] = young_girl_median

# %% [markdown]
# For the rest of the passengers, we can impute values with title and passenger
# class medians. The reason we include the passenger class is simple.
# First-class passengers were generally wealthy businessmen, who were generally
# middle-aged or older, and third-class passengers were usually workers and
# immigrants, who were younger.

# %%
df["age"] = df["age"].fillna(
    df.groupby(by=["title", "pclass"])["age"].transform("median")
)
print(f"Missing age values count: {df.age.isna().sum()}")

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>NOTE:</b> Although our methodology was solid, we may have introduced
#     some noise to the dataset. In future versions, if a more careful
#     procedure for preparing the data for model training is deemed necessary,
#     this section can be improved.
#
# Here are some improvement options:
#
# 1. Refine the method
# 2. Manually fill as much data as possible (some missing values from the
# dataset can be found by research)
# 3. Automation or web scraping for filling as much data as possible.
#
# </div>
#
# Titanic has been a research area for many years, and through the collective
# efforts of many brilliant minds, a lot of the missing data has been
# recovered. Therefore, we can fill in the missing passenger ages in case they
# are known but not present in our dataset. This kind of improvement is out of
# our scope, though.


# %% [markdown]
# ### Cabin

# %% [markdown]
# As we've briefly mentioned before, filling individual cabin numbers is a
# near-impossible task with a high risk of introducing a lot of noise. Instead,
# we can fill the `deck` column we've added. To do this, we can make use of the
# fact that decks A-E included first-class cabins while decks D-G included
# second and third-class cabins.
#
# We can group decks by `pclass` and `fare` to fill in the missing values.
# However, one issue with the `fare` column is that it is not passenger-based
# but rather ticket-based. For example, a family can buy a single ticket with a
# whole sum fare, and both the ticket number and the fare amount will be the
# same for every member of the family.
#
# We could first resolve the `fare` column before imputation. However, the
# difficulties with the fare are
#
# 1. The training dataset may not include every member with the same ticket
# number.
#     - For example, if a family of 5 bought a 5-person ticket, but our
#     training data includes only 4 of the members, then doing a basic
#     calculation like `fare = fare / 4` for each passenger will result in an
#     incorrect ticket cost.
# 2. Some tickets could be handed out for </b>ee or with discounts, which we
# have no way of knowing.
# 3. Apparently, ticket discounts were applied based on age: adults, children,
# and babies had different ticket costs.
#     - Since we've imputed 177 missing age values, we may introduce more noise
#     by feature extraction for `age`.
#
# On the other hand, not dealing with the `fare` column can also misguide us in
# imputation. Before we make a strategy, let's first fill in `deck` column
# based on `ticket`: if a passenger with an empty `deck` column shares his/her
# ticket with another passenger with a valid `deck` value, we can use this for
# imputation.


# %%
def fill_same_ticket_decks():

    na_count_before = df.deck.isna().sum()

    ticket_deck_mode = (
        df.dropna(subset=["deck"])
        .groupby(by="ticket")["deck"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    df["deck"] = df["deck"].fillna(df["ticket"].map(ticket_deck_mode))

    na_count_after = df.deck.isna().sum()

    print(f"Filled {na_count_before - na_count_after} missing values.")


# %%
fill_same_ticket_decks()

# %% [markdown]
# We've only filled 11 missing values. There are still 676 missing values. Now,
# we need to make a decision.
#
# All things considered, imputing `deck` by using `pclass` and `fare` seems to
# be the most logical approach. Due to how each cabin (and therefore its
# corresponding deck) was designed to target a specific wealth group, there
# should be a natural distinction between decks. Hence, we can use a predictive
# model for imputation. In our case, a random forest classifier works the best
# since
#
# - Random forests are resistant to nonlinearity and outliers
# - Doesn't require encoding (which can introduce nonexistent ordering between
# categories)
# - Should work well with the `fare` despite our difficulties making inferences

# %%
ticket_counts = df["ticket"].map(df["ticket"].value_counts())
family_size = df["sibsp"] + df["parch"] + 1

group_size = np.where(
    ticket_counts > 1,
    np.maximum(ticket_counts, family_size),
    family_size
)

df["fare_per_person"] = df["fare"] / group_size

df[["fare", "fare_per_person"]].head(10)

# %% [markdown]
# What we essentially do above is as follows:
#
# 1. We first count how many passengers share the same ticket number
# 2. Then, we estimate group size using `sibsp` and `parch` columns
# 3. If a ticket is shared, we split the fare among either the ticket count or
# the group size
# 4. If a ticket is not shared, the fare is split between family members only
# 5. Compute each passenger's fare per person by `fare / group_size`
#
# Of course, we're still not able to capture every fare per person value
# accurately. For example, one case our code doesn't account for is when a
# passenger in our training data doesn't share his/her ticket number with
# someone else, but in reality, their peer is in the testing data (which we do
# not have access to until we train our model). We've accounted for such cases
# and decided to use `parch` and `sibsp` for solving this issue, but it remains
# if the person they shared their ticket with does not fall into those two
# categories, such as a maid not counted for either of those columns, and our
# code will still not capture them.
#
# The bottom line here is that we cannot deterministically capture every single
# "fare per person" amount; we can only approximate. Therefore, the tree
# classifier we'll build for imputing the decks will also be subject to these
# approximations. As a result, we cannot really expect such great accuracy. On
# the contrary, with these few predictor variables and a small sample size, a
# higher accuracy would hint at extreme overfitting, data leakage, etc., rather
# than hinting at discovering relations between feature and target variables
# for imputation.
#
# Let's proceed with developing our model.


# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)

# %%
seed = 6

rf_data = df.drop("cabin", axis=1).dropna(axis=0)
rf_data = rf_data[rf_data["deck"] != "T"]

X = rf_data[["fare_per_person", "pclass"]]
y = rf_data["deck"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state = seed
)

print(X_train.shape, y_test.shape)

# %%
rf = RandomForestClassifier(n_jobs=-1, random_state=seed)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

param_grid = {
    "n_estimators": np.arange(10, 101, 10),
    "max_depth": np.arange(3, 16, 2),
    "min_samples_split": [2, 3, 4],
    "min_impurity_decrease": [0, 0.000125, 0.00025, 0.005],
}

clf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
    cv=cv,
    verbose=1,
)

clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

print(f"Accuracy: {balanced_accuracy_score(y_test, y_pred)}")

print(clf.best_params_)


# %% [markdown]
# 72.34% accuracy is sufficient for our case. We won't go much deeper into why,
# but one quick metric we could obtain is to train a base model with the
# following logic:
#
# - Predict decks A-E for `pclass == 1`
# - Predict decks D-G for `pclass == 2` or `pclass == 3`
#
# Let's also investigate the confusion matrix.

# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

disp.plot()

# %% [markdown]
# Here's a breakdown of the confusion matrix:
#
# - Deck A is always confused with its neighbors
# - Deck B is often predicted correctly, model's accuracy is quite high
# - Deck C is similar to deck B, mostly correct but sometimes confused with its
# neighbors
# - Deck D neighbor confusion is a bit higher than deck B and C
# - Deck E is 100% accurate, could even be overfit
# - Deck F is the same case as deck E
# - Deck G has only 1 occurence which was predicted correctly
#
# Decks are often confused with their neighbors. This hints at a possible
# improvement in feature engineering stage: grouping decks into classes. For
# example, we can group decks into "Higher", "Middle", and "Lower" classes such
# as
#
# ```python
# {
#   "higher": ["A, B"],
#   "middle": ["C", "D", "E"],
#   "lower": ["F", "G"]
# }
# ```
#
# However, this is out of our scope for this section. What we consider here is
# to whether use this model for imputing missing `deck` values or not. We'll
# proceed with doing so. For that, we'll retrain the model with `best_params_`
# obtained from GridSearchCV with the whole data and use `predict_proba`. We'll
# assign a predicted deck to a passenger only when the model's confidence is
# above our threshold. We can start with 60% and tune this parameter later.

# %%
model = RandomForestClassifier(random_state=42, n_jobs=-1, **clf.best_params_)

model.fit(X, y)

# %% [markdown]
# Let's also `pickle` this model for future use.

# %%
import pickle

with open("../models/deck_imputer.pkl", "wb") as file:
    pickle.dump(model, file)


# %% [markdown]
# Now, we can impute the missing decks for passengers who share their tickets
# first. We'll make it so that we'll only impute the first passenger
# corresponding to the duplicated ticket, then set the decks for the rest of
# the passengers with the same ticket number as the first passenger.

# %%
def impute_decks(
    data: pd.DataFrame,
    passenger_idx: pd.Series,
    threshold: float = 0.6
) -> None:
    """
    Imputes the missing deck information of the specified passengers.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataframe
    passenger_idx : pd.Series
        Passenger indices series
    threshold : float
        Minimum confidence level requirement for RandomForestClassifier
        imputation

    Returns
    -------
    None
        Imputes the values directly into the given pd.DataFrame object
    """
    X_missing = data.loc[passenger_idx, ["fare_per_person", "pclass"]]

    proba = model.predict_proba(X_missing)
    classes = model.classes_

    max_proba = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    pred_labels = classes[pred_idx]

    na_count_before = data.deck.isna().sum()

    df.loc[passenger_idx, "deck"] = np.where(
        max_proba >= threshold,
        pred_labels,
        np.nan
    )

    na_count_after = data.deck.isna().sum()

    print(f"Imputed {na_count_before - na_count_after} missing values.")

    return None


# %%
impute_decks(data=df, passenger_idx=df.ticket.duplicated(keep="first"))

# %% [markdown]
# Now, we can set the decks of the passengers with the same tickets again.

# %%
fill_same_ticket_decks()

# %%
df.deck.isna().sum()

# %% [markdown]
# We still have 507 missing values to deal with. For the rest of the data, we
# can lower the threshold to 0, perform a check on whether `pclass` matches the
# assigned deck, and apply the same deck to the other ticket holders as we just
# did.

# %%
impute_decks(
    data=df,
    passenger_idx=df.ticket.duplicated(keep="first"),
    threshold=0.0
)

invalid = (
    "(pclass == 1 and deck in ['F', 'G']) or "
    "(pclass == 2 and deck in ['A', 'B', 'C']) or "
    "(pclass == 3 and deck in ['A', 'B', 'C'])"
)

mismatched = df.dropna(subset=["deck"]).query(invalid)
mismatched[["passengerid", "pclass", "deck"]]

# %% [markdown]
# We don't have any inconsistencies regarding `pclass` and `deck` mismatches.
# Therefore, we can proceed with imputing the rest of the values.

# %%
fill_same_ticket_decks()

print(f"Missing deck values count: {df.deck.isna().sum()}")

# %% [markdown]
# Now, we have 458 passengers left who do not share their ticket with anyone
# else. We'll impute these values and then check whether there's an
# inconsistency.

# %%
impute_decks(data=df, passenger_idx=df.deck.isna(), threshold=0.0)

print(f"Missing deck values count: {df.deck.isna().sum()}")

invalid = (
    "(pclass == 1 and deck in ['F', 'G']) or "
    "(pclass == 2 and deck in ['A', 'B', 'C']) or "
    "(pclass == 3 and deck in ['A', 'B', 'C'])"
)

mismatched = df.dropna(subset=["cabin"]).query(invalid)
mismatched[["passengerid", "pclass", "deck", "cabin"]]

# %% [markdown]
# We've successfully imputed the `deck` column without any inconsistencies!
# We'll save the modified data for future use in our EDA and model building. 

# %%
df.to_csv(
    "../data/modified/cleaned.csv",
    encoding="utf-8",
    header=True,
    index=False
)

# %% [markdown]
# ## Takeaways

# %% [markdown]
# In the data wrangling phase, we've completed the following procedures:
#
# 1. Imported the data
# 2. Converted column names to lowercase letters
# 3. Validated data types
# 4. Extracted a new column `deck` from `cabin` column
# 5. Extracted a new column `title` from `name` column
# 6. Validated the data (except for the `fare` column, read the related section
# for more details)
# 7. Imputed 2 `embarked`, 177 `age`, and 687 `deck` values
#
# Our strategy for imputation was as follows:
#
# 1. In `embarked` column, there were only 2 missing values. Both passengers
# shared the same ticket number, which indicated that they probably both
# embarked from the same place. We've checked which port was the most common
# among first-class passengers to embark from. Then, we've assigned that value
# to the missing entries.
# 2. In `age` column, we've followed different strategies for handling missing
# values. For example, we've handled the "Master." title case separately since
# that title is specific to a certain age group. We sliced the passengers into
# groups using their titles, and for the rest, we've used a title and passenge
# class based median calculation and assigned the value to the missing entries.
# 3. Instead of imputing the `cabin` column, we've decided to impute the `deck`
# column as it would introduce less noise. We've built a random forest
# classifier and trained it on a feature-engineered column `fare_per_person`.
# We hyperparameter-tuned this model and used it to impute the missing values.
# We've also checked whether there were inconsistencies, for example, a
# first-class passenger being assigned to deck G, but there were none.
#
# We've saved the resulting `pd.DataFrame` object as a new dataset under the
# directory `data/modified/cleaned.csv`, which we'll use in the upcoming
# sections.
#
# Throughout this section, we had to make certain decisions on our methodology
# for validating the existing data and imputing the missing data. We've stated
# the difficulties, possible consequences of our choices, and what can be
# improved with our work if a refinement is needed in the future. Overall, I'm
# satisfied with this level of work. We've stayed realistic, made solid steps
# towards obtaining a cleaner dataset, and we've been conscious of the
# decisions we had to make.

# %% [markdown]
# ## References

# %% [markdown]
# - countess. (2026).
# https://dictionary.cambridge.org/dictionary/english/countess
# - Research guide D1: RMS Titanic: Fact sheet. (n.d.). Royal Museums
# Greenwich.
# https://www.rmg.co.uk/collections/research-guides/research-guide-d1-rms-titanic-fact-sheet
# - Titanic Deckplans : RMS Titanic : Plan of Boat Deck. (n.d.).
# https://www.encyclopedia-titanica.org/titanic-deckplans/boat-deck.html
# - Titanic Deckplans : RMS Titanic : Plan of F Deck. (n.d.).
# https://www.encyclopedia-titanica.org/titanic-deckplans/f-deck.html
# - Titanica, E. (2025a, September 3). Olympic & Titanic : Passenger
# Accommodation.
# https://www.encyclopedia-titanica.org/passenger-accommodation.html
# - Titanica, E. (2025b, September 16). Cabin Allocations.
# https://www.encyclopedia-titanica.org/cabins.html
# - Wikipedia contributors. (n.d.). File:Olympic & Titanic cutaway diagram.png
# - Wikipedia.
# https://en.wikipedia.org/wiki/File:Olympic_%26_Titanic_cutaway_diagram.png
# - Wikipedia contributors. (2025a, March 1). Mademoiselle (title). Wikipedia.
# https://en.wikipedia.org/wiki/Mademoiselle_(title)
# - Wikipedia contributors. (2025b, September 28). Don (honorific). Wikipedia.
# https://en.wikipedia.org/wiki/Don_(honorific)
# - Wikipedia contributors. (2025c, October 10). Jonkheer. Wikipedia.
# https://en.wikipedia.org/wiki/Jonkheer
# - Wikipedia contributors. (2025d, December 5). English honorifics. Wikipedia.
# https://en.wikipedia.org/wiki/English_honorifics
# - Wikipedia contributors. (2025e, December 17). First-class facilities of the
# Titanic. Wikipedia.
# https://en.wikipedia.org/wiki/First-class_facilities_of_the_Titanic
# - Wikipedia contributors. (2025f, December 19). Title. Wikipedia.
# https://en.wikipedia.org/wiki/Title Wikipedia contributors. (2025g, December
# 21). Titanic. Wikipedia. https://en.wikipedia.org/wiki/Titanic
