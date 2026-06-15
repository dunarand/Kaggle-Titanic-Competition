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
# # 2 - Data Wrangling


# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** March, 2026


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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
print(df.dtypes)


# %% [markdown]
# As stated in the previous notebook, the `survived`, `pclass`, `sex`, and
# `embarked` features are categorical variables. We'll convert these variables.
#
# One remark here is the conversion of `pclass` from an integer feature to a
# categorical feature. Even though it is not necessary for a tree-based model,
# this conversion is the correct approach where the implied "distances" between
# categories are not the same as their numeric values. For example, it may not
# be the case that the "distance" between first and second-class passengers is
# equal to the "distance" between second and third-class passengers. Therefore,
# for other model types such as linear models, it is a good practice to convert
# these variables to categorical types or transform them to uncover underlying
# ordering and differences properly. In our case, a tree-based model is able to
# make these deductions by itself, but it is still a good practice nonetheless.


# %%
cat_cols = ["survived", "pclass", "sex", "embarked"]
df[cat_cols] = df[cat_cols].astype("category")

print(df.dtypes)


# %%
df[["age", "fare"]].agg(["min", "max"])


# %%
for col in ["sex", "sibsp", "parch", "embarked"]:
    print(df[col].value_counts(dropna=False, sort=False))
    print()


# %% [markdown]
# We see no invalid values among these columns except for `fare`, where there
# exist rows with a `fare` amount of 0.


# %%
df["fare"].describe()


# %% [markdown]
# Now, there isn't much we can do to validate the `fare` column. We can verify
# that the most expensive ticket was in fact GBP 512.00, and some tickets were
# handed out for free, as we see with the minimum. The mean ticket price also
# makes sense, but besides these, there isn't much more we can do for the time
# being.
#
# Let's investigate the cabin column. Titanic accommodated luxurious cabins
# with utmost comfort, especially for the first-class passengers. Below is a
# cutaway diagram depicting these facilities.
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
print(df.loc[df.cabin.notna(), "cabin"].unique()[0:25])


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
df.loc[df.cabin.str.contains("T"), "name"]


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
# information from the [Titanic's deck
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
    if lst[0][0] == "F" and lst[1][0] in ["G", "E"]:
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
SECTIONS_MAP = {
    "D": ["A", "O"],
    "E": ["B", "K", "M", "Q"],
    "F": ["C", "E", "G", "H", "J", "R"],
    "G": ["D", "F", "N", "S"],
}


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
    str_list = cabin.split(" ")

    # Case where there's only one cabin number or cabin is of type
    # deck + cabin number
    if len(str_list) == 1:
        return str_list[0][0]

    # Case where there are multiple cabin numbers within the same deck
    if all([str_list[0][0] == x[0] for x in str_list]):
        return str_list[0][0]

    if len(str_list) > 1:
        # Case where subsection is included in the cabin number, eg F E69
        if (
            len(str_list[0]) == 1
            and str_list[1][0] in SECTIONS_MAP[str_list[0][0]]
        ):
            return str_list[0][0]
        # Case where there are multiple cabins each from different decks
        # We'll just assign the mode
        if len(str_list[0]) != 1:
            deck_arr = np.array([x[0] for x in str_list])
            u, c = np.unique(deck_arr, return_counts=True)
            return u[np.argmax(c)]

    return cabin


df["deck"] = df.loc[df["cabin"].notna(), "cabin"].map(extract_deck_sectionless)
df["deck"] = df["deck"].astype("category")
df.loc[df["cabin"].notna(), ["cabin", "deck"]].head(10)


# %%
print(df.loc[df["cabin"].str.contains(" "), ["cabin", "deck"]].head(10))


# %% [markdown]
# Notice our assumption that the group tickets' cabin numbers were all in the
# same deck was true for the non-missing data. We'll keep this assumption in
# mind when filling in the missing values for the cabin column.
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
print(mismatched[["passengerid", "pclass", "deck", "cabin"]])


# %% [markdown]
# There are no mismatches between passenger classes and their decks. Thus, the
# `cabin` column contains no invalid entries.
#
# Finally, we can validate the `sex` column using passenger names. To start
# with, we can extract passenger titles from the `name` column. To do that, we
# need to extract what titles are present in the dataset. What makes it easy in
# our case is that titles end with a dot at the end of the word, so we can
# filter by using that. Also, the names are consistent in formatting. Every
# name string starts with the surname, followed by the title with a dot at the
# end, and then the first and middle names follow. For example, Ward, Miss.
# Anna.


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
print(
    df.loc[
        df["name"].str.contains(".*L\\..*", regex=True), ["passengerid", "name"]
    ]
)


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
TITLES = {
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
}


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
    for word in str_list:
        if word in TITLES:
            return word

    dotted = [word for word in str_list if "." in word]

    # Every person has a title;
    if len(dotted) == 1:
        return dotted[0]

    # Every shortened name is exactly 2 chars long
    if len(dotted) > 1:
        return next(word for word in dotted if len(word) > 2)

    return None


# %%
df["title"] = df["name"].apply(extract_title)
df["title"] = df["title"].astype("category")

print(f"Missing title values count: {df.title.isna().sum()}")
df[["name", "title"]].head(10)


# %% [markdown]
# We made two assumptions when extracting titles from passenger names:
#
# 1. Every passenger has an honorific title
# 2. Every shortened name is exactly two characters long, including the dot
#
# Now, these two assumptions are needed only in the case that the name string
# does not contain any of the titles from the `TITLES` set we've defined.
#
# Now, we can cross-reference titles with their respective genders. Note that
# some of the titles in our dataset are gender-neutral. Thus, we'll only check
# for the ones that are gender-specific initially.


# %%
pd.crosstab(df["title"], df["sex"])


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
print(na_counts[na_counts > 0])


# %% [markdown]
# ### Embarked


# %% [markdown]
# We'll start with the `embarked` column. Referring back to the data dictionary
# we've provided in the notebook `Understanding & Planning`, `embarked` column
# refers to the port of embarkation, meaning where they boarded the Titanic.
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
# Considering the vast majority of first-class passengers boarded the Titanic
# from Southampton, we'll assign the `Embarked` as `S` for these passengers. In
# fact, we'll just use a frequency map of the port of embarkation for each
# passenger class.


# %%
embarked_by_class = df.groupby(by="pclass")["embarked"].value_counts()
print(embarked_by_class)


# %% [markdown]
# Notice that the proposed approach for filling missing `embarked` values is not
# good since all missing values would be imputed as `"S"` regardless of
# `pclass`. Therefore, we change our approach. Instead, we'll use a step-by-step
# process to infer the information. The first step is to use the ticket number:
# a family of passengers is most likely to board the ship together. The second
# step would be to use `fare`. However, there are some difficulties involving
# the `fare` column.
#
# In our dataset, fare is not a passenger-based feature. Instead, fare is
# calculated on a ticket basis. If a person buys two £10 tickets, their fare
# will show as £20, which shouldn't be treated differently from a passenger who
# bought the same ticket but only for himself. The difficulties we have with
# fare can be summed up as follows:
#
# 1. The training dataset may not include every member with the same ticket
# number.
#     - For example, if a family of 5 bought a 5-person ticket, but our training
#     data includes only 4 of the members, then doing a basic
#     calculation like `fare = fare / 4` for each passenger will result in an
#     incorrect ticket cost.
# 2. Some tickets could be handed out for free or with discounts, which we have
# no way of knowing.
# 3. Evidently, ticket discounts were applied based on age: adults, children,
# and babies had different ticket costs.
#     - Since we have missing age values, filtering is not possible without age
#     imputation, which will introduce noise to our approximation anyway.
#
# Thus, we'll try to resolve the `fare` column by extracting `fare_per_person`
# using two metrics: ticket numbers and family/group sizes.


# %%
ticket_counts = df.groupby("ticket")["ticket"].transform("count")
family_size = df["sibsp"] + df["parch"] + 1

df["famsize"] = family_size

group_size = np.where(
    ticket_counts > 1, np.maximum(ticket_counts, family_size), family_size
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
# "fare per person" amount; we can only approximate. In our model, we'll use
# these approximated `fare_per_person` medians to match with ports. If we also
# add the passenger class to the equation, we can further narrow the range of
# possibilities for a missing value.


# %%
def fill_embarked(df: pd.DataFrame) -> None:
    """
    Fills missing embarked values by

    1. Matching ticket numbers
    2. Comparing fare_per_person to class-specific median fares
    """
    na_idx = df[df["embarked"].isna()].index

    fare_medians = df.groupby(["embarked", "pclass"])[
        "fare_per_person"
    ].median()

    for i in na_idx:
        ticket = df.loc[i, "ticket"]
        fare = df.loc[i, "fare_per_person"]
        pclass = df.loc[i, "pclass"]

        # 1. ticket-based mode
        modes = df.loc[df["ticket"] == ticket, "embarked"].mode()
        if not modes.empty:
            df.loc[i, "embarked"] = modes.iloc[0]
            continue

        # 2. fallback: closest class-specific median fare
        best_port = None
        best_diff = float("inf")

        for port in ["C", "Q", "S"]:
            try:
                median_fare = fare_medians.loc[(port, pclass)]
            except KeyError:
                continue

            diff = abs(fare - median_fare)

            if diff < best_diff:
                best_diff = diff
                best_port = port

        df.loc[i, "embarked"] = best_port


# %%
fill_embarked(df)
print(f"Missing embarked values count: {df.embarked.isna().sum()}")


# %%
df[df["ticket"] == "113572"]


# %% [markdown]
# ### Age


# %% [markdown]
# We'll start by creating a column that indicates the `age` value was missing
# before processing the data.


# %%
df["age_was_missing"] = df.age.isna().astype(int)


# %% [markdown]
# The dataset has 177 missing age values. We shall start by checking the
# distribution of missing values across other features. If we confirm that the
# missing values occur at random, we can then proceed with imputation.


# %%
tab = pd.crosstab(df.title, df.age.isna(), normalize=True)
tab[tab[True] > 0]


# %%
pd.crosstab(df.sex, df.age.isna(), normalize=True)


# %%
pd.crosstab(df.pclass, df.age.isna(), normalize=True)


# %% [markdown]
# - The distribution of missing age values varies substantially across titles
# and passenger classes. Although we haven't conducted a statistical test, the
# data itself is clearly showcasing the variance. For example,
#
# - Master. title has 10% missing values, while Mr. has 23%
# - Only 6% of second-class passengers have missing age values compared to
# almost 28% for the third-class passengers
#
# Therefore, it is reasonable to say that the data is *NOT* missing completely
# at random. Since `title`, `sex`, and `pclass` show variance in missing age
# value proportions, it is feasible to assume that these features can reasonably
# explain the missing values. Thus, we'll proceed to impute `age` using the
# specified feature set. Before continuing, we should also plot the age
# distribution to compare with that after imputation.


# %%
sns.histplot(data=df.loc[df.age.notna()], x="age", bins=np.arange(0, 85, 5))

plt.xlabel("Age")
plt.title("Age Distribution")


# %% [markdown]
# While some honorific titles, such as Dr., are broad titles in terms of age
# ranges, some of them are highly focused on a specific range. For example,
# Master. title is given only to males who have not reached adulthood yet. On
# the other hand, Miss. title is given to young girls or unmarried women, which
# doesn't really narrow the age range. However, the `parch` feature can be used
# to narrow the range further down since unmarried women generally don't have
# children while young girls probably have parents on board. Though `parch` is
# more likely a proxy than a direct signal, since adult unmarried women could be
# traveling with their parents too.
#
# We can also use the `sibsp` feature and train a model on these features. The
# model should be able to further differentiate a young girl from an unmarried
# woman using family-related features. It should also be useful for the rest of
# the age imputation process.
#
# Before we proceed with this idea, we should also check the missing age
# proportions of the ' parch ' and ' sibsp ' features.


# %%
pd.crosstab(df.parch, df.age.isna(), normalize=True)


# %%
pd.crosstab(df.sibsp, df.age.isna(), normalize=True)


# %% [markdown]
# We observe that missingness in age values is higher for passengers with no
# other family members onboard, both for `parch` and `sibsp` features
# separately. The variance is not negligible; therefore, we assume that values
# are missing at random.
#
# Now, we shall train a model to impute the missing values. The obvious choice
# in this case would be a random forest model. The reasons are:
#
# 1. We are not sure of any linear relationship between features and the target
# `age`
# 2. Suspicions that arise regarding passenger classes, such as passengers with
# zero `parch` and `sibsp`, or third-class male passengers with Mr. title (we'll
# investigate this class of passengers more closely in a later notebook), are
# naturally handled by the model as it makes the distinctions by itself
# 3. No further need for data assumptions than what we've already stated


# %%
data = df[df.age.notna()]

categorical = ["title", "sex"]
numeric = ["pclass", "sibsp", "parch"]

y = data["age"]
X = data[["title", "sex", "pclass", "sibsp", "parch"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=40
)


# %%
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)
model = RandomForestRegressor(random_state=40, n_jobs=-1)
pipe = Pipeline([("preprocess", preprocess), ("model", model)])
pipe.fit(X_train, y_train)


# %%
y_pred = pipe.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")


# %% [markdown]
# A mean absolute error of 8.33 is not that bad in our case. For comparison,
# we'll also create a baseline model that assigns `title` medians to missing age
# values.


# %%
title_medians = data.groupby("title")["age"].median()
y_pred_baseline = X_test["title"].map(title_medians)
y_pred_baseline = y_pred_baseline.fillna(y_train.median())
print(f"Baseline MAE: {mean_absolute_error(y_test, y_pred_baseline)}")


# %% [markdown]
# The baseline model's mean absolute error is 9.31 compared to 8.33 of our
# random forest regressor. This result is satisfactory enough given that none of
# the features are direct signals for age but rather proxies that narrow our
# guess range down. Moving on, we should check and compare the age distribution
# after filling the missing values.


# %%
pipe.fit(X, y)

missing = df[df.age.isna()][["title", "sex", "pclass", "sibsp", "parch"]]

df.loc[df.age.isna(), "age"] = pipe.predict(missing)


# %%
plt.figure(figsize=(8, 6))

bins = np.arange(0, 85, 5)

sns.histplot(
    data=df,
    x="age",
    bins=bins,
    color="blue",
    alpha=0.75,
    label="After imputation",
)

sns.histplot(
    data=df.loc[df.age_was_missing == 0],
    x="age",
    bins=bins,
    color="orange",
    alpha=0.75,
    label="Before imputation (observed)",
)

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution: Before vs After Imputation")
plt.legend()
plt.show()


# %% [markdown]
# **Remarks:**
#
# 1. Age distribution is preserved for the majority of age groups
# 2. The lower and higher ends of the curve are basically identical
# 3. Age groups of 5-15 and 30-45 are smoothed
# 4. Age distribution is more concentrated around the median
# 5. Tails are less extreme
#
# Overall, we should be satisfied with the result. Even though the imputed age
# distribution is more concentrated around the median, we still have a better
# model than just imputing with title medians, shown by comparing the MAE scores
# of both models. At the same time, the distribution isn't distorted
# significantly, especially outside the $1-\sigma$ range from the median.


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
# However, the same difficulties with the `fare` column as in the `embarked`
# case apply here as well. Therefore, we'll use `fare_per_person` instead.


# %%
print(f"Missing decks before: {df.deck.isna().sum()}")


def fill_same_ticket_decks(df: pd.DataFrame):
    df["deck"] = df["deck"].fillna(
        df.groupby("ticket")["deck"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
        )
    )


fill_same_ticket_decks(df)

print(f"Missing decks after: {df.deck.isna().sum()}")


# %% [markdown]
# We've only filled 11 missing values. There are still 676 missing values.
#
# All things considered, imputing `deck` by using `pclass` and `fare_per_person`
# seems to be the most logical approach. Due to how each cabin (and therefore
# its corresponding deck) was designed to target a specific wealth group, there
# should be a natural distinction between decks. Hence, we can use a predictive
# model for imputation. In our case, a random forest classifier works the best
# since
#
# - Random forests are resistant to nonlinearity and outliers
# - Doesn't require encoding (which can introduce nonexistent ordering between
# categories)
#
# The tree classifier we'll build for imputing the decks will also be subject to
# the approximations we have done so far, such as in the "fare per person" case.
# As a result, we cannot really expect such great accuracy. On the contrary,
# with these few predictor variables and a small sample size, a higher accuracy
# would hint at extreme overfitting, data leakage, etc., rather than hinting at
# discovering relations between feature and target variables for imputation.
#
# Let's proceed with developing our model.


# %%
data = df.drop("cabin", axis=1).dropna(axis=0)
data = data[data["deck"] != "T"]

X = data[["fare_per_person", "pclass"]]
y = data["deck"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=6
)


# %%
rf = RandomForestClassifier(n_jobs=-1, random_state=6)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=6)

param_grid = {
    "n_estimators": np.arange(5, 51, 5),
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
    "max_depth": np.arange(3, 16, 2),
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2],
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
# 72.3% accuracy is sufficient for our case. We won't go much deeper into why,
# but one quick metric we could obtain is to train a base model with the
# following logic:
#
# - Predict decks A-E for `pclass == 1`
# - Predict decks D-G for `pclass == 2` or `pclass == 3`
#
# Let's also investigate the confusion matrix.


# %%
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
rfc = RandomForestClassifier(random_state=6, n_jobs=-1, **clf.best_params_)

rfc.fit(X, y)


# %% [markdown]
# Now, we can impute the missing decks for passengers who share their tickets
# first. We'll make it so that we'll only impute the first passenger
# corresponding to the duplicated ticket, then set the decks for the rest of
# the passengers with the same ticket number as the first passenger.


# %%
def impute_decks(
    model: RandomForestClassifier,
    data: pd.DataFrame,
    passenger_idx: pd.Series,
    threshold: float = 0.6,
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
    subset = passenger_idx
    X_missing = data.loc[subset, ["fare_per_person", "pclass"]]
    proba = model.predict_proba(X_missing)
    classes = model.classes_

    if not isinstance(proba, np.ndarray):
        raise TypeError(
            f"model.predict_proba returned {type(proba)} instead of np.ndarray"
        )

    max_proba = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    pred_labels = classes[pred_idx]

    na_count_before = data.deck.isna().sum()
    subset_missing = data.loc[subset, "deck"].isna()

    # Use the index of the selected subset for the Series
    subset_idx = data.loc[subset].index
    imputed_series = pd.Series(
        np.where(max_proba >= threshold, pred_labels, np.nan), index=subset_idx
    )

    # Assign only to NaN slots
    data.loc[subset, "deck"] = data.loc[subset, "deck"].where(
        ~subset_missing, imputed_series
    )

    na_count_after = data.deck.isna().sum()
    print(f"Imputed {na_count_before - na_count_after} missing values.")


# %%
impute_decks(
    model=rfc, data=df, passenger_idx=df.ticket.duplicated(keep="first")
)


# %% [markdown]
# Now, we can set the decks of the passengers with the same tickets again.


# %%
fill_same_ticket_decks(df)


# %%
df.deck.isna().sum()


# %% [markdown]
# We still have 487 missing values to deal with. For the rest of the data, we
# can lower the threshold to 0, perform a check on whether `pclass` matches the
# assigned deck, and apply the same deck to the other ticket holders as we just
# did.


# %%
impute_decks(
    model=rfc,
    data=df,
    passenger_idx=df.ticket.duplicated(keep="first"),
    threshold=0.0,
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
fill_same_ticket_decks(df)

print(f"Missing deck values count: {df.deck.isna().sum()}")


# %% [markdown]
# Now, we have 458 passengers left who do not share their ticket with anyone
# else. We'll impute these values and then check whether there's an
# inconsistency.


# %%
impute_decks(model=rfc, data=df, passenger_idx=df.deck.isna(), threshold=0.0)

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
df.to_parquet(
    "../data/modified/cleaned.parquet",
    index=False,
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
# 6. Extracted a new column `famsize` from `parch` and `sibsp`
# 7. Extracted a new column `fare_per_person`
# 8. Validated the data (except for the `fare` column, read the related section
# for more details)
# 9. Imputed 2 `embarked`, 177 `age`, and 687 `deck` values
#
# Our strategy for imputation was as follows:
#
# 1. In the `embarked` column, we first matched based on ticket numbers. If two
# passengers share the same ticket number but one of them is missing his/her
# embarkation information, we assume that they most probably boarded the ship
# together; therefore, assign the same `embarked` value as the other passenger.
# If this option is not available, then we match by fare means per port. We
# calculate the difference between the passenger's fare and each port's fare
# mean and choose the closest.
# 2. In the `age` column, we developed a random forest regressor model that is
# trained on `title`, `sex`, `pclass`, `sibsp`, and `parch` columns. The model
# performed better than the "assign each title's median age" model, with a mean
# absolute error of 8.33 compared to the median model's 9.32. We also checked
# the distribution of age values before and after imputation. We decided to
# proceed with the model.
# 3. In the `cabin` column, we followed the same principle as with the
# `embarked` case, where we filled in first by ticket number. Then, we developed
# a random forest classifier to fill in by using `fare_per_person` and `pclass`
# features. We created a function to fill missing values only when the model's
# confidence was above a certain threshold. We recursively applied both methods
# back and forth until no further progress could be achieved. Lastly, we dropped
# the model's confidence threshold to 0 so that it would fill all the values
# that couldn't have been filled with the recursive method.
#
# We saved the resulting `pd.DataFrame` object as a new dataset under the
# directory `data/modified/cleaned.parquet`, which we'll use in the upcoming
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
