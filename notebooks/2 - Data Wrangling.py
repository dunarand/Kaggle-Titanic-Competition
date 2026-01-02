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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2 - Data Wrangling

# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** December, 2025

# %% [markdown]
# ## Introduction

# %% [markdown]
# In the previous section of the project, we conducted initial data
# wrangling. We've discovered that the data contains no duplicates, but we have
# missing values that we need to deal with.
#
# Let's start by importing the modules and the data.

# %% [markdown]
# ## Imports

# %%
import sqlite3

import pandas as pd

conn = sqlite3.connect(":memory:")

df = pd.read_csv("../data/raw/train.csv", encoding="utf-8")
df.head()

# %% [markdown]
# ## Data Cleaning & Validation

# %% [markdown]
# Recall that we've already checked for duplicates in the understanding and
# planning phase, and we've found no duplicates. Let's start by converting
# the column names to lowercase.

# %%
df.columns = df.columns.str.lower()

# %% [markdown]
# Let's also validate data types.

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
#
#
#
# Let's also validate the `cabin` column. During the understanding phase, we've
# discovered that this column includes quite a lot of missing values. Before we
# attempt to fill those values in, let's validate the existing ones.


# %% [markdown]
# Titanic accommodated luxurious cabins with utmost comfort, especially for
# the first-class passengers. Below is a cutaway diagram depicting these
# facilities.
#
# <center>
# <div>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Olympic_%26_Titanic_cutaway_diagram.png/960px-Olympic_%26_Titanic_cutaway_diagram.png" alt="Titanic cutaway diagram" height="500">
# </div>
# </center>
#
# > "The accomodation for first-class passengers is placed amidships and
# extends over five decks, the promenade (A), bridge (B), shelter (C), saloon
# (D), and upper (E) decks. *(Titanica, 2025)*"
#
# First-class accommodations were placed in cabins A through E, second and
# third-classes in D through G.
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
#   - Cabin numbers start with a capital English letter
#   - Some passengers accommodated multiple cabins
#       - This is likely due to cabin numbers being shared between passengers
#       groups such as families (similar to the ticket number)
#   - Some passengers have cabin numbers starting with F, followed by another
#   cabin string, eg, F G73
#   - There exists a "T" cabin
#   - Some passengers don't have full cabin numbers but just the deck letter
#
# Firstly, let's investigate the cabin T.

# %%
df.loc[df.cabin.str.contains("T") == True, "name"]

# %% [markdown]
# There's only 1 passenger with this cabin number, Mr. Stephen Weart Blackwell.
# According to Titanica, Mr. Blackwell was the only passenger with the cabin
# number T[1], and this number referred to the boat deck[2]. Thus, this is a
# legitimate entry.
#
# Now, the entries like "F G73" are not mistakes either. The reason is that
# this cabin number denotes the deck F, section G, cabin 73. We infer this
# information from the Titanic's plans themselves[3].
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
# F-J did not contain any rooms; some sections that had rooms are still not
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
# - "Mr.": for men, regardless of marital status, who do not have another
# professional or academic title
# - "Mrs.": for married women who do not have another professional or academic
# title
# - "Miss.": for girls, unmarried women, and (in the United Kingdom) married
# women who continue to use their maiden name
# - "Master.": for boys and young men, or as a style for the heir to a Scottish
# peerage
# - "Don.": commonly used in Spain, Portugal, and Italy, it is an honorific
# prefix derived from the Latin Dominus, meaning "lord" or "owner"
# - "Rev.": used generally for members of the Christian clergy, regardless of
# affiliation, but especially in Catholic and Protestant denominations, for
# both men and women
# - "Dr.": for the holder of a doctoral degree in many countries, and for
# medical practitioners, dentists, and veterinary surgeons
# - "Mme.": the French abbreviation for Madame, for women, a term of general
# respect or flattery, originally used only for a woman of rank or authority
# - "Ms.": for women, regardless of marital status or when marital status is
# unknown
# - "Major.": a military title
# - "Lady.": for female peers with the rank of baroness, viscountess, countess,
# and marchioness, or the wives of men who hold the equivalent titles
# - "Sir.": for men, formally, if they have a British knighthood or if they are
# a baronet
# - "Mlle.": is a French courtesy title traditionally given to an unmarried
# woman
# - "Col.": a military title
# - "Capt.": a military title or a ship's highest responsible officer acting on
# behalf of the ship's owner
# - "Countess.": a woman of high social rank, or the wife of a count or earl
# - "Jonkheer.": is literally translated as 'young lord' in Dutch
#
# (Paraphrased or directly quoted from the references [4], [5], [6], [7], [8],
# [9])
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

print(df["title"].isna().sum())
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
# We see that the gender-specific titles all match with a passenger's sex. On
# the other hand, we cannot methodically check for gender-neutral honorifics
# and that the only way to verify those would be to individually research the
# passengers with these titles. This verification is out of the scope of this
# project and will not be needed for our purposes. Therefore, we deem that
# we've verified the `sex` column using the passenger titles we've extracted
# from their names.


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
# the first being Queenstown, Ireland, and the second being Cherbourg, France.
# This route was popular among British ocean liners for the Southmapton-New
# York route.
# >
# > *"Titanic's maiden voyage was intended to be the first of many
# trans-Atlantic crossings between Southampton and New York via Cherbourg and
# Queenstown on westbound runs."*
# >
# > (*Wikipedia contributors, 2025*).
#
# Let's first see the passengers with missing boarding locations and then try
# to make some connections for filling these missing values.

# %%
df[df["embarked"].isna()]

# %% [markdown]
# Both passengers have first class tickets for cabin "B28" and they share the
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
print(na_age_titles)

# %%
df.loc[df.title.isin(na_age_titles), "title"].value_counts()

# %% [markdown]
# We see that the titles with missing age values contain nonempty entries as
# well. Therefore, we can use groupings to impute the missing age entries. For
# the majority of the cases, we can just use the medians to fill the missing
# values. However, we should handle some cases manually. For example, Miss.
# title includes girls and unmarried woman and first class passengers often
# traveled with the family maids who were often unmarried women. Therefore,
# we'll determine some cases to handle first.
#
# For titles Dr. and Master., we can just fill in by using title medians. Let's
# start with that first.

# %%
dr_median = df.loc[df.title == "Dr.", "age"].median()
df.loc[(df.title == "Dr.") & (df.age.isna()), "age"] = dr_median

master_median = df.loc[df.title == "Master.", "age"].median()
df.loc[(df.title == "Master.") & (df.age.isna()), "age"] = master_median

# %% [markdown]
# Next up, let's manually fill missing the young girl ages. We can filter by
# title Miss. and `parch`. Recall that `parch` denotes the number of parents
# or children onboard. Since Miss. title is exclusive to unmarried women and
# having children without marriage was not common at all at the time, we can
# assume that `parch` > 0 implies they have parents, which means that they are
# girls or young women.

# %%
mask = (df.title == "Miss.") & (df.parch > 0)
young_girl_median = df.loc[mask, "age"].median()
df.loc[mask & (df.age.isna()), "age"] = young_girl_median

# %% [markdown]
# For the rest of the passengers, we can impute values with title and passenger
# class medians. The reason we include passenger class is simple. First-class
# passengers were generally wealthy businessmen, who are generally middle-aged
# or older, and third-class passengers were usually workers and immigrants, who
# are younger.

# %%
df["age"] = df["age"].fillna(
    df.groupby(by=["title", "pclass"])["age"].transform("median")
)
df.age.isna().sum()

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
# 2. Manually fill as much data as possible (some missing values from
# the dataset can be found by research)
# 3. Automation or web scraping for filling as much data as possible.
#
# </div>
#
# Titanic is has been a research area for many years and through the collective
# efforts of many brilliant minds, a lot of the missing data has been
# recovered. Therefore, we can fill the missing passenger ages in case they
# are known but not present in our dataset. This kind of improvement is out of
# our scope though.
