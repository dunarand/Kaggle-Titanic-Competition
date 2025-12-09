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
# ---

# %% [markdown]
# # 1 - Understanding & Planning

# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** December, 2025

# %% [markdown]
# ## The Problem <a name="1"></a>

# %% [markdown]
# Regardless of the workflow a data scientist choses for a project, it should
# start with a question. Of course, there'll be many questions asked and
# answered along the way; however, the first question should spark the
# curiosity within and guide us through the project: the ultimate question
# we're trying to answer with this project. Thus, we state our question:
#
# > Did people survive the Titanic incident out of pure luck, or were social
# constructs resulted in certain groups having more chances at survival?
#
# Remember, one of the reasons why that many people died in this incident was
# because of the lack of lifeboats. Therefore, people on board had to make
# certain choices; some sacrifices had been made and some people were saved.
# But we wonder whether a person's traits; such as, age, wealth, gender, etc.
# played a role in their survival, and if so, which groups were more likely to
# survive.

# %% [markdown]
# ## The Data

# %% [markdown]
# With this project, we've been handed out three datasets:
#
# 1. `train.csv`: Contains the passenger information for training
# 2. `test.csv`: Similar to `train.csv`, used for testing
# 3. `gender_submission.csv`: Example submission data
#
# We'll not need the third dataset until the submission phase. Let's now get
# to understand the data.

# %% [markdown]
# ### Data Dictionary


# %% [markdown]
# Markdown
