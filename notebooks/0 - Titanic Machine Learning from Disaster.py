# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Titanic - Machine Learning from Disaster

# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** December, 2025

# %% [markdown]
# ## Introduction <a name="1"></a>
#
# Titanic is one of the most widely known deadly incidents in history.
#
# > RMS Titanic was a British ocean liner that sank in the early hours of
# 15 April 1912, as a result of striking an iceberg on her maiden voyage from
# Southampton, England, to New York City, United States. Of the estimated 2,224
# passengers and crew aboard, approximately 1,500 died (estimates vary), making
# the incident one of the deadliest peacetime sinkings of a single ship.
# > (Wikipedia contributors, 2025)
#
# The Kaggle competition page states the challenge as follows:
#
# > On April 15, 1912, during her maiden voyage, the widely considered
# “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately,
# there weren’t enough lifeboats for everyone onboard, resulting in the death
# of 1502 out of 2224 passengers and crew.
# >
# > While there was some element of luck involved in surviving, it seems some
# groups of people were more likely to survive than others.
# >
# > In this challenge, we ask you to build a predictive model that answers the
# question: “What sorts of people were more likely to survive?” using passenger
# data (i.e., name, age, gender, socio-economic class, etc).

# %% [markdown]
# ## Project Outline

# %% [markdown]
# ### Objective

# %% [markdown]
# We will develop a machine learning model to predict whether a passenger
# survived or not based on features such as `Age`, `Sex`, or socio-economic
# class. The dataset, together with the variables just stated (and others), will
# be studied at later stages using descriptive statistics, visualizations, and
# statistical analyses.
#
# One important note here is the purpose. Why do we develop such a project?
# The main reason in my case is to implement what I've learnt and learn
# more along the way, practice my skills, and to simulate a real-life work
# experience. Of course, the project is not completely useless in terms of
# real-life applications. The model could be used to study population dynamics
# under emergency conditions, as an example. However, there are many talented
# and experienced researchers who worked on the same data many times, so the
# reader should take this project only as a reflection of my individual
# capacities and workflow.

# %% [markdown]
# ### The Model and Model Evaluation
#
# We'll decide on the model and the performance metrics while we develop
# further into the project. We'll have a better idea about the model choice
# at the understanding and planning phase. Our initial decision on the
# performance would be to optimize for F1-score. However, since our data
# science workflow is an iterative process, everything regarding the project
# plan is subject to change as insights we reveal could alter our perspective
# and reveal that certain choices would be more beneficial. At this stage,
# we can only make logical decisions based on shallow and brief inspection
# of the data, which we'll carry out in the upcoming notebook.
#
# As for evaluation, we'll use model validation, various performance metrics
# (depends on the selected model), and visualizations, such as a confusion
# matrix. We'll assess the model's performance and decide on whether it is
# satisfactory or not.
#
# The project's performance goal is at around 85% accuracy and F1-score. We'll
# talk about these performance targets in the next notebook, the planning
# phase.

# %% [markdown]
# ### Structure

# %% [markdown]
# We'll divide the project into separate notebooks for each chapter. These
# chapters are as follows:
#
# 1. [Understanding & Planning](1%20-%20Understanding%20&%20Planning.ipynb)
# 2. [Data Wrangling](2%20-%20Data%20Wrangling.ipynb)
# 3. [Exploratory Data Analysis](3%20-%20Exploratory%20Data%20Analysis.ipynb)
# 4. [Building a Model](4%20-%20Building%20a%20Model.ipynb)
# 5. [Results](5%20-%20Results.ipynb)
#
# Stages of the data science workflow are divided into their own notebooks.
# Also, since this process is iterative, we may include sections in those
# notebooks that require the reader to jump between notebooks. The notebooks
# aren't conducted only as a document for presenting the findings but also to
# document my thought-process and workflow, how I obtained and used information
# throughout the project, and so on.

# %% [markdown]
# ## Next Up

# %% [markdown]
# In the upcoming notebook, we start by understanding the problem. We briefly
# explore the data to have an understanding and make educated decisions on
# the project's development.
