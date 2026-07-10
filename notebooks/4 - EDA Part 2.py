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
# # 4 - Exploratory Data Analysis Part 2


# %% [markdown]
# **Author:** M. Görkem Ulutürk
#
# **Date:** July, 2026


# %% [markdown]
# ## Introduction


# %% [markdown]
# In the previous EDA notebook, we analyzed our dataset. We've investigated each
# column, including the extracted features, in the univariate and bivariate
# analysis sections. In this notebook, we'll investigate interfeature
# relationships and interactions, proxies, and check our previous suspicions
# about how some columns interacted with the target `survived`. We'll start with
# the imports once again.


# %% [markdown]
# ## Imports


# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# %%
df = pd.read_parquet("../data/modified/edap1.parquet")

df.head(10)


# %%
df.info()


# %% [markdown]
# ## Multivariate Analysis


# %% [markdown]
# In previous sections, we've investigated which features were related to the
# `survived` target variable. In this section, we'll try to understand why
# they're related. For example, we've discovered that passengers with no
# siblings/spouses had significantly lower survival rates than expected.
# Similarly, we've seen the same situation with the `parch` feature.
#
# The first order of business is to decide which variables in our hands are
# independent signals and which are proxies of those signals. Recall that we've
# engineered columns
#
# - `deck`: Extracted from present `cabin` numbers and imputed using
# ticket-based matching and a confidence level-based random forest classifier
# trained on `fare_per_person` and `pclass`
# - `title`: Extracted using the `name` column
# - `famsize`: Extracted using the `sibsp`, `parch`, and `ticket` columns
# - `fare_per_person`: Extracted using the `fare` and `famsize` columns
# - `age_group`: Extracted by mapping age ranges to age groups
#
# Since we're developing a tree-based model, we won't be using the `age_group`
# column. We've used this column for EDA purposes only. On the other hand, we'll
# explore the possibility of analyzing other variables on the list above and
# possibly including them in our final model. Therefore, in the next section,
# we'll make sure they carry a signal regarding a passenger's `survived` status
# independent of the underlying extraction method and those variables used.


# %% [markdown]
# ### Determining Proxies


# %% [markdown]
# In this section, we'll develop multiple models to test for interfeature
# dependencies. Let's start by defining helper functions. We'll use logistic
# regression models and likelihood ratio tests between models for proxy testing.
# LLR test is a goodness-of-fit test for two models, one of which is more
# complex than the other. Essentially, it asks the question whether introducing
# a new feature to the model improves the fit.
#
# As per usual, we'll set $\alpha = 0.05$ and state the hypothesis of the test
# as follows:
#
# - $\text{H}_0$: The control model sufficiently explains the target (adding the
# investigated feature does not improve the fit)
# - $\text{H}_\text{a}$: The full model better explains the target (adding the
# investigated feature does improve the fit)


# %%
P_VALS = dict()


def logit_model(_formula: str, _data: pd.DataFrame, print_summary: bool = True):
    """
    Fits a logistic regression model and returns the model summary.
    """
    _model = smf.logit(_formula, _data).fit(maxiter=500)
    _odds_ratios = np.exp(_model.params)

    if print_summary:
        print(_model.summary())
        print("\n======= Odds Ratios =======\n")
        print(_odds_ratios.drop("Intercept"))

    return _model


def llr_test(
    control_model, full_model, alpha: float = 0.05, key: str | None = ""
):
    """
    Performs a likelihood ratio test
    """
    lr_stat = 2 * (full_model.llf - control_model.llf)
    df_diff = full_model.df_model - control_model.df_model
    p_lrt = stats.chi2.sf(lr_stat, df_diff)

    print("Likelihood Ratio Test")
    print("LLR Statistic: ", lr_stat)
    print("Degrees of Freedom: ", df_diff)
    print("P-Value: ", p_lrt)
    print("")

    if p_lrt < alpha:
        print(f"p-value < {alpha}: Reject the null hypothesis")
    else:
        print(f"p-value >= {alpha}: Failed to reject the null hypothesis")

    if key:
        P_VALS[key] = p_lrt


# %% [markdown]
# Our main goal is to determine whether each feature is a proxy or not. The
# other variables are extracted from already existing data, which we've shown to
# have some relationship with the target. For example, in the `deck` case, a
# feasible model in this case would be:
#
# ```python
# "survived ~ pclass + sex + age_group + deck"
# ```
#
# The problem here is that we don't know whether the deck carries significant
# information that can explain `survived` when controlled for other variables.
# The deck column itself was engineered using `fare_per_person`, which should
# correlate with `pclass` by logic. Therefore, we should instead test for:
#
# ```python
# "survived ~ pclass + sex + age_group + fare_per_person + deck"
# ```
#
# However, to determine whether the test is applicable or not, we must make sure
# that `fare_per_person` actually belongs in our control set.
#
# In part 1 of EDA, we've already established that `pclass`, `sex`, and
# `age_group` were associated with the target `survived`. Therefore, we'll be
# using these features in our base (control) models. Even though we've shown
# some other variables to be related as well, we'll still test them in the
# following chapters. There are two reasons for this decision:
#
# 1. Some features, such as `sibsp`, were used to extract features, and if the
# extracted feature is already a reliable signal, then there's no reason to
# include underlying features. For example, instead of introducing both `sibsp`
# and `parch`, combining them into a single `famsize` feature can decrease
# noise.
# 2. Intuitively, some features are not as primitive as the three listed above
# in the sense that suspecting a proxy is actually a natural observation. For
# example, passenger class is already tied very closely to how much a ticket
# costs; therefore, it is intuitive to think of `fare` carrying the passenger
# class signal.
#
# So, we'll define our base model using the three features mentioned and use it
# as a control model. We'll introduce more features to the model and test their
# performance to determine whether the newly introduced features contribute any
# information in estimating `survived`.


# %%
reduced_formula = """
survived ~
    C(pclass, Treatment(1)) +
    C(sex, Treatment('female')) +
    C(age_group, Treatment('Child'))
"""
control_model = logit_model(reduced_formula, df, print_summary=True)


# %% [markdown]
# **Remarks**
#
# - The base model takes `pclass == 1`, `sex == "female"`, and `age_group ==
# "Child"` as the references for each feature.
# - LLR p-value is less than our usual significance value of 0.05; therefore, we
# reject the null hypothesis to conclude that at least one of the variables can
# explain `survived`.
# - Looking at the individual classes' p-values and coefficients, we can state
# that compared to the base group, second- and third-class passengers, male
# passengers, and adult and middle-aged passengers had lower odds of survival
# since p-values are statistically significant and coefficients are negative.
# - For example, third-class passengers had roughly 91% lower odds of survival.
# - The other groups don't have enough statistical evidence to support the claim
# that their odds of survival were any different from the control group.


# %% [markdown]
# #### Fare


# %% [markdown]
# We'll start by investigating the `fare` feature. Recall that a passenger's
# fare is calculated based on the cumulative cost of every single ticket that
# the passenger bought. There's no distinction between tickets when they're
# bought together. If a passenger buys two tickets, one for himself and one for
# his wife, for example, then both of those tickets will have the same ticket
# number as well as the fare value. Thus, `fare` itself may not be the best
# feature to determine one's survival status. However, we've found that there is
# a statistically significant relationship anyway.


# %%
fare_full = reduced_formula + "+ fare"
fare_full_model = logit_model(fare_full, df, print_summary=True)


# %%
llr_test(control_model, fare_full_model, key="fare")


# %% [markdown]
# Computed p-value suggests that we fail to reject the null hypothesis of the
# LLR test, which was that introducing `fare` to the baseline model does not
# improve the fit. It implies that `fare` is not an independent signal but
# rather a proxy, most probably of `pclass`. A passenger's fee may be
# proportional to both passenger class (higher class tickets are more expensive)
# and how many tickets they've bought (fare is calculated based on the
# cumulative cost of all tickets a passenger has bought). We'll investigate
# these claims in depth in the feature interactions section.
#
# Hence, we'll not be including this feature in our final model.


# %% [markdown]
# #### Fare per Person


# %% [markdown]
# We'll check whether `fare_per_person` can explain `survived` when controlled
# for other variables. We've found out `fare` wasn't carrying a strong enough
# individual signal when controlled for baseline model features. However,
# `fare_per_person` was engineered using other variables as well.


# %%
fpp_full = reduced_formula + "+ fare_per_person"
fpp_full_model = logit_model(fpp_full, df, print_summary=True)


# %% [markdown]
# **Remarks**
#
# - From the summary table's LLR p-value, which is less than 0.05, we conclude
# that at least one of the coefficients is non-zero in the regression model.
# Note that this is not the LLR p-value we're using to test whether the
# investigated feature contributes to a better fit; it only tells us that at
# least one of the features in the regression model contributes to explaining
# `survived`.
# - `pclass` and `sex` can explain `survived` when other variables are
# controlled, which was expected from the first part of EDA.
# - Some age groups seem to explain the `survived` target. In particular, adults
# and middle-aged passengers had considerably lower odds of survival than
# children, 40.3% and 18.6%, respectively.


# %%
llr_test(control_model, fpp_full_model, key="fare_per_person")


# %% [markdown]
# As a result of the LLR test, we conclude that introducing `fare_per_person`
# does not improve the fit of the model. Therefore, it is safe to state that
# `fare_per_person` is a proxy, most likely carrying the `pclass` signal. In
# part 1 of EDA, we discovered that `fare_per_person` was associated with the
# target variable. And now, we've found out that the association was most likely
# due to the `pclass` signal carried over during feature engineering.
#
# Consequently, we'll not use this feature when developing our model.


# %% [markdown]
# #### Embarked


# %% [markdown]
# In our previous analysis, we discovered that passengers who embarked from
# Cherbourg survived at a significantly higher rate than expected, while
# Southampton port had the exact opposite effect. From historical data, we know
# that Cherbourg port was the choice of wealthier passengers; therefore, there's
# a good chance that `embarked` is a proxy for `pclass`.


# %%
embarked_full = reduced_formula + "+ C(embarked, Treatment('C'))"
embarked_full_model = logit_model(embarked_full, df, print_summary=True)


# %%
llr_test(control_model, embarked_full_model, key="embarked")


# %% [markdown]
# **Remarks**
#
# - LLR test indicates that `embarked` is indeed an independent signal as the
# computed p-value is statistically significant. In other words, adding
# `embarked` to the model's feature set improves the resulting fit.
# - Passengers who embarked from Southampton had significantly lower odds of
# survival, roughly 44% lower.
# - There's not enough statistical evidence to suggest that passengers embarked
# from Queenstown had different odds of survival compared to those who embarked
# from Cherbourg
#
# Hence, we'll be including `embarked` in our final model.


# %% [markdown]
# #### Deck


# %% [markdown]
# We'll apply the same procedure as in the `fare_per_person` case. Here, since
# `fare_per_person` is found to be not contributing, the base control model will
# be the same as `fpp_control_model`. Thus, instead of refitting, we'll use the
# same model.


# %%
deck_full = reduced_formula + "+ C(deck, Treatment('A'))"
deck_full_model = logit_model(deck_full, df, print_summary=True)


# %%
llr_test(control_model, deck_full_model, key="deck")


# %% [markdown]
# In the case of the `deck` feature, we've failed to reject the null hypothesis,
# concluding that when controlled for other variables (`pclass`, `sex`, and
# `age_group`), introducing `deck` does not improve the fit of the model. Thus,
# we'll not be including this feature on our model as well.


# %% [markdown]
# #### Title


# %% [markdown]
# We'll test whether the `title` column can add any explainability to the
# target. Recall that we've extracted `title` from the `name` column. We've also
# discovered statistically significant evidence of its association with
# `survived`.


# %%
title_full = reduced_formula + "+ C(title_group, Treatment('Mrs.'))"
title_full_model = logit_model(title_full, df, print_summary=True)


# %% [markdown]
# Observe that some standard errors have blown up, such as for the titles Miss.
# and Mrs. The values we need to pay attention to can be summarized in the
# following table:
#
# Class     | Key     | Coefficient | Standard Error | P-Values
# ----------|---------|-------------|----------------|---------
# Sex       | Female  | -524.1      | 7.76+06        | 1.000
# Title     | Dr.     | 521.0       | 7.76+06        | 1.000
# Title     | Master. | 524.0       | 7.76+06        | 1.000
# Title     | Mr.     | 520.1       | 7.76+06        | 1.000
# Title     | Other   | 520.7       | 7.76+06        | 1.000
#
# Even though the model is reported to have converged, the model is obviously
# not reliable. Judging by the extreme coefficients, blown-up standard errors,
# and exactly 1 p-value, we can safely state that the model exhibited
# characteristics of quasi-complete separation. In other words, `title` is a
# strong proxy of `sex`, so much so that the model could not separate the
# signals even though the algorithm converged.
#
# This is to be expected. In notebook 2, Data Wrangling, we've extracted the
# `title` column to validate `sex` itself since most of the passengers had
# gender-specific honorific titles. However, we've conducted the test regardless
# to double-check if there was something we've missed. This is a reasonable
# suspicion since `title` can also carry age group and passenger class
# information to a degree. For example, a captain, countess, lady, or doctor is
# not likely to be traveling as a third-class passenger since those people
# belong to higher social and economic classes on average.
#
# There's no need to conduct an LLR test in this case. Title is excluded from
# the test family due to the logit model demonstrating separation. It'd render
# the LLR test result unreliable; therefore, the test is omitted. We'll not be
# including `title` in our final model.


# %% [markdown]
# #### Family Size


# %% [markdown]
# We've previously discovered a relationship between a passenger's survival
# chances and the size of the family they belonged to. As a reminder, families
# of 1, 6, and 7+ had lower-than-expected survival rates, and families of 2, 3,
# and 4 had higher-than-expected odds.


# %%
famsize_full = reduced_formula + "+ C(famsize_group, Treatment(1))"
famsize_full_model = logit_model(famsize_full, df, print_summary=True)


# %%
llr_test(control_model, famsize_full_model, key="famsize")


# %% [markdown]
# **Remarks**
#
# - LLR test p-value is less than 0.05 so we reject the null hypothesis
#   - Family size is a contributing factor to a passenger's survival even after
#   accounting for `pclass`, `sex`, and `age_group`
#   - In other words, when `famsize` is introduced to the model, the logistic
#   regression model fits better to the data at hand
# - Regarding individual p-values of family size groups, we can conclude that:
#   - Families of 2, 3, and 4 have no different chances at survival than solo
#   travelers, understood by the p-values of those groups being bigger than our
#   significance level 0.05
#   - Families larger than 4 have considerably lower chances of survival
#   - Families of 5 roughly had 96% less odds at survival compared to solo
#   travelers
#   - Odds at survival are roughly 93% and 92% lower for families of 6 and 7+,
#   respecively
#
# One note here is that when we directly use `famsize` column, the model not
# only fails to converge but it also exceeds maximum iterations of 500 and tends
# to infinity. The most likely cause would be similar to problems we've had when
# we conducted hypothesis tests on features versus target. Too few observations
# within a particular family size may result in unstable coefficients, or
# singular matrices during calculations. On the other hand, binning categories
# with fewer observations creates large-enough groups that the problem can
# disappear, which did in our case.
#
# This was the reason we've created binned columns in the first place: to
# overcome unstability and satisfy data assumptions. However, a tree-based model
# can make these distinctions internally. It learns thresholds such as `famsize
# <= 4.5` on its own.
#
# Since we've shown that `famsize_group` carries the `survived` signal, and a
# tree-based model can create bins internally, substituting the raw, integer
# `famsize` feature for `famsize_group` is the wiser choice, especially since a
# tree-based model does not need to satisfy the same data assumptions that lead
# to the creation of the `famsize_group` feature.
#
# In summary, we'll include `famsize` in our final model for its contribution to
# the model's fit and convenience for a tree-based model.


# %% [markdown]
# #### Test Validation


# %% [markdown]
# We've conducted five likelihood ratio tests, all with a significance level set
# to 0.05. Since we've conducted all tests on the premise of "base model versus
# full model" to see if there was any improvement, we ran into the issue of
# accumulating odds of errors.
#
# Similar to the procedure we've applied in the previous EDA notebook, we'll
# apply the Holm-Bonferroni method to correct the family-wise error rate.


# %%
def holm_bonferroni(p_vals: dict) -> pd.DataFrame:

    tests, pvalues = zip(*p_vals.items(), strict=True)
    sort_index = np.argsort(pvalues)

    tests = [tests[x] for x in sort_index]
    pvalues = [pvalues[x] for x in sort_index]

    reject, pvals_corrected, _, _ = multipletests(
        pvalues, alpha=0.05, method="holm", is_sorted=True
    )

    results = pd.DataFrame(
        {
            "Test": tests,
            "p-value": pvalues,
            "Adjusted p-value": pvals_corrected,
            "Reject": reject,
        }
    )

    return results


# %%
holm_bonferroni(P_VALS)


# %% [markdown]
# **Remarks**
#
# - After correcting for error, we still reject the null hypothesis for the
# `famsize` case and fail to reject it in other cases.
# - The only change in conclusion is for `embarked`, where the LLR test resulted
# in rejection of the null hypothesis; however, the corrected p-value indicates
# a failure to reject.
#
# Although we've seen a relationship between `embarked` and `survived`, and the
# LLR test confirmed that `embarked` improved the model's fit, the adjusted
# p-value says otherwise.
#
# As a result of the test validations we've conducted, we can reinstate our
# decision to include `embarked` in our final model. We'll not be using this
# feature.
#
# In summary, here's the feature set we'll be using after analyzing for proxies:
#
# 1. `pclass`
# 2. `sex`
# 3. `age_group`: `age` itself was shown to not have enough statistical evidence
# to have any explanatory power over `survived`, but `age_group` does
# 4. `famsize`
#
# After we conclude the feature interactions section, we may add more features
# to the model. Thus, this is not a conclusive list, just a short summary of the
# decisions we've taken so far.
