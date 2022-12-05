#%%
import pandas as pd
import numpy as np

# one-way ANOVA
from scipy.stats import f_oneway

import seaborn as sns
import matplotlib.pyplot as plt

# sci-kit learn 
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

# stats models
import statsmodels.api as sm
from statsmodels.formula.api import mnlogit
from statsmodels.formula.api import glm

#%%
# load the data
surveyDf = pd.read_csv('../Datasets/cleaned_data2.csv')
# analyze structure
print(f"Shape (rows, columns): {surveyDf.shape}")
print("\n\n")
print("Dataframe info:\n")
print(surveyDf.info())
print("\n\n")
print("Dataframe - first 5 rows\n")
print(surveyDf.head())

#%%[markdown]
# ### ANOVA analysis on BMI

# The goal of this analysis is to determine if there are differences between the mean BMI across answers for questions asking participants about their race, amount of time spent doing physical activity, and amount of time watching TV.

#%%
# common data
tv_answers = ['0', '< 1', '1', '2', '3', '4', '>= 5']

#%%
# common functions
def get_unique(df: pd.DataFrame, column: str):
    """
    Returns all unique values from the column in the provided dataframe.
    :param: :df: the dataframe to filter
    :param: :column: string name of column
    :return:  the sorted unique elements of df or raises error if column does not belong to df

    """
    try:
        return np.unique(df[column])
    except Exception as err:
        raise(err)
#%%[markdown]
# #### BMI across hours spent watching TV 
# <br/><br/>
# Let us start by examining the distribution of BMI across answers for the television question:

#%%
# violin plot of BMI distribution across hours of TV watched answers
sns.violinplot(y=surveyDf.bmi, x=surveyDf.Television, alpha=0.6, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

# boxplot of BMI vs hours of TV

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_tv = get_unique(surveyDf, 'Television')
samples_by_tv = [
    surveyDf[surveyDf.Television == answer]['bmi']
        for answer in unique_by_tv
]


print("Total size: ", len(samples_by_tv))
print("Size of each group: ", [len(sample) for sample in samples_by_tv])

tv_anova_result = f_oneway(*samples_by_tv)
print("TV ANOVA result:\n", tv_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across television hours watched survey answeres are equal. Our result indicates that the BMI is significantly different between groups of children who watch television for differing amounts of time.
# <br/><br/>

# #### BMI across hours spent exercising
# This time, we will examine the distribution of BMI across answers for the physical activity question.
# <br/><br/>
# We can begin by plotting the BMI distribution across answer choices:
#%%
sns.violinplot(y=surveyDf.bmi, x=surveyDf.Physical_Activity, alpha=0.6, palette='husl')
plt.title('BMI by hours of days with physical activity')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_phy = get_unique(surveyDf, 'Physical_Activity')
samples_by_phy = [
    surveyDf[surveyDf.Physical_Activity == answer]['bmi']
        for answer in unique_by_tv
]


print("Total size: ", len(samples_by_phy))
print("Size of each group: ", [len(sample) for sample in samples_by_phy])

phy_anova_result = f_oneway(*samples_by_phy)
print("Physical activity ANOVA result:\n", phy_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across physical activity survey answers are equal. Our result indicates that the BMI is significantly different between groups of children who were active for differing amount of days within the past week.

# #### BMI across race
# This time, we will examine the distribution of BMI across which race the participants responded with.
# <br/><br/>
# We can begin by plotting the BMI distribution across answer choices:
#%%
sns.violinplot(y=surveyDf.bmi, x=surveyDf.race, alpha=0.6, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_race = get_unique(surveyDf, 'race')
samples_by_race = [
    surveyDf[surveyDf.race == answer]['bmi']
        for answer in unique_by_race
]


print("Total size: ", len(samples_by_race))
print("Size of each group: ", [len(sample) for sample in samples_by_race])

race_anova_result = f_oneway(*samples_by_race)
print("Race ANOVA result:\n", race_anova_result)

#%%[markdown]
# Our results again yield a significant result. With a p-value close to 0, we must reject Hٖ₀ that the mean BMI of samples across race are equal. Our result indicates that the BMI is significantly different between children belonging to different race groups.

#%%[markdown]
# TODO: Perform some sort of supervised regression (linear, logistic, kNN, Random forest)?