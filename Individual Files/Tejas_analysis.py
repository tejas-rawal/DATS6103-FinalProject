#%%
import pandas as pd
import numpy as np

# one-way ANOVA
from scipy.stats import f_oneway

import seaborn as sns
import matplotlib.pyplot as plt

# stats models
import statsmodels.api as sm
from statsmodels.formula.api import mnlogit
from statsmodels.formula.api import glm

#%%
# load the data
surveyDf = pd.read_csv('../Datasets/cleaned_data5.csv')
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
phys_answers = ['0', '1', '2', '3' , '4', '5', '6', '7']
race_groups = ['White', 'Black or African American', 'Hispanic/Latino', 'All Other Races']
sex = ['Female', 'Male']
grades = ["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"]

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
# To begin, we can analyze the value counts for each answer to the questions we will be utilizing in our models

#%%
# Television use
sns.countplot(y=surveyDf.Television, color='#1B065E')
plt.yticks(list(range(len(tv_answers))), tv_answers)
plt.xlabel('Count')
plt.ylabel('Hours of TV watched')
plt.title('Counts for hours of television watched responses')
plt.show()

# Physical Activity
sns.countplot(y=surveyDf.Physical_Activity, color='#34E5FF')
plt.yticks(list(range(len(phys_answers))), phys_answers)
plt.xlabel('Count')
plt.ylabel('Days physically active')
plt.title('Counts for days physically active responses')
plt.show()

# Electronic device use
sns.countplot(y=surveyDf.Electronic_Devices, color='#3E8989')
plt.xlabel('Count')
plt.ylabel('Hours of electronic device use')
plt.title('Counts for hours of electronic device usage')
plt.show()

# Race
sns.countplot(y=surveyDf.race, color='#6B0F1A',
    order=[1, 3, 2, 4])
plt.yticks(list(range(len(race_groups))), race_groups,  rotation=45)
plt.xlabel('Count')
plt.ylabel('Race')
plt.title('Counts of race responses in survey population')
plt.show()

# Sex
sns.countplot(y=surveyDf.sex, color='#F49E4C')
plt.yticks(list(range(len(sex))), sex)
plt.xlabel('Count')
plt.ylabel('Sex')
plt.title('Counts of each sex in survey population')
plt.show()

#%%[markdown]
# For the televisons hours watched survey questions, most participants responded that they watched no TV on an average school day.
# <br/><br/>
# For the question addressing days of physcical activity within a typical school week, a majority of participants responded that they were active for at least 60 minutes on all 7 days of the week.
# <br/><br/>
# In our survey population, the majority of respondents identified as White.
# <br/><br/>
# There is a near 50-50 split of each sex (male, female) in the survey population.

#%%[markdown]
# #### BMI across hours spent watching TV 
# <br/><br/>
# Let us start by examining the distribution of BMI across answers for the television question:

#%%
# TODO: BMI distribution plot
sns.distplot(surveyDf.bmi, color="#60D394", bins=40,
    hist_kws=dict(edgecolor="#000000", linewidth=1),
    kde_kws=dict(linewidth=2, color="#313715"))
plt.xlabel('BMI (kg/in²)')
plt.ylabel('Density')
plt.title('Density Plot of Survey Population BMI')
plt.show()

#%%[markdown]
# The distribution of BMI within our population seems failry normal, with a slight right-skewness. This can be expalined by respondendts with unusually high BMIs shifting the distribution.

#%%
# violin plot of BMI distribution across hours of TV watched answers
sns.violinplot(y=surveyDf.bmi, x=surveyDf.Television, alpha=0.6, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=surveyDf.bmi, x=surveyDf.Television, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
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

# #### BMI across electronic device usage
# # Examining the distribution of BMI across hours of electronic device usage within survey population

#%%
# violin plot
sns.violinplot(y=surveyDf.bmi, x=surveyDf.Electronic_Devices, alpha=0.6, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=surveyDf.bmi, x=surveyDf.Electronic_Devices, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.show()

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_device = get_unique(surveyDf, 'Electronic_Devices')
samples_by_device = [
    surveyDf[surveyDf.Television == answer]['bmi']
        for answer in unique_by_device
]


print("Total size: ", len(samples_by_device))
print("Size of each group: ", [len(sample) for sample in samples_by_device])

device_anova_result = f_oneway(*samples_by_device)
print("TV ANOVA result:\n", device_anova_result)
#%%[markdown]
# 
# #### BMI across hours spent exercising
# This time, we will examine the distribution of BMI across answers for the physical activity question.
# <br/><br/>
# We can begin by plotting the BMI distribution across answer choices:
#%%
sns.violinplot(y=surveyDf.bmi, x=surveyDf.Physical_Activity, alpha=0.6, palette='husl')
plt.title('BMI by hours of days with physical activity')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=surveyDf.bmi, x=surveyDf.Physical_Activity, palette='husl')
plt.title('BMI by days with physical activity')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_phys = get_unique(surveyDf, 'Physical_Activity')
samples_by_phys = [
    surveyDf[surveyDf.Physical_Activity == answer]['bmi']
        for answer in unique_by_tv
]

print("Total size: ", len(samples_by_phys))
print("Size of each group: ", [len(sample) for sample in samples_by_phys])

phy_anova_result = f_oneway(*samples_by_phys)
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
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=surveyDf.bmi, x=surveyDf.race, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
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
# #### BMI by sex
# This time, we will examine the distribution of BMI by the sex of the participant.
# <br/><br/>
# We can begin by plotting the BMI distribution across gender choices:
#%%
sns.violinplot(y=surveyDf.bmi, x=surveyDf.sex, alpha=0.6, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=surveyDf.bmi, x=surveyDf.sex, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

#%%[markdown]
# The hypothesis setup for this test looks as follows:
# <br/><br/>
# * Hٖ₀ = The mean BMIs for each sex are equal
# * Hₐ = The mean BMIs is significantly different between each sex
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_sex = get_unique(surveyDf, 'sex')
samples_by_sex = [
    surveyDf[surveyDf.sex == sex]['bmi']
        for sex in unique_by_sex
]

print("Total size: ", len(samples_by_sex))
print("Size of each group: ", [len(sample) for sample in samples_by_sex])

sex_anova_result = f_oneway(*samples_by_sex)
print("Sex ANOVA result:\n", sex_anova_result)

#%%[markdown]
# With a p-value of 0.00013, we can reject our Hٖ₀ that the mean BMI of participants of each sex are equal. Our result indicates that the BMI is significantly different between female and male participants.

#%%[markdown]
# ### Race classification
# Physical and educational outcomes can be determined by race.
# <br/><br/>
# ANOVA results targeting BMI across the 4 race groups were significant. Can we find other features in the dataset that could help us classify the race of respondendts?

# <br/><br/>

# Plot and desribe counts of television hours and physically activity respones by race

#%%
# reclassify race column
surveyDf.race = surveyDf.race.replace(to_replace=[1, 2, 3, 4], value=race_groups)

# race proportions grouped by each survey response
tv_by_race = surveyDf.groupby('Television')['race']\
    .value_counts(normalize=True).mul(100)\
    .rename('percent')\
    .reset_index()

# tv_by_race pivot table
tv_by_race = tv_by_race.pivot(index='Television', columns='race', values='percent')

# stacked bar chart
tv_by_race.plot(kind='bar', stacked=True)
plt.xticks(list(range(len(tv_answers))), tv_answers, rotation=0)
plt.xlabel('Hours of TV watched')
plt.ylabel('Percentage')
plt.title('Hours of TV watched by race')
plt.legend(title='Race', bbox_to_anchor=(1, 1))
plt.show()

#%%[markdown]
# Talk about analysis of chart

#%%
# physical activity race proportions
phys_by_race = surveyDf.groupby('Physical_Activity')['race']\
    .value_counts(normalize=True).mul(100)\
    .rename('percent')\
    .reset_index()

# phys_by_race pivot table
phys_by_race = phys_by_race.pivot(index='Physical_Activity', columns='race', values='percent')

# stacked bar chart of physical activity by race
phys_by_race.plot(kind='bar', stacked=True)
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.xlabel('Days physically active')
plt.ylabel('Percentage')
plt.title('Days physically active by by race')
plt.legend(title='Race', bbox_to_anchor=(1, 1))
plt.show()

#%%[markdown]
# Discuss results

#%%
# electronic device usage by race
el_by_race = surveyDf.groupby('Electronic_Devices')['race']\
    .value_counts(normalize=True).mul(100)\
    .rename('percent')\
    .reset_index()

# el_by_race pivot table
el_by_race = el_by_race.pivot(index='Electronic_Devices', columns='race', values='percent')

# stacked bar chart of electronic usage by race
el_by_race.plot(kind='bar', stacked=True)
plt.xticks(rotation=0)
plt.xlabel('Hours of electronic use')
plt.ylabel('Percentage')
plt.title('Hour of electronic use by by race')
plt.legend(title='Race', bbox_to_anchor=(1, 1))
plt.show()

#%%[markdown]
# Discuss results

#%%
# grades by race
grades_by_race = surveyDf.groupby('Grades')['race']\
    .value_counts(normalize=True).mul(100)\
    .rename('percent')\
    .reset_index()

# grades_by_race pivot table
grades_by_race = grades_by_race.pivot(index='Grades', columns='race', values='percent')

# stacked bar chart of grades by race
grades_by_race.plot(kind='bar', stacked=True)
plt.xticks(list(range(len(grades))), grades, rotation=45)
plt.xlabel('Grades')
plt.ylabel('Percentage')
plt.title('Grades by by race')
plt.legend(title='Race', bbox_to_anchor=(1, 1))
plt.show()

#%%[markdown]
# Discuss results

#%%[markdown]
# ### Linear model: Correlation of BMI with TV + Electronic usage

# 1. Data cleanup
# 2. Transform columns
# 3. Correlation matrix
# 4. Model
# 5. Model analysis

# %%
# code to add random number to column
# transform television column data using method below
# adds a number between 0.0 and 1.0 to each value in column
def transform_tv_value(hours):
    if hours == 0.0:
        return hours
    
    if hours < 5.0:
        return hours +  np.random.uniform()
    
    return hours + np.random.uniform(high=3.0)

surveyDf.Television = surveyDf.Television.apply(transform_tv_value)

# first 10 rows
surveyDf.Television[0:10]

# BMI vs TV hours watch scatter plot

# %%
def transform_ec_value(hours):
    if hours == 0.0:
        return hours
    
    if hours < 1.0:
        return hours +  np.random.uniform(high=0.4)
    
    if hours < 5.0:
        return hours +  np.random.uniform()
    
    return hours + np.random.uniform(high=3.0)

surveyDf.Electronic_Devices = surveyDf.Electronic_Devices.apply(transform_ec_value)

# first 10 rows
surveyDf.Electronic_Devices[0:10]
# %%
surveyDf.plot(x='Television', y='bmi', kind='scatter')
surveyDf.plot(x='Electronic_Devices', y='bmi', kind='scatter')

# %%
from statsmodels.formula.api import ols
model = ols(formula='bmi ~ Television + Electronic_Devices + Physical_Activity + C(race)', data=surveyDf)

#%%
modelFit = model.fit()
print( type(modelFit) )
print( modelFit.summary() )

#%%[markdown]
# Only at 0.002 R^2 fit. Not a good model