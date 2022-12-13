#%%[markdown]
# # Introduction
# For our final project, we analyzed data collected by the CDC’s Youth Risk Behavior Surveillance System. This is a survey that is conducted every 2 years by states and local school districts across the country to collect information about adolescent tendencies as it relates to their physical and educational outcomes.
#
# We hypothesize that technology use may have the potential to impede on the adolescents' well-being and lead to adverse effects on their physical health and educational performance. Furthermore, we beleive risky behaviors such as drug use can also have detrimental effects.

# # SMART Questions
# 1. How does technology and drug use relate to positive health and academic outcomes in adolescents?
# 2. Do adolescents of various races differ in their physical health and academic success?

# # Variables of Interest
# - Physical activity
#       - The amount of days with the past week that respondent was active for at least 60 minutes
# - Television use
#       - Hours of TV watched on an average school day
# - Electronics use
#       - Hours of non-school related electronic device usage (computers, smartphones, video games) on an average school day
# - Marijuana use
#       - Has smoked marijuana
# - Vape use
#       - Has used an electronic vapor product
# - Grades
#       - Description of overall grades
# - Race
#       - Race/ethnicity they classified as
# - Body Mass Index
#       - Calculated using height and weight

#%%
# package imports
import pandas as pd
import numpy as np

# one-way ANOVA
from scipy.stats import f_oneway

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# The cleaned datasets can be found here: https://github.com/tejas-rawal/DATS6103-FinalProject/tree/main/Datasets
# load the data
data = pd.read_csv('Datasets/cleaned_data5.csv')
# analyze structure
print(f"Shape (rows, columns): {data.shape}")
print("\n\n")
print("Dataframe info:\n")
print(data.info())
print("\n\n")
print("Dataframe - first 5 rows\n")
print(data.head())

#%%[markdown]
# # Adolescent behaviors and BMI
# The goal of this analysis is to determine if there are differences between the mean BMI across race, sex, and responses for: days of physical activity, hours of watching TV, hours of electronic device usage.

#%%
# response labels
tv_answers = ['0', '< 1', '1', '2', '3', '4', '>= 5']
phys_answers = ['0', '1', '2', '3' , '4', '5', '6', '7']
race_groups = ['White', 'Black or African American', 'Hispanic/Latino', 'All Other Races']
sex = ['Female', 'Male']
grades = ["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"]

# helper functions
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
# To begin, we analyze the value counts of responses to the behaviors and labels we will utilize in our models

#%%
# Television use
sns.countplot(y=data.Television, color='#1B065E')
plt.yticks(list(range(len(tv_answers))), tv_answers)
plt.xlabel('Count')
plt.ylabel('Hours of TV watched')
plt.title('Counts for hours of television watched responses')
plt.show()

# Physical Activity
sns.countplot(y=data.Physical_Activity, color='#34E5FF')
plt.yticks(list(range(len(phys_answers))), phys_answers)
plt.xlabel('Count')
plt.ylabel('Days physically active')
plt.title('Counts for days physically active responses')
plt.show()

# Electronic device use
sns.countplot(y=data.Electronic_Devices, color='#3E8989')
plt.xlabel('Count')
plt.ylabel('Hours of electronic device use')
plt.title('Counts for hours of electronic device usage')
plt.show()

# Race
sns.countplot(y=data.race, color='#6B0F1A',
    order=[1, 3, 2, 4])
plt.yticks(list(range(len(race_groups))), race_groups,  rotation=45)
plt.xlabel('Count')
plt.ylabel('Race')
plt.title('Counts of race responses in survey population')
plt.show()

# Sex
sns.countplot(y=data.sex, color='#F49E4C')
plt.yticks(list(range(len(sex))), sex)
plt.xlabel('Count')
plt.ylabel('Sex')
plt.title('Counts of each sex in survey population')
plt.show()

#%%[markdown]
# For the televisons hours watched survey questions, most participants responded that they watched no TV on an average school day.
#
# For the question addressing days of physcical activity within a typical school week, a majority of participants responded that they were active for at least 60 minutes on all 7 days of the week.
#
# In our survey population, the majority of respondents identified as White.
#
# There is a near 50-50 split of each sex (male, female) in the survey population.

#%%[markdown]
# #### BMI Distribution
# Let's start by examining the distribution of our target variable
#%%
sns.distplot(data.bmi, color="#60D394", bins=40,
    hist_kws=dict(edgecolor="#000000", linewidth=1),
    kde_kws=dict(linewidth=2, color="#313715"))
plt.xlabel('BMI (kg/in²)')
plt.ylabel('Density')
plt.title('Density Plot of Survey Population BMI')
plt.show()

#%%[markdown]
# The distribution of BMI within our population seems failry normal, with a slight right-skewness. This can be expalined by respondendts with unusually high BMIs shifting the distribution.

# #### BMI across hours spent watching TV
# Distribution of BMI across responses to television watching

#%%
# violin plot of BMI distribution across hours of TV watched answers
sns.violinplot(y=data.bmi, x=data.Television, alpha=0.6, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Television, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for hours of television watched

# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_tv = get_unique(data, 'Television')
samples_by_tv = [
    data[data.Television == answer]['bmi']
        for answer in unique_by_tv
]


print("Number of samples: ", len(samples_by_tv))
print("Size of each sample: ", [len(sample) for sample in samples_by_tv])

tv_anova_result = f_oneway(*samples_by_tv)
print("TV ANOVA result:\n", tv_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across TV hours watched survey answeres are equal. Our result indicates that the BMI is significantly different between adolescents who watch TV for differing amounts of time on an average school day.

#%%[markdown]
# #### BMI across electronic device usage
# Distribution of BMI across hours of electronic device usage

#%%
# violin plot
sns.violinplot(y=data.bmi, x=data.Electronic_Devices, alpha=0.6, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Electronic_Devices, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.show()

#%%[markdown]
# #### One-way ANOVA test for electronic device usage
# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_device = get_unique(data, 'Electronic_Devices')
samples_by_device = [
    data[data.Television == answer]['bmi']
        for answer in unique_by_device
]

print("Number of samples: ", len(samples_by_device))
print("Size of each sample: ", [len(sample) for sample in samples_by_device])

device_anova_result = f_oneway(*samples_by_device)
print("Electronic device ANOVA result:\n", device_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across adolescent electronic device usage are equal. Our result indicates that the BMI is significantly different between adolescents who use electronic devices for varying amounts of time. 

#%%[markdown]
# #### BMI across days of physical activity
# Distribution of BMI across responses to the physical activity question.

#%%
sns.violinplot(y=data.bmi, x=data.Physical_Activity, alpha=0.6, palette='husl')
plt.title('BMI by days physical active')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Physical_Activity, palette='husl')
plt.title('BMI by days physical active')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for physical activity
# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_phys = get_unique(data, 'Physical_Activity')
samples_by_phys = [
    data[data.Physical_Activity == answer]['bmi']
        for answer in unique_by_tv
]

print("Number of samples: ", len(samples_by_phys))
print("Size of each sample: ", [len(sample) for sample in samples_by_phys])

phy_anova_result = f_oneway(*samples_by_phys)
print("Physical activity ANOVA result:\n", phy_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across physical activity survey answers are equal. Our result indicates that the BMI is significantly different between adolescents who are active for different amount of days within the past week.

#%%[markdown]
# #### BMI across race
# Distribution of BMI by race

#%%
sns.violinplot(y=data.bmi, x=data.race, alpha=0.6, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.race, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for race 
# The hypothesis setup for this test looks as follows:
#
# - Hٖ₀ = The mean BMI for each race is equal
# - Hₐ = The mean BMI is significantly different across race
# - alpha = 0.5
#%%
# code for ANOVA here
unique_by_race = get_unique(data, 'race')
samples_by_race = [
    data[data.race == answer]['bmi']
        for answer in unique_by_race
]

print("Number of samples: ", len(samples_by_race))
print("Size of each sample: ", [len(sample) for sample in samples_by_race])

race_anova_result = f_oneway(*samples_by_race)
print("Race ANOVA result:\n", race_anova_result)

#%%[markdown]
# Our results again yield a significant result. With a p-value close to 0, we must reject Hٖ₀ that the mean BMI across race are equal. Our result indicates that the BMI is significantly different between adolescents belonging to different race groups.

#%%[markdown]
# #### BMI by sex
# Distribution of BMI by the sex of the participant.
#%%
sns.violinplot(y=data.bmi, x=data.sex, alpha=0.6, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.sex, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for sex
# The hypothesis setup for this test looks as follows:
# 
# - Hٖ₀ = The mean BMIs for each sex are equal
# - Hₐ = The mean BMIs is significantly different between each sex
# - alpha = 0.5
#%%
# code for ANOVA here
unique_by_sex = get_unique(data, 'sex')
samples_by_sex = [
    data[data.sex == sex]['bmi']
        for sex in unique_by_sex
]

print("Number of samples: ", len(samples_by_sex))
print("Size of each sample: ", [len(sample) for sample in samples_by_sex])

sex_anova_result = f_oneway(*samples_by_sex)
print("Sex ANOVA result:\n", sex_anova_result)

#%%[markdown]
# With a p-value of 0.00013, we can reject our Hٖ₀ that the mean BMI for adolescents of each sex is equal. Our result indicates that the BMI is significantly different between female and male adolescents in the survey population.

#%%[markdown]
# # Effect of behaviors on physical outcomes (Shreyas)

#%%[markdown]
# # Effect of behaviors and race on academic outcomes (Rajeev)

#%%[markdown]
# # Adolescent behaviors and vape use classification (Carrie)

#%%[markdown]
# # Conclusion
# We received statistically significant results from performing ANOVA and Chi-squared tests agains the relationships between our variables of interest.
# This allowed us to continue our pursuit of studying the effects of different behaviors on health and education outcomes for adolescent youth by developing classification models.
#
# The models we built to classify health outcomes and grades did not return a significantly high accuracy. We believe this was due to weak correlation amongst our predictors.
#
# The classification models we built to predict adolescent vape use did result in a significantly high accuracy.
# 
# Some of our challenges were:
# - Cleaning a relatively large dataset with many missing values for responses from earlier years.
# - We had a large number of behaviors to choose from which can be a good and bad problem.
# 
# 
# Overall, modeling adolescent social behaviors and outcomes is a complicated task. We could potentially include responses from other behavioral questions in the survey to improve the accuracy of our model, but that will require deeper analysis of each new variable we decide to use.
