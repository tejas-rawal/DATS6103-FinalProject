#!/usr/bin/env python
# coding: utf-8

#%%


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.formula.api import mnlogit
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import scipy.stats as stats


#%%
data = pd.read_csv("/Users/carriemagee/Downloads/DATS6103-FinalProject/cleaned_data5.csv")
# In[176]:
data
#%%
data["Vape_Use"]=data["Vape_Use"].replace([1,2],["Yes","No"])
data.head()
vape_yes = data[data["Vape_Use"]=="Yes"]
vape_no = data[data["Vape_Use"]=="No"]
#%%
#vape_use visulization
ax = sns.countplot(x=data["Vape_Use"],data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Vape Use" , size = 12 )
plt.ylabel( "Frequency" , size = 12 )
plt.title("Distribution of Vape Use")
data["Vape_Use"].value_counts()
## This figure shows the proportion of individuals in the sample who do and do not use electronic vapor products.
## More specifically, there are 17,102 individuals who do not engage in vaping and 14,482 who do engage in vaping which makes about a 2,620 person difference. 

#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
data.head()
#%%
#race visulization
plt.figure(figsize=(8,4))
ax = sns.countplot(x=data["race"],data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Race" , size = 12 )
plt.ylabel( "Frequency" , size = 12 ) 
plt.title("Racial Makeup of Sample")
data["race"].value_counts()
## The figure above shows the racial makeup of our sample. About 49% of the sample is White , 28% are Hispanic or Latino,  12% are Black or African American, and 11% identify as another race.
#%%
#race and vape visulization
plt.figure(figsize=(9,4))
ax = sns.countplot(x=data["Vape_Use"],hue="race",data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Vape Use" , size = 12 )
plt.ylabel( "Frequency" , size = 12 ) 
plt.title("Distribution of Vape Use by Race")
#In relation to race and vaping habits, there is a pretty similar distribution between races in terms of individuals that vape and do not vape. 
#%%
ax = sns.countplot(x="Television",hue="Vape_Use", data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bar Chart of Hours Watching Television per Day by Vape Use")
plt.ylabel( "Frequency" , size = 12 )
plt.xlabel( "Time Watching Television (Hours)" , size = 12 )
stats.ttest_ind(a=vape_yes["Television"], b=vape_no["Television"], equal_var=True)
## The plot shows the differences in hours of television watched per day by individuals who do and do not vape. Interestingly, between the housrs of 0.0 and 2.0 there are many more individuals who report not vaping. 
## In comparison, between the hours of 3.0 and 5.0 it is apparent that a greater proportion of individuals report vaping. It is important to the pattern that
## the more hours of television watched in the day, the more the individuals report vaping in comparison to not vaping. These results may imply a relationship between number of hours of television per day and vaping habits considering that the gap between those who vape and those who do not vape becomes
## smaller and smaller with every extra hour of television watched per day. 

## In addition, after running a two-sample t-test between those who and who do not vape, the results indicate that there is a significant different in the average number hours of television watch per day between groups (p<0.05).
#%%
ax = sns.countplot(x="Electronic_Devices",hue="Vape_Use", data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bar Chart of Hours on Electronic Devices per Day by Vape Use")
plt.ylabel( "Frequency" , size = 12 )
plt.xlabel( "Time on Electronic Devices (Hours)" , size = 12 )
stats.ttest_ind(a=vape_yes["Electronic_Devices"], b=vape_no["Electronic_Devices"], equal_var=True)

## The plot shows mixed results with the two ends of the hour distribution having the smallest differences between those who do and do not report vaping.
## The largest difference between groups clusters at 2.0 hours. Overall, there is no definite trend in this graph depicting differences in time spent
## on electronic devices per day between vaping and non-vaping individuals. This conclusion is furter supported by the insignicant (p>0.05) t-test which indicates that
## there is no significant difference in the average number of hours spent on electronic devices between the vaping and non-vaping groups. 
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace(["White","Black or African American","Hispanic/Latino","All Other Races"],[1,2,3,4])
data.head()
#%%
data["Vape_Use"]=data["Vape_Use"].replace(["No","Yes"],[0,1])
data.head()
#%%
data["marijuana_use"]=data["marijuana_use"].replace([1,2],[1,0])
data.head()
#%%
data["cyber_bullied"]=data["cyber_bullied"].replace([1,2],[1,0])
data.head()
#%%
#recoding data for logit regression
xdata = data[["Television","Electronic_Devices","race"]]
ydata = data[["Vape_Use"]]
#%%
model = glm(formula="Vape_Use ~ Television + Electronic_Devices+C(race, Treatment(reference = 1))",data=data, family=sm.families.Binomial())
model = model.fit()
print(model.summary())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()  # instantiate
logit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logit.score(x_test, y_test))
print('Logit model accuracy (with the train set):',logit.score(x_train, y_train))
#%%
#recoding data for logit regression
xdata1 = data[["Television","Electronic_Devices","race"]]
ydata1 = data[["marijuana_use"]]

model1 = glm(formula="marijuana_use ~ Television + Electronic_Devices+C(race, Treatment(reference = 1))",data=data, family=sm.families.Binomial())
model1 = model1.fit()
print(model1.summary())

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(xdata1, ydata1, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logit1 = LogisticRegression()  # instantiate
logit1.fit(x_train1, y_train1)
print('Logit model accuracy (with the test set):', logit1.score(x_test1, y_test1))
print('Logit model accuracy (with the train set):', logit1.score(x_train1, y_train1))

#%%
xdata2 = data[["Television","Electronic_Devices","race"]]
ydata2 = data[["cyber_bullied"]]

model2 = glm(formula="cyber_bullied ~ Television + Electronic_Devices+C(race, Treatment(reference = 1))",data=data, family=sm.families.Binomial())
model2 = model2.fit()
print(model2.summary())

from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(xdata2, ydata2, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logit2 = LogisticRegression()  # instantiate
logit2.fit(x_train2, y_train2)
print('Logit model accuracy (with the test set):', logit2.score(x_test2, y_test2))
print('Logit model accuracy (with the train set):', logit2.score(x_train2, y_train2))
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
data.head()
#%%
#recoding grades from numeric to categorical
data["Grades"]=data["Grades"].replace([1,2,3,4,5,6,7],["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"])
data.head()
#%%
#creating a contingency table for race and grades
contigency = pd.crosstab(index=data['race'], columns=data['Grades'], margins=True, margins_name="Total")
#%%
plt.figure(figsize=(9,4))
sns.heatmap(contigency, annot=True, cmap="Blues", vmin= 40, vmax=36000,fmt='g')
plt.title("Contingency Table of Race and Grades")

## The contingency table between racial groups and their grades reveals that a majority individuals, regardless of race, report
## having mostly A's and B's for their grades. A majority of white individuals and individuals of other races have mostly A's
## while a majority of Black/African American and Hispanic/Latino students report having mostly B's. 
#chi-squared test of independence
stat, p, dof, expected = chi2_contingency(contigency)
#checking the significance
alpha = 0.05
print("The results of the chi-squared test of independence showed that the p value is " + str(p) + " which indicates a significant dependent relationship between race and grades.")
#if p <= alpha:
    #print('Dependent (reject H0)')
#else:
    #print('Independent (H0 holds true)')

# %%
