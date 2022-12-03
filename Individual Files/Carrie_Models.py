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


#%%
data = pd.read_csv("/Users/carriemagee/Downloads/cleaned_data2.csv")
# In[176]:
data
#%%
#recoding television from factors to numeric
data["Television"]=data["Television"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()
#%%
#recoding electronic devices from factors to numeric
data["Electronic_Devices"]=data["Electronic_Devices"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()
#%%
#recoding physical activity from factors to numeric
data["Physical_Activity"]=data["Physical_Activity"].replace([1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7])
data.head()
#%%
data["Vape_Use"]=data["Vape_Use"].replace([1,2],["Yes","No"])
data.head()
#%%
#vape_use visulization
sns.countplot(x=data["Vape_Use"],data=data).set(title="Distribution of Vape Use")
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
data.head()
#%%
#race visulization
plt.figure(figsize=(8,4))
sns.countplot(x=data["race"],data=data).set(title="Racial Makeup of Sample")
#%%
#race and vape visulization
plt.figure(figsize=(9,4))
sns.countplot(x=data["Vape_Use"],hue="race",data=data).set(title="Distribution of Vape Use by Race")
#%%
sns.boxplot(x="Vape_Use",y="Television", data=data)
#%%
sns.boxplot(x="Vape_Use",y="Electronic_Devices", data=data)
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace(["White","Black or African American","Hispanic/Latino","All Other Races"],[1,2,3,4])
data.head()
#%%
data["Vape_Use"]=data["Vape_Use"].replace([1,2],[0,1])
data.head()
#%%
#recoding data for logit regression
xdata = data[["Television","Electronic_Devices","race"]]
print(xdata.head())
ydata = data[["Vape_Use"]]
print(ydata.head())
#%%
model = glm(formula="Vape_Use ~ Television + Electronic_Devices+C(race, Treatment(reference = 1))",data=data, family=sm.families.Binomial())
model = model.fit()
print(model.summary())
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
data.head()
#%%
#recoding grades from numeric to categorical
data["Grades"]=data["Grades"].replace([1,2,3,4,5,6,7],["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"])
data.head()
#%%
data.Grades.value_counts()
#%%
#creating a contingency table for race and grades
contigency = pd.crosstab(index=data['race'], columns=data['Grades'], margins=True, margins_name="Total")
#%%
plt.figure(figsize=(9,4))
sns.heatmap(contigency, annot=True, cmap="YlGnBu")
#%%
#chi-squared test of independence
stat, p, dof, expected = chi2_contingency(contigency)
print(p)
#checking the significance
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression

admitlogit = LogisticRegression()  # instantiate
admitlogit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', admitlogit.score(x_test, y_test))
print('Logit model accuracy (with the train set):', admitlogit.score(x_train, y_train))


# %%
