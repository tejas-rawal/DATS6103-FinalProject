#!/usr/bin/env python
# coding: utf-8

# In[157]:


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


# In[175]:


data = pd.read_csv("/Users/carriemagee/Downloads/cleaned_data2.csv")


# In[176]:


data


# In[177]:


#recoding television from factors to numeric
data["Television"]=data["Television"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()


# In[178]:


#recoding electronic devices from factors to numeric
data["Electronic_Devices"]=data["Electronic_Devices"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()


# In[179]:


#recoding physical activity from factors to numeric
data["Physical_Activity"]=data["Physical_Activity"].replace([1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7])
data.head()


# In[180]:


#recoding physical activity from factors to numeric
data["Vape_Use"]=data["Vape_Use"].replace([1,2],[0,1])
data.head()


# In[181]:


#recoding data for logit regression
xdata = data[["bmi","Physical_Activity","Television","Electronic_Devices","Grades","race","sex"]]
print(xdata.head())
ydata = data[["Vape_Use"]]
print(ydata.head())


# In[182]:


model = glm(formula="Vape_Use ~ Television + Electronic_Devices+C(race, Treatment(reference = 1))",data=data, family=sm.families.Binomial())
model = model.fit()
print(model.summary())


# In[172]:


#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
data.head()


# In[156]:


#recoding grades from numeric to categorical
data["Grades"]=data["Grades"].replace([1,2,3,4,5,6,7],["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"])
data.head()


# In[77]:


data.Grades.value_counts()


# In[78]:


#creating a contingency table for race and grades
contigency = pd.crosstab(index=data['race'], columns=data['Grades'], margins=True, margins_name="Total")


# In[98]:


sns.heatmap(contigency, annot=True, cmap="YlGnBu")


# In[80]:


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


# In[ ]:




