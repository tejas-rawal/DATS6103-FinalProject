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
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier


#%%
data = pd.read_csv("/Users/carriemagee/Downloads/DATS6103-FinalProject/Datasets/cleaned_data5.csv")
#%%
data["Vape_Use"]=data["Vape_Use"].replace([1,2],["Yes","No"])
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
## This figure shows the proportion of individuals in the sample who do and do not use electronic vapor products.
## More specifically, there are 17,102 individuals who do not engage in vaping and 14,482 who do engage in vaping which makes about a 2,620 person difference. 

#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
#%%
#race visulization
plt.figure(figsize=(8,4))
ax = sns.countplot(x=data["race"],data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Race" , size = 12 )
plt.ylabel( "Frequency" , size = 12 ) 
plt.title("Racial Makeup of Sample")
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
ts = stats.ttest_ind(a=vape_yes["Television"], b=vape_no["Television"], equal_var=True)
print("Two-Sample T-test:",ts)
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
plt.legend(loc='center right')
s = stats.ttest_ind(a=vape_yes["Electronic_Devices"], b=vape_no["Electronic_Devices"], equal_var=True)
print("Two-Sample T-test:",s)

## The plot shows mixed results with the two ends of the hour distribution having the smallest differences between those who do and do not report vaping.
## The largest difference between groups clusters at 2.0 hours. Overall, there is no definite trend in this graph depicting differences in time spent
## on electronic devices per day between vaping and non-vaping individuals. This conclusion is furter supported by the insignicant (p>0.05) t-test which indicates that
## there is no significant difference in the average number of hours spent on electronic devices between the vaping and non-vaping groups. 
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace(["White","Black or African American","Hispanic/Latino","All Other Races"],[0,1,2,3])

#%%
data["Vape_Use"]=data["Vape_Use"].replace(["No","Yes"],[0,1])
#data["Vape_Use"] = data["Vape_Use"].astype('category')
#%%
data["marijuana_use"]=data["marijuana_use"].replace([1,2],[1,0])

#%%
#recoding data for logit regression
xdata = data[["Television","Electronic_Devices",'marijuana_use',"race"]]
ydata = data[["Vape_Use"]]

features = ["Television","Electronic_Devices", 'marijuana_use','race']
print(xdata)
#%%9
model = glm(formula="Vape_Use ~ Television + Electronic_Devices + C(marijuana_use)+ C(race)",data=data, family=sm.families.Binomial())
model = model.fit()
print(model.summary())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()  # instantiate
logit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logit.score(x_test, y_test))
print('Logit model accuracy (with the train set):',logit.score(x_train, y_train))

y_pred = logit.predict(x_test)
print(y_pred)

#%%
# Classification Report
#

from sklearn.metrics import classification_report
y_true, y_pred = y_test, logit.predict(x_test)
print("Classification Report:",end='\n')
print(classification_report(y_true, y_pred))

from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",end='\n')
print(c)
#%%
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logit.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic - Logisitc Regression")
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
clf = DecisionTreeClassifier(class_weight='balanced',max_depth=3)
# Train Decision Tree Classifier
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Accuracy of Decision Tree Classifier is:",metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",end="\n")
print(cm)
import seaborn as sns
ax = sns.heatmap(cm, annot=True,fmt='g',cmap="Blues")
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted')
ax.tick_params(length=0, labeltop=True, labelbottom=False)
ax.xaxis.set_label_position('top')
ax.set_xticklabels(['Positive', 'Negative'])
ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
ax.add_patch(plt.Rectangle((0, 1), 1, 0.1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
ax.add_patch(plt.Rectangle((0, 0), -0.1, 1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
plt.tight_layout()
plt.title("Confusion Matrix")
plt.show()
#%%
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = clf.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('DecisionTree: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='DecisionTree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic - DecisionTree ")
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (7,7), dpi=800)
tree.plot_tree(clf,
              feature_names = features);

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
plt.ylabel("Race")

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
data["marijuana_use"]=data["marijuana_use"].replace([1,0],["Yes","No"])
data["Vape_Use"]=data["Vape_Use"].replace([0,1],["No","Yes"])
#creating a contingency table for race and grades
contigency1 = pd.crosstab(index=data['marijuana_use'], columns=data['Vape_Use'], margins=True, margins_name="Total")
#%%
plt.figure(figsize=(9,4))
sns.heatmap(contigency1, annot=True, cmap="Blues", vmin= 40, vmax=36000,fmt='g')
plt.title("Contingency Table of Marijuana Use and Vape Use")
plt.xlabel('Vape Use')
plt.ylabel('Marijuana Use')


stat, p, dof, expected = chi2_contingency(contigency1)
#checking the significance
alpha = 0.05
print("The results of the chi-squared test of independence showed that the p value is " + str(p) + " which indicates a significant dependent relationship between marijuana use and e-cig use.")
#if p <= alpha:
    #print('Dependent (reject H0)')
#else:
    #print('Independent (H0 holds true)')



# %%
