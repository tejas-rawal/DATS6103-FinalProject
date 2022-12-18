#-----------------------------------IMPORT LIBTATIES--------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Sklearn processing
from sklearn.preprocessing import MinMaxScaler

# Sklearn classification algorithms
from sklearn.tree import DecisionTreeClassifier

# Sklearn classification model evaluation function
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import math
from sklearn import metrics


#--------------------------------INPUT FILE-------------------------------------#
input_path=r"C:\Users\ADMIN\Desktop\masters\GW\intro to data mining"
df=pd.read_csv(r"%s\cleaned_data5.csv"%input_path)
df["Grades"]=df["Grades"].replace([1,2,3,4,5,6,7],["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"])
df["marijuana_use"]=df["marijuana_use"].replace([1,2],[1,0])
#--------------------------Grades BY Television-------------------------------#
x, y ="Television", "Grades"

df1=df.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of Television on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    


#--------------------------Grades BY Electronic_Devices-------------------------------#
x, y ="Electronic_Devices", "Grades"

df1=df.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of Electronic Devices on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    


#--------------------------Grades BY marijuana_use-------------------------------#
x, y ="marijuana_use", "Grades"

df1=df.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of marijuana_use on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    

# #--------------------------Grades BY use of technical devices-------------------------------#
#--------------------Synthetic Minority Oversampling Technique (SMOTE)---------------------#

X=df.drop(columns=['index','year','bmi','Vape_Use','Physical_Activity','Grades','sex','weight','cyber_bullied'])
#dependent_variable
y=df[['Grades']]

os=SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

columns=X_train.columns
smote_data_X, smote_data_y=os.fit_resample(X_train, y_train)
smote_data_X=pd.DataFrame(data=smote_data_X, columns=columns)
smote_data_y=pd.DataFrame(data=smote_data_y, columns=['Grades'])

'''
Now, as not all the feature will be contributing towards the prediction, 
we need to first figure out the most important features that will contribute towards the prediction. 
This is very crucial to improve the efficiency of the model.
'''
feature_names=smote_data_X.columns.to_list()
model=LogisticRegression(random_state=0).fit(df[feature_names].values, df['Grades'].values)
#Get the score
score=model.score(df[feature_names].values, df['Grades'].values)
print("Logistic Regression score is:",score)


w0=model.intercept_[0]
w=model.coef_[0]

feature_importance=pd.DataFrame(feature_names, columns=['feature'])
feature_importance['importance']=pow(math.e,w)
feature_importance=feature_importance.sort_values(by=['importance'], ascending=False)
feature_importance=feature_importance[:10].sort_values(by=['importance'], ascending=False)

ax=feature_importance[:10].sort_values(by=['importance'], ascending=True).plot.barh(x="feature", y="importance")
plt.title('Important Feature')
plt.savefig('feature.png')
# print(feature_importance)

#Electronic Devices,Television, Marijuana Use,Race – Decision Tree Classifier
#MAke a list of all feature

feature_importance_list=feature_importance['feature'].to_list()

X=smote_data_X[feature_importance_list]
y=smote_data_y['Grades']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=0)


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

# Check the model performance with the training data
predictions_dt = model_dt.predict(X_train)
print("DecisionTreeClassifier", accuracy_score(y_train, predictions_dt))

predictions_dt = model_dt.predict(X_test)
print("DecisionTreeClassifier TEST: ", accuracy_score(y_test, predictions_dt))

#Confusion Matrix
conf_matrix =confusion_matrix(y_test, predictions_dt)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["A","B","C","D","F","NG","NS"]); 
ax.yaxis.set_ticklabels(["A","B","C","D","F","NG","NS"]);