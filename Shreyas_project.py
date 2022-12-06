#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
# %%
data = pd.read_csv('/Users/carriemagee/Downloads/sadc_data.csv')

# %%
data.head()
# %%
data.columns
# %%
data.info()
# %%
data_subset = data[['year','bmi','q34','q78','q79','q80','q89','race4','race7','sex',"stweight","qn45","qn24"]]
data_subset.head()
# %%
data_2009 = data_subset[data_subset['year']>=2009]
# %%
data_2009.head()
# %%
data_2009.tail()
# %%
data_2009.info()
# %%
data_2009 = data_2009.replace(' ', float('NaN'), regex = True)  # Replace blanks by NaN

#%%
fig=plt.gcf()
fig.set_size_inches(10,10)
fig=sns.heatmap(data_2009.corr(),annot=True,linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
plt.savefig('output_1.png')

#%%
missing_df = data_2009.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots(figsize=(6,10))
rects = ax.barh(ind,missing_df.missing_count.values,color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.ticklabel_format(useOffset=False, style='plain', axis='x')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


#%%
data_2009.isnull().sum()

#%%
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
# One of the best notebooks on getting started with a ML problem.

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

#%%
city_day_missing= missing_values_table(data_2009)
city_day_missing

#%%
import missingno as msno #install missingno using pip install missingno
msno.bar(data_2009)


#%%
data_2009.info()

#%%
data_2009.head()

#%%
data_2009 = data_2009.drop('race7', axis=1)

#%%
data_2009.dropna(axis=0,inplace=True)
data_2009.head()

#%%

# displaying the datatypes
print(data_2009.dtypes)
  
# converting 'Field_2' and 'Field_3' from float to int
data_2009['q34'] = data_2009['q34'].apply(np.int64)
data_2009['q78'] = data_2009['q78'].apply(np.int64)
data_2009['q79'] = data_2009['q79'].apply(np.int64)
data_2009['q80'] = data_2009['q80'].apply(np.int64)
data_2009['q89'] = data_2009['q89'].apply(np.int64)
data_2009['race4'] = data_2009['race4'].apply(np.int64)
data_2009['sex'] = data_2009['sex'].apply(np.int64)
data_2009['qn45'] = data_2009['qn45'].apply(np.int64)
data_2009['qn24'] = data_2009['qn24'].apply(np.int64)
# displaying the datatypes
print(data_2009.dtypes)

#%%

data_2009=data_2009.rename(columns={"q34": "Vape_Use", "q78": "Physical_Activity", "q79": "Television", "q80": "Electronic_Devices", "q89": "Grades", "race4": "race", "stweight":"weight","qn45":"marijuana_use","qn24":"cyber_bullied"})
data_2009=data_2009.reset_index()
data_2009
#%%
#recoding television from factors to numeric
data_2009["Television"]=data_2009["Television"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data_2009.head()
#recoding electronic devices from factors to numeric
data_2009["Electronic_Devices"]=data_2009["Electronic_Devices"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()
#recoding physical activity from factors to numeric
data_2009["Physical_Activity"]=data_2009["Physical_Activity"].replace([1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7])
data_2009.head()
#%%
import missingno as msno
msno.bar(data_2009)

# %%
data_2009.to_csv('cleaned_data5.csv') 
# %%
