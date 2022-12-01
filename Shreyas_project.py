#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
# %%
data = pd.read_csv('data.csv')

# %%
data.head()
# %%
data.columns
# %%
data.info()
# %%
data_subset = data[['year','bmi','q21','q41','q42','q43','q49','race4','race7','sex' ]]
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
data_2009=data_2009[data_2009['q21']!=' ']
data_2009=data_2009[data_2009['q41']!=' ']
data_2009=data_2009[data_2009['q42']!=' ']
data_2009=data_2009[data_2009['q49']!=' ']
data_2009=data_2009[data_2009['race4']!=' ']
data_2009=data_2009[data_2009['q43']!=' ']
data_2009=data_2009[data_2009['sex']!=' ']
data_2009=data_2009[data_2009['bmi']!=' ']

# %%
data_2009
# %%
data_2009=data_2009.reset_index()
# %%
data_2009=data_2009.drop(columns=['race7'])
data_2009=data_2009.drop(columns=['index'])


# %%
data_2009=data_2009.rename(columns={"q21": "Vape Use", "q41": "Physical Activity", "q42": "Television", "q43": "Electronic Devices", "q49": "Grades", "race4": "race"})
data_2009.head()
# %%
data_2009.to_csv('output.csv') 
# %%
data_2009.value_counts('year')
# %%
data_2009
# %%
