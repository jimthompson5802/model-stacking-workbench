# -*- coding: utf-8 -*-


#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

MAX_TOP_LEVELS = 3
#%%
#generate test data
number_rows = 1000
col1 = np.random.choice(list('abcde'),number_rows,p=[0.01,0.09, 0.3,0.3,0.3])
col2 = np.random.choice(list('xyz'),number_rows,p=[0.1,0.8,0.1])
df = pd.DataFrame(dict(col1=col1,col2=col2))
df.head()

#%%
cat_predictors=['col1','col2']
encoded_list = []

for c in cat_predictors:

    # get one column of categorical variables
    this_cat = df[c]
    
    # determine number of unique levels
    number_of_levels = len(this_cat.unique())
    
    # handle situation where number of unique levels exceed threshold
    if number_of_levels > MAX_TOP_LEVELS:
        counts_by_level = this_cat.value_counts()
        
        # get level values for those not in the top ranks
        low_count_levels = counts_by_level.index[MAX_TOP_LEVELS:]
    
        # eliminate NULL value if present
        levels_to_other = [x for x in low_count_levels if len(x)>0]
        
        # set less frequent levels to special valid value 
        idx = [x in set(levels_to_other)for x in this_cat]
        this_cat.loc[idx] = '__OTHER__'
            
    # impute special value for any missing values
    idx = [len(x) == 0 for x in this_cat]
    this_cat.loc[idx] = '__N/A__'
    
    # now hot-one encode categorical variable
    lb = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    
    temp = lb.fit_transform(this_cat)
    temp = temp.reshape(-1,1)
    temp = ohe.fit_transform(temp)
    
    #generate column names for one-hot encoding
    column_names = [this_cat.name + '.' + x for x in lb.classes_]
    encoded_list.append(pd.DataFrame(temp,columns=column_names))

#%% 
# flatten out into single dataframe
df2 = pd.concat(encoded_list,axis=1)

