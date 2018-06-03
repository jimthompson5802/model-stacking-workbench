from framework.model_stacking import FeatureGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

#%%
import yaml
import os.path 
import pandas as pd
import numpy as np


#%%
#
# get parameters 
#
with open('./config.yml') as f:
    CONFIG = yaml.load(f.read())
    
print('root dir: ',CONFIG['ROOT_DIR'])

#%%
#
# feature set 1
#
fs = FeatureGenerator('raw','L0FS01')

# get raw data
fs.getRawData()

new_train = fs.raw_train_features_df
new_train.fillna(-999,inplace=True)

new_test = fs.raw_test_features_df
new_test.fillna(-999,inplace=True)

fs.saveFeatureSet(new_train, new_test)


#%%
#
# feature set 2
#
fs = FeatureGenerator('raw','L0FS02')

# get raw data
fs.getRawData()

new_train = fs.raw_train_features_df

# find only numberic attributes
numeric_predictors = [x for x in new_train.columns if new_train[x].dtype != 'O']

new_train = new_train.loc[:,numeric_predictors]
new_train.fillna(-999,inplace=True)
new_train.shape

new_test = fs.raw_test_features_df.loc[:,numeric_predictors]
new_test.fillna(-999,inplace=True)
new_test.shape

fs.saveFeatureSet(new_train, new_test)

#%%
#
# feature set 3
#
fs = FeatureGenerator('raw','L0FS03')

# get raw data
fs.getRawData()

new_train = fs.raw_train_features_df

# find only numberic attributes
numeric_predictors = [x for x in new_train.columns if new_train[x].dtype != 'O']

new_train = new_train.loc[:,numeric_predictors]

# impute mean value for missing values
imp = Imputer()
new_train = imp.fit_transform(new_train)

mms = MinMaxScaler()

# min/max scale data and convert to data frame, ensure index values match
# original data frame
new_train = pd.DataFrame(mms.fit_transform(new_train),index=fs.raw_train_id_df.index)

print(new_train.shape)

new_test = fs.raw_test_features_df.loc[:,numeric_predictors]

# impute missinvg values
new_test = imp.transform(new_test)

# Apply min/max transform and 
# convert to data frame,  ensure index values match original data frame
new_test = pd.DataFrame(mms.transform(new_test),index=fs.raw_test_id_df.index)


# save new feature set
fs.saveFeatureSet(new_train, new_test)

#%%