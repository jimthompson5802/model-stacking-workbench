from framework.model_stacking import FeatureGenerator

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
fs = FeatureGenerator(CONFIG['ROOT_DIR'],'raw','L0FS01')

# get raw data
fs.getRawData()

new_train = fs.raw_train_features_df
new_train.fillna(-999,inplace=True)

new_test = fs.raw_test_features_df
new_test.fillna(-999,inplace=True)

#%%
fs.saveFeatureSet(new_train, new_test)


#%%
#
# feature set 2
#
fs = FeatureGenerator(CONFIG['ROOT_DIR'],'raw','L0FS02')

# get raw data
fs.getRawData()

new_train = fs.raw_train_features_df

#%%
# find only numberic attributes
numeric_predictors = [x for x in new_train.columns if new_train[x].dtype != 'O']
numeric_predictors
#%%
new_train = new_train.loc[:,numeric_predictors]
new_train.fillna(-999,inplace=True)
new_train.shape


#%%
new_test = fs.raw_test_features_df.loc[:,numeric_predictors]
new_test.fillna(-999,inplace=True)
new_test.shape

#%%
fs.saveFeatureSet(new_train, new_test)

