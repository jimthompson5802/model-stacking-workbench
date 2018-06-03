from framework.model_stacking import FeatureGenerator

#%%
import yaml
import os.path 
import pandas as pd


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
fs1 = FeatureGenerator(CONFIG['ROOT_DIR'],'raw','L0FS01')

# get raw data
fs1.getRawData()

new_train = fs1.raw_train_features_df
new_train.fillna(-999,inplace=True)

new_test = fs1.raw_test_features_df
new_test.fillna(-999,inplace=True)

#%%
fs1.saveFeatureSet(new_train, new_test)


