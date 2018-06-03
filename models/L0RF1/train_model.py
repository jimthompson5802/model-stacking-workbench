# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer
from sklearn.ensemble import RandomForestClassifier as RFC
import yaml
import pickle
import os.path
import pandas as pd
#%%
#
# get parameters 
#
with open('./config.yml') as f:
    CONFIG = yaml.load(f.read())
    
print('root dir: ',CONFIG['ROOT_DIR'])

train_ds = 'train.csv'
test_df = 'test.csv'
feature_set = 'L0FS01'
model_id = 'L0RF1'
model_parms = dict(n_estimators=200)

#%%
#
# retrieve KFold specifiction
#
with open(os.path.join(CONFIG['ROOT_DIR'],'data/fold_specification.pkl'),'rb') as f:
    k_folds = pickle.load(f)
    
    
#%%
# retrieve training data
train_df = pd.read_csv(os.path.join(CONFIG['ROOT_DIR'],'data',feature_set,train_ds))

predictors = sorted(list(set(train_df) - set(CONFIG['ID_VAR']) - set(CONFIG['TARGET_VAR'])))


X = train_df.loc[:,predictors]
y = train_df.loc[:,CONFIG['TARGET_VAR']]



for fold in k_folds:
    train_idx = fold[0]
    test_idx = fold[1]

#%%
rf = RFC(**model_parms)
print(rf.get_params())


X = 