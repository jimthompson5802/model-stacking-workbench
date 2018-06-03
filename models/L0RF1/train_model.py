# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer
from sklearn.ensemble import RandomForestClassifier as MODEL
import yaml
import pickle
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

train_ds = 'train.csv'
test_df = 'test.csv'
feature_set = 'L0FS01'
model_id = 'L0RF1'
model_params = dict(n_estimators=20,n_jobs=4)

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


X1 = train_df.loc[:,predictors]
y1 = train_df[CONFIG['TARGET_VAR']]



for fold in k_folds[:1]:
    train_idx = fold[0]
    X = train_df.iloc[train_idx,:]
    y = train_df[CONFIG['TARGET_VAR']].iloc[train_idx]
    
    model = MODEL(**model_params)
    
    model.fit(X,y)
    


#%%
model = MODEL(**model_params)
print(model.get_params())


