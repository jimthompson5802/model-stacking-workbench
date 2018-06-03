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
feature_set = 'L0FS02'
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

predictors = sorted(list(set(train_df) - set(CONFIG['ID_VAR']) - set([CONFIG['TARGET_VAR']])))


X1 = train_df.loc[:,predictors]
y1 = train_df[CONFIG['TARGET_VAR']]


i = 0
for fold in k_folds:
    i += 1
    print('running fold: {:d}'.format(i))
    train_idx = fold[0]
    X_train = train_df.iloc[train_idx,:]
    X_train = X_train.loc[:,predictors]
    y_train = train_df[CONFIG['TARGET_VAR']].iloc[train_idx]
    
    model = MODEL(**model_params)
    
    model.fit(X_train,y_train)
    
    #generate feature for next level
    # get indices for hold out set
    holdout_idx = fold[1]
    X_holdout = train_df.iloc[holdout_idx,:]
    id_holdout = X_holdout.loc[:,CONFIG['ID_VAR']]
    X_holdout = X_holdout.loc[:,predictors]
    y_holdout = train_df[CONFIG['TARGET_VAR']].iloc[holdout_idx]

    #
    y_hat = pd.DataFrame(model.predict_proba(X_holdout),index=id_holdout.index)
    


#%%
model = MODEL(**model_params)
print(model.get_params())


