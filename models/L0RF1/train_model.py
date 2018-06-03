# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer
from sklearn.ensemble import RandomForestClassifier as MODEL
import yaml
import pickle
import os.path
import pandas as pd
import numpy as np
import functools
import pickle
#%%
#
# get parameters 
#
with open('./config.yml') as f:
    CONFIG = yaml.load(f.read())
    
print('root dir: ',CONFIG['ROOT_DIR'])

train_ds = 'train.csv'
test_ds = 'test.csv'
feature_set = 'L0FS02'
model_id = 'L0RF1'
model_params = dict(n_estimators=20,n_jobs=-1)

#%%
#
# retrieve KFold specifiction
#
with open(os.path.join(CONFIG['ROOT_DIR'],'data/fold_specification.pkl'),'rb') as f:
    k_folds = pickle.load(f)
    
    
#%%
#
# generate features for next level
#

# retrieve training data
train_df = pd.read_csv(os.path.join(CONFIG['ROOT_DIR'],'data',feature_set,train_ds))

predictors = sorted(list(set(train_df.columns) - set(CONFIG['ID_VAR']) - set([CONFIG['TARGET_VAR']])))


#
# create features for next level using the hold out set
#
next_level = []
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
    
    # set up predictors and target for hold out set
    X_holdout = train_df.iloc[holdout_idx,:]
    id_holdout = X_holdout.loc[:,CONFIG['ID_VAR']]
    X_holdout = X_holdout.loc[:,predictors]
    y_holdout = train_df[CONFIG['TARGET_VAR']].iloc[holdout_idx]

    # geneate features for next level
    y_hat = pd.DataFrame(model.predict_proba(X_holdout),index=id_holdout.index)
    y_hat.columns = [model_id+'_'+str(col) for col in y_hat.columns]
    y_hat = id_holdout.join(y_holdout).join(y_hat)
    
    next_level.append(y_hat)
    

#%%
#
# combine the generated features into single dataframe & save to disk
#
pd.concat(next_level).sort_values(CONFIG['ID_VAR'])\
    .to_csv(os.path.join(CONFIG['ROOT_DIR'],'models',model_id,model_id+'_features.csv'),
            index=False)



#%%
#
# train model on complete training data set
#
# retrieve training data
train_df = pd.read_csv(os.path.join(CONFIG['ROOT_DIR'],'data',feature_set,train_ds))
predictors = sorted(list(set(train_df.columns) - set(CONFIG['ID_VAR']) - set([CONFIG['TARGET_VAR']])))


X_train = train_df[predictors]
y_train = train_df[CONFIG['TARGET_VAR']]
    
model = MODEL(**model_params)
    
model.fit(X_train,y_train)

with open(os.path.join(CONFIG['ROOT_DIR'],'models',model_id,model_id+'_model.pkl'),'wb') as f:
    pickle.dump(model,f)
    
    
#%%
#
# create Kaggle Submission
#
with open(os.path.join(CONFIG['ROOT_DIR'],'models',model_id,model_id+'_model.pkl'),'rb') as f:
    model = pickle.load(f)
    
# create data set to make predictions
test_df = pd.read_csv(os.path.join(CONFIG['ROOT_DIR'],'data',feature_set,test_ds))
predictors = sorted(list(set(test_df.columns) - set(CONFIG['ID_VAR']) - set([CONFIG['TARGET_VAR']])))

test_id = test_df[CONFIG['ID_VAR']]

predictions = pd.DataFrame(model.predict_proba(test_df[predictors]),index=test_df.index)
predictions.columns = [model_id+'_'+str(x) for x in list(predictions.columns)]

kaggle_submission_headers = CONFIG['KAGGLE_SUBMISSION_HEADERS']

submission = test_id.join(predictions[model_id+'_1'])
submission.columns = kaggle_submission_headers

submission.to_csv(os.path.join(CONFIG['ROOT_DIR'],'models',model_id,
                               model_id+'_submission.csv'),index=False)

