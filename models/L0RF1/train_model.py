# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer
from sklearn.ensemble import RandomForestClassifier as ThisModel
import yaml
import pickle
import os.path
import pandas as pd
import numpy as np
import functools
import pickle

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(n_estimators=20,n_jobs=-1), #hyper-parameters
        model_id='L0RF1',   # Model Identifier
        feature_set='L0FS02'  # feature set to use
        )


#%%
#
# clear out old results
#
this_model.cleanPriorResults()

#%%
#
# create features for the next level
#
this_model.createFeaturesForNextLevel()


#%%
#
# train model on all the data
#
this_model.trainModel()


#%%
#
# create Kaggle submission
#
this_model.createKaggleSubmission()

#%%