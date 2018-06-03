# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer
from sklearn.neural_network import MLPClassifier as ThisModel
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
        model_params=dict(hidden_layer_sizes=(100,)), #hyper-parameters
        model_id='L0NN1',   # Model Identifier
        feature_set='L0FS03'  # feature set to use
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
