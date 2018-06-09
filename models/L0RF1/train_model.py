# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.ensemble import RandomForestClassifier as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(n_estimators=20,n_jobs=-1), #hyper-parameters
        test_prediction_method='k-fold_average_model',
        model_id='L0RF1',   # Model Identifier
        feature_set='KFS02'  # feature set to use
        )

model_tracker = ModelPerformanceTracker(model_trainer=this_model)
#%%
#
# clear out old results
#
this_model.cleanPriorResults()

#%%
#
# train model on all the data
#
this_model.trainModel()

#%%
# create Test predictions
this_model.createTestPredictions()

#%%
#
# create Kaggle submission
#
this_model.createKaggleSubmission()

#%%
#
# record model performance metrics
#
model_tracker.recordModelPerformance()

#%%