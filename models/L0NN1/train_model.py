# -*- coding: utf-8 -*-
#%%
from framework.model_stacking import ModelTrainer, ModelPerformanceTracker
from sklearn.neural_network import MLPClassifier as ThisModel

#%%
#
# Set up model for training
#
this_model = ModelTrainer(
        ModelClass=ThisModel,  #Model algorithm
        model_params=dict(hidden_layer_sizes=(100,)), #hyper-parameters
        test_data_method='Type1',
        model_id='L0NN1',   # Model Identifier
        feature_set='KFS03'  # feature set to use
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