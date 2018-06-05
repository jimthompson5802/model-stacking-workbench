
import yaml
import pandas as pd
import os
import os.path
import shutil
import pickle
import datetime
import time
import numpy as np


from sklearn.metrics import log_loss
###
#
# function to calculate Kaggle performance metric during CV 
#
###
def calculateKaggleMetric(y=None,y_hat=None):
    return log_loss(y,y_hat)


###
#
# Class for generating feature sets
#
###
class FeatureGenerator():
    
    
    def __init__(self,
                 in_dir=None,  # directory containing input training/test data sets
                 out_dir=None, # directory to contain generated feature set
                 id_vars=['ID'],  # variables used to identify records
                 target_var='target'  # target variable name
                 ):

        self.in_dir = in_dir        
        self.out_dir = out_dir      
        self.id_vars = id_vars      
        self.target_var = target_var 
        
        #
        # get parameters 
        #
        with open('./config.yml') as f:
            self.CONFIG = yaml.load(f.read())
            
        self.root_dir = self.CONFIG['ROOT_DIR']
        
        self.makeOutputDirectory()
       
    def makeOutputDirectory(self):
        #create directory to hold feature set
        # clean out out_dir
        try:
            shutil.rmtree(os.path.join(self.root_dir,'data',self.out_dir))
        except:
            pass
        
        os.makedirs(os.path.join(self.root_dir,'data',self.out_dir))
    
    
    def getRawData(self):
        #
        # default behaviour - can be overridden for different raw data structures
        #
        # Assumes existence of train.csv and test.csv in in_dir location
        # Expected function: create raw_id_df, raw_target_df, raw_train_features_df
        # and raw_test_features_df data frames.
        #
       
        df = pd.read_csv(os.path.join(self.root_dir,'data',self.in_dir,'train.csv'))
        
        # split data into identifiers, predictors and target data frames
        self.raw_train_id_df = df.loc[:,self.id_vars]
        raw_train_target_df = df.loc[:,[self.target_var]]
        
        # isolate predictor variables
        predictors = sorted(set(df.columns) - set(self.id_vars) - set([self.target_var]))
        
        # isoloate training predictiors
        raw_train_features_df = df.loc[:,predictors]
        
        # get test data set
        df = pd.read_csv(os.path.join(self.root_dir,'data',self.in_dir,'test.csv'))
        self.raw_test_id_df = df.loc[:,self.id_vars]
        raw_test_features_df = df.loc[:,predictors]
        
        return raw_train_features_df, raw_train_target_df, raw_test_features_df
        
    
    def saveFeatureSet(self,new_train_features_df=None,
                       new_train_target_df=None,
                       new_test_features_df=None):
        #
        # default behaviour - can be overriddent for different new feature storage
        #
        # append id_vars and target_var save new_train_features_df and 
        # new_test_features_df in self.out_dir
        #
        # append id-vars and target to new feature set and save as csv
        

        self.raw_train_id_df.join(new_train_target_df)\
            .join(new_train_features_df)\
            .sort_values(self.id_vars)\
            .to_csv(os.path.join(self.root_dir,'data',self.out_dir,'train.csv'),index=False)
        
        self.raw_test_id_df\
            .join(new_test_features_df)\
            .sort_values(self.id_vars)\
            .to_csv(os.path.join(self.root_dir,'data',self.out_dir,'test.csv'),index=False)
 
###
#
#  Class for training models
#
###  
class ModelTrainer():
    
    def __init__(self,
                 ModelClass=None,  #Model Algorithm
                 model_params=None,  # Model hyper-parameters
                 model_id=None,    # model identifier
                 feature_set=None,  # feature set to use
                 train_ds='train.csv',  # feature set training data set
                 test_ds='test.csv'  # feature set test data set
                 ):   
        
        self.ModelClass = ModelClass
        self.model_params = model_params
        self.model_id = model_id
        self.feature_set = feature_set
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        #
        # get global parameters 
        #
        with open('./config.yml') as f:
            self.CONFIG = yaml.load(f.read())
            
        print('Model training starting for {} at {:%Y-%m-%d %H:%M:%S}'\
              .format(self.model_id,datetime.datetime.now()))
            
    def cleanPriorResults(self):
        try:
            os.remove(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                   self.model_id,
                                   self.model_id+'_features.csv'))
        except:
            pass
        
        
        try:    
            os.remove(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                   self.model_id,
                                   self.model_id+'_train_features.csv'))
        except:
            pass
        
        try:    
            os.remove(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                   self.model_id,
                                   self.model_id+'_test_features.csv'))
        except:
            pass
        
        try:
            os.remove(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                   self.model_id,
                                   self.model_id+'_model.pkl'))
        except:
            pass
        
        try:
            os.remove(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                   self.model_id,
                                   self.model_id+'_submission.csv'))
        except:
            pass
            
    def createFeaturesForNextLevel(self):
        
        print('Starting createFeaturesForNextLevel: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        
        #
        # retrieve KFold specifiction
        #
        with open(os.path.join(self.CONFIG['ROOT_DIR'],'data','k-fold_specification.pkl'),'rb') as f:
            k_folds = pickle.load(f)
            
         
        #
        # generate features for next level
        #
        
        # retrieve training data
        train_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],'data',
                                            self.feature_set,self.train_ds))
        
        predictors = sorted(list(set(train_df.columns) - 
                                 set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        
        
        #
        # create features for next level using the hold out set
        #
        self.cv_performance_metric = []
        
        next_level = []
        i = 0
        for fold in k_folds:
            i += 1
            print('running fold: {:d} at {:%Y-%m-%d %H:%M:%S}'.format(i,datetime.datetime.now()))
            train_idx = fold[0]
            X_train = train_df.iloc[train_idx,:]
            X_train = X_train.loc[:,predictors]
            y_train = train_df[self.CONFIG['TARGET_VAR']].iloc[train_idx]
            
            model = self.ModelClass(**self.model_params)
            
            model.fit(X_train,y_train)
            
            #generate feature for next level
            # get indices for hold out set
            holdout_idx = fold[1]
            
            # set up predictors and target for hold out set
            X_holdout = train_df.iloc[holdout_idx,:]
            id_holdout = X_holdout.loc[:,self.CONFIG['ID_VAR']]
            X_holdout = X_holdout.loc[:,predictors]
            y_holdout = train_df[self.CONFIG['TARGET_VAR']].iloc[holdout_idx]
            
            # make preduction on hold out set to calculate metric and generate
            # features for next level of stack
            y_hat = model.predict_proba(X_holdout)
            self.cv_performance_metric.append(calculateKaggleMetric(y_holdout,y_hat))
        
            # geneate features for next level
            y_hat = pd.DataFrame(y_hat,index=id_holdout.index)
            y_hat.columns = [self.model_id+'_'+str(col) for col in y_hat.columns]
            y_hat = id_holdout.join(y_holdout).join(y_hat)
            
            next_level.append(y_hat)
            
        print('Completed createFeaturesForNextLevel: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            
        #
        # combine the generated features into single dataframe & save to disk
        #
        pd.concat(next_level).sort_values(self.CONFIG['ID_VAR'])\
            .to_csv(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                 self.model_id,
                                 self.model_id+'_train_features.csv'),
                    index=False)


    def trainModel(self):
        print('Starting trainModel: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        
        #
        # train model on complete training data set
        #
        # retrieve training data
        train_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],'data',self.feature_set,self.train_ds))
        predictors = sorted(list(set(train_df.columns) - set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        
        
        X_train = train_df[predictors]
        y_train = train_df[self.CONFIG['TARGET_VAR']]
        
        self.training_rows = X_train.shape[0]
        self.training_columns = X_train.shape[1]
            
        model = self.ModelClass(**self.model_params)
        
        start_training = time.time()
        model.fit(X_train,y_train)
        self.training_time = time.time() - start_training
        
        with open(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                               self.model_id,self.model_id+'_model.pkl'),'wb') as f:
            pickle.dump(model,f)
         
        print('Completed trainModel: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    def createKaggleSubmission(self,feature_set=None,test_ds='test.csv'):
        #
        # create Kaggle Submission
        #
        # Assumes:  trained model has been saved under "`model_id`_model.pkl"
        #
        
        print('Starting createKaggleSubmission: {:%Y-%m-%d %H:%M:%S}'\
              .format(datetime.datetime.now()))
        
        with open(os.path.join(self.CONFIG['ROOT_DIR'],'models',self.model_id,
                               self.model_id+'_model.pkl'),'rb') as f:
            model = pickle.load(f)
            
        # create data set to make predictions
        test_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],'data',
                                           self.feature_set,self.test_ds))
        
        predictors = sorted(list(set(test_df.columns) - 
                                 set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        
        test_id = test_df[self.CONFIG['ID_VAR']]
        
        predictions = pd.DataFrame(model.predict_proba(test_df[predictors]),index=test_df.index)
        predictions.columns = [self.model_id+'_'+str(x) for x in list(predictions.columns)]
        
        # save test predictions for next level
        pred_df = test_id.join(predictions).sort_values(self.CONFIG['ID_VAR'])
        pred_df.to_csv(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                    self.model_id,
                                    self.model_id+'_test_features.csv'), index=False)
        
        # save Kaggle submission
        submission = test_id.join(predictions[self.model_id+'_1'])
        submission.columns = self.CONFIG['KAGGLE_SUBMISSION_HEADERS']
        
        submission.to_csv(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                       self.model_id,
                                       self.model_id+'_submission.csv'),index=False)
        
        print('Completed createKaggleSubmission: {:%Y-%m-%d %H:%M:%S}'\
              .format(datetime.datetime.now()))
    
###
#
#  Model Performance Tracker
#
###
class ModelPerformanceTracker():
    
    tracking_file = None
    
    def __init__(self,model_trainer=None):
        self.model_trainer = model_trainer
        
        #
        # get global parameters 
        #
        with open('./config.yml') as f:
            self.CONFIG = yaml.load(f.read())
            
        self.tracking_file = os.path.join(self.CONFIG['ROOT_DIR'],'results','model_performance_data.csv')
        
        
    
    def recordModelPerformance(self,
                               cv_metric_list=None  # list of cv performance metrics
                               ):
        # retrieve basic model information from model trainer
        model_params = str(self.model_trainer.model_params)
        model_id = self.model_trainer.model_id
        feature_set = self.model_trainer.feature_set
        date_time = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        
        #create a row
        df = pd.DataFrame([date_time,
                           model_id,
                           feature_set,
                           self.model_trainer.training_rows,
                           self.model_trainer.training_columns,
                           self.model_trainer.training_time,
                           np.min(self.model_trainer.cv_performance_metric),   #cv_min_metric
                           np.max(self.model_trainer.cv_performance_metric),   #cv_max_metric
                           np.mean(self.model_trainer.cv_performance_metric),   #cv_avg_metric
                           "",  #public_leaderboard
                           model_params]).T
        df.columns = ['date_time',
                      'model_id',
                      'feature_set',
                      'number_of_rows',
                      'number_of_columns',
                      'training_time',
                      'cv_min_metric',
                      'cv_max_metric',
                      'cv_avg_metric',
                      'public_leaderboard',
                      'model_params']
        
        
        # write out model performance metric
        if not os.path.isfile(self.tracking_file):
            
           df.to_csv(self.tracking_file, header=True, index=False)
           
        else: # else it exists so append without writing the header
        
           df.to_csv(self.tracking_file, mode='a', header=False, index=False)
        
