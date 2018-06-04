
import yaml
import pandas as pd
import os
import os.path
import shutil
import pickle
import datetime

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
        #
        # retrieve KFold specifiction
        #
        with open(os.path.join(self.CONFIG['ROOT_DIR'],'data','fold_specification.pkl'),'rb') as f:
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
        next_level = []
        i = 0
        for fold in k_folds:
            i += 1
            print('running fold: {:d}'.format(i))
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
        
            # geneate features for next level
            y_hat = pd.DataFrame(model.predict_proba(X_holdout),index=id_holdout.index)
            y_hat.columns = [self.model_id+'_'+str(col) for col in y_hat.columns]
            y_hat = id_holdout.join(y_holdout).join(y_hat)
            
            next_level.append(y_hat)
            
        #
        # combine the generated features into single dataframe & save to disk
        #
        pd.concat(next_level).sort_values(self.CONFIG['ID_VAR'])\
            .to_csv(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                                 self.model_id,
                                 self.model_id+'_train_features.csv'),
                    index=False)


    def trainModel(self):
        #
        # train model on complete training data set
        #
        # retrieve training data
        train_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],'data',self.feature_set,self.train_ds))
        predictors = sorted(list(set(train_df.columns) - set(self.CONFIG['ID_VAR']) - set([self.CONFIG['TARGET_VAR']])))
        
        
        X_train = train_df[predictors]
        y_train = train_df[self.CONFIG['TARGET_VAR']]
            
        model = self.ModelClass(**self.model_params)
            
        model.fit(X_train,y_train)
        
        with open(os.path.join(self.CONFIG['ROOT_DIR'],'models',
                               self.model_id,self.model_id+'_model.pkl'),'wb') as f:
            pickle.dump(model,f)
            

    def createKaggleSubmission(self,feature_set=None,test_ds='test.csv'):
        #
        # create Kaggle Submission
        #
        # Assumes:  trained model has been saved under "`model_id`_model.pkl"
        #
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
            
        self.tracking_file = os.path.join(self.CONFIG['ROOT_DIR'],'results','model_performance_data.tsv')
        
        
    
    def recordModelPerformance(self,
                               cv_performanc_list=None  # list of cv performance metrics
                               ):
        # retrieve basic model information from model trainer
        model_params = str(self.model_trainer.model_params)
        model_id = self.model_trainer.model_id
        feature_set = self.model_trainer.feature_set
        
        ####
        
        
