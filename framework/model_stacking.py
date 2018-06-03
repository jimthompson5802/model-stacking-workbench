#%%
import yaml
import pandas as pd
import os
import os.path
import shutil


class FeatureGenerator():
    
    raw_id_df = None        # pandas data frame to hold id variables
    raw_target_df = None    ## pandas series holding target variable
    raw_train_features_df = None    # pandas dataframe hoding orginial predictor variables    
    raw_test_features_df = None     # test data set for features
    
    def __init__(self,
                 in_dir=None,
                 out_dir=None,
                 id_vars=['ID'],
                 target_var='target'):

        self.in_dir = in_dir        # location for input data
        self.out_dir = out_dir      # location to save generated feature set
        self.id_vars = id_vars      # list of identifier attributes
        self.target_var = target_var #  name of target variable
        
        #
        # get parameters 
        #
        with open('./config.yml') as f:
            CONFIG = yaml.load(f.read())
            
        self.root_dir = CONFIG['ROOT_DIR']
        
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
        self.raw_target_df = df.loc[:,[self.target_var]]
        
        # isolate predictor variables
        predictors = sorted(set(df.columns) - set(self.id_vars) - set([self.target_var]))
        
        # isoloate training predictiors
        self.raw_train_features_df = df.loc[:,predictors]
        
        # get test data set
        df = pd.read_csv(os.path.join(self.root_dir,'data',self.in_dir,'test.csv'))
        self.raw_test_id_df = df.loc[:,self.id_vars]
        self.raw_test_features_df = df.loc[:,predictors]
        
    
    def saveFeatureSet(self,new_train_features_df=None,new_test_features_df=None):
        #
        # default behaviour - can be overriddent for different new feature storage
        #
        # append id_vars and target_var save new_train_features_df and 
        # new_test_features_df in self.out_dir
        #
        # append id-vars and target to new feature set and save as csv
        
        try:
            self.raw_train_id_df.join(self.raw_target_df)\
                .join(new_train_features_df)\
                .sort_values(self.id_vars)\
                .to_csv(os.path.join(self.root_dir,'data',self.out_dir,'train.csv'),index=False)
        except:
            pass
        
        
        try:
            self.raw_test_id_df.join(self.raw_target_df)\
                .join(new_test_features_df)\
                .sort_values(self.id_vars)\
                .to_csv(os.path.join(self.root_dir,'data',self.out_dir,'test.csv'),index=False)
        except:
            pass
        
     
class ModelTrainer():
    
    def __init__(self,
                 Model=None,
                 model_params=None,
                 model_id=None,
                 feature_set=None,
                 train_ds='train.csv',
                 test_ds='test.csv'):
        
        self.ModelClass = ModelClass
        self.model_params = model_params
        self.model_id = model_id
        self.feature_set = feature_set
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        
    
    def createFeaturesForNextLevel(self):
        pass


    def trainModel(self):
        pass
    


    def createSubmission(self,feature_set=None,test_ds='test.csv'):
        pass
    

