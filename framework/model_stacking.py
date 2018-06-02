#%%
import yaml
import pandas as pd
import os
import os.path


class FeatureGenerator():
    
    raw_id_df = None        # pandas data frame to hold id variables
    raw_target_df = None    ## pandas series holding target variable
    raw_train_features_df = None    # pandas dataframe hoding orginial predictor variables    
    raw_test_features_df = None     # test data set for features
    
    def __init__(self,root_dir=None,
                 in_dir=None,
                 out_dir=None,
                 id_vars=['ID'],
                 target_var='target'):
        self.root_dir = root_dir
        self.in_dir = in_dir        # location for input data
        self.out_dir = out_dir      # location to save generated feature set
        self.id_vars = id_vars      # list of identifier attributes
        self.target_var = target_var #  name of target variable
    
    
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
        self.raw_id_df = df.loc[:,self.id_vars]
        self.raw_target_df = df.loc[:,[self.target_var]]
        
        # isolate predictor variables
        predictors = sorted(set(df.columns) - set(self.id_vars) - set([self.target_var]))
        
        self.raw_train_features_df = df.loc[:,predictors]
        self.raw_test_features_df = df.loc[:,predictors]
        
    
    def saveFeatureSet(self,new_features_df):
        #
        # default behaviour - can be overriddent for different new feature storage
        #
        pass

