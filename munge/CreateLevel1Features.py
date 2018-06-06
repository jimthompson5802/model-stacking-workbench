#%%
from framework.model_stacking import FeatureGenerator

#%%
import yaml
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

#%%
#
# Define Feature Generator for Level 1
#
class FeatureGeneratorNextLevel(FeatureGenerator):
    
    def getRawData(self):
        features = []
        for model_id in self.in_dir:
            train_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],
                                          'models',
                                          model_id,
                                          model_id+'_train_features.csv'))

            test_df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],
                                          'models',
                                          model_id,
                                          model_id+'_test_features.csv'))
             
            features.append((train_df,test_df))
            
        return features
    
    
#%%
fs = FeatureGeneratorNextLevel(in_dir=['L0RF1','L0NN1','L0XTC1'],
                               out_dir='L1FS01')


#%%
#
# retrieve candidate train and test data sets from prior level
#
# this is a list of tuples.  Each tuple structure: (train_df, test_df)
features = fs.getRawData()

#%%
#
# Assemble train and test data sets for next Level
#

# assemble train data set
found_first_df = False
for t in features:
    if not found_first_df:
        fs.raw_train_id_df = pd.DataFrame(t[0].iloc[:,0])
        y_train_df = pd.DataFrame(t[0].iloc[:,1])
        X_train_df = pd.DataFrame(t[0].iloc[:,3])
        fs.raw_test_id_df = pd.DataFrame(t[1].iloc[:,0])
        X_test_df = pd.DataFrame(t[1].iloc[:,2])
        found_first_df = True
    else:
        X_train_df = X_train_df.join(t[0].iloc[:,3])
        X_test_df = X_test_df.join(t[1].iloc[:,2])
        
#%%
fs.saveFeatureSet(X_train_df,y_train_df,X_test_df)