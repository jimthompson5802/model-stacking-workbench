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
fs = FeatureGeneratorNextLevel(in_dir=['L0RF1','L0NN1'],
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
        train_df = t[0].iloc[:,:2].join(t[0].iloc[:,3])
        test_df = t[1].iloc[:,:1].join(t[1].iloc[:,2])
        found_first_df = True
    else:
        train_df = train_df.join(t[0].iloc[:,3])
        test_df = test_df.join(t[1].iloc[:,2])
        
#%%
fs.saveFeatureSet(train_df,test_df)