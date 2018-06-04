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
            df = pd.read_csv(os.path.join(self.CONFIG['ROOT_DIR'],
                                          'models',
                                          model_id,
                                          model_id+'_features.csv'))
            features.append(df)
            
        return features
    
    
#%%
fs = FeatureGeneratorNextLevel(in_dir=['L0RF1','L0NN1'],
                               out_dir='L1FS01')

#%%
print(fs.in_dir)
print(fs.out_dir)
print(fs.CONFIG)


#%%
df = fs.getRawData()

#%%