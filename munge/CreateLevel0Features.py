from framework.model_stacking import FeatureGenerator

#%%
import yaml
import os.path 
import pandas as pd


with open('./config.yml') as f:
    CONFIG = yaml.load(f.read())
    
print('root dir: ',CONFIG['ROOT_DIR'])

my_generator = FeatureGenerator(CONFIG['ROOT_DIR'],'raw','L0FS01')

#%%
my_generator.getRawData()

#%%
dir(yaml)


df = pd.read_csv('./data/raw/train.csv')