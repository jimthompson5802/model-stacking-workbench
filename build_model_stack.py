# -*- coding: utf-8 -*-

###
# This will invoke required python modules to build feature sets
# and models to create the model stack.
###
#%%
import yaml
import os.path

#%%
#
# get parameters 
#
with open('./config.yml') as f:
    CONFIG = yaml.load(f.read())

ROOT_DIR = CONFIG['ROOT_DIR']
#%%
#
# Create k-fold specification for training
#
exec(open(os.path.join(ROOT_DIR,'munge','createCVFolds.py')).read())

#%%
#
# Create feature sets for Level 0
#
exec(open(os.path.join(ROOT_DIR,'munge','createLevel0BaselineFeatureSet.py')).read())
exec(open(os.path.join(ROOT_DIR,'munge','createLevel0Features.py')).read())

#%%
#
# Train models on Level 0
#
exec(open(os.path.join(ROOT_DIR,'models','L0LOG1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,'models','L0NN1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,'models','L0RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,'models','L0XTC1','train_model.py')).read())

#%%
#
# Create feature sets for Level 1
#
exec(open(os.path.join(ROOT_DIR,'munge','createLevel1Features.py')).read())

#%%
#
# Train models on Level 1
#
exec(open(os.path.join(ROOT_DIR,'models','L1RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,'models','L1NN1','train_model.py')).read())

#%%
#
# Create feature sets for Level 2
#
exec(open(os.path.join(ROOT_DIR,'munge','createLevel2Features.py')).read())

#%%
#
# Train models on Level 2
#
exec(open(os.path.join(ROOT_DIR,'models','L2RF1','train_model.py')).read())
exec(open(os.path.join(ROOT_DIR,'models','L2NN1','train_model.py')).read())