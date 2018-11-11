#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:04:39 2018

@author: jim
"""

import yaml
import pandas as pd
import os.path
import numpy as np
from scipy.stats import mstats

#%%
###
#
# Class for blending models
#
###
class ModelBlender():
    """
    Blends two or model predicitons.
    
    Blending is either simple averaging or geometric mean
    
    Arguments:
        in_dir: list of directories containing predictions in test.csv to blend
        model_id: identifier for the blender
        method: "simple_average" or "geometric_mean"

    """    
    def __init__(self,in_dir=None,
                 method='simple_average'):
        
        self.in_dir = in_dir
        self.model_id = model_id
        self.method = method
        
        #
        # get parameters 
        #
        with open('./config.yml') as f:
            self.CONFIG = yaml.load(f.read())
            
        self.root_dir = self.CONFIG['ROOT_DIR']
        self.id_vars = self.CONFIG['ID_VAR']      
        self.target_var = self.CONFIG['TARGET_VAR'] 
        
        
    def blendPredictions(self):
        
        combine_list = []
        found_first_df = False
        for d in self.in_dir:
            test_df = pd.read_csv(os.path.join(self.root_dir,'models',d,'test.csv'))
            if not found_first_df:
                id_df = test_df[self.id_vars]
                found_first_df = True
                
            combine_list.append(test_df.drop(self.id_vars))
            
        combined = np.dstack(combine_list)  
            
        if self.method == 'simple_average':
            result = combined.mean(axis=2)
        else:
            result = mstats.gmean(combined.axis=2)
            
            
        results_df = id_df.join(pd.DataFrame(result),index=id_df.index)
        results_df.columns = self.id_vars + [self.model_id+"_0", self.model_id+'_1']
        
        result_df.to_csv(os.path.join(seld.root_dir,'model',self.model_id,"test.csv"),
                         index=False)
            
        
        
                