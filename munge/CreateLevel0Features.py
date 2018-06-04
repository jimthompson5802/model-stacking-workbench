from framework.model_stacking import FeatureGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

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

#%%
#
# feature set 1
#
print("Preparing Level 0 Feature Set 1")

fs = FeatureGenerator('raw','L0FS01')

# get raw data
X_train, y_train, X_test = fs.getRawData()

X_train.fillna(-999,inplace=True)

X_test.fillna(-999,inplace=True)

fs.saveFeatureSet(X_train, y_train, X_test)


#%%
#
# feature set 2
#
print("Preparing Level 0 Feature Set 2")

fs = FeatureGenerator('raw','L0FS02')

# get raw data
X_train, y_train, X_test = fs.getRawData()


# find only numberic attributes
numeric_predictors = [x for x in X_train.columns if X_train[x].dtype != 'O']

X_train = X_train.loc[:,numeric_predictors]
X_train.fillna(-999,inplace=True)
X_train.shape

X_test = X_test.loc[:,numeric_predictors]
X_test.fillna(-999,inplace=True)
X_test.shape

fs.saveFeatureSet(X_train, y_train, X_test)

#%%
#
# feature set 3
#
print("Preparing Level 0 Feature Set 3")

fs = FeatureGenerator('raw','L0FS03')

# get raw data
X_train, y_train, X_test = fs.getRawData()

# find only numberic attributes
numeric_predictors = [x for x in X_train.columns if X_train[x].dtype != 'O']

X_train = X_train.loc[:,numeric_predictors]

# impute mean value for missing values
imp = Imputer()
X_train = imp.fit_transform(X_train)

mms = MinMaxScaler()

# min/max scale data and convert to data frame, ensure index values match
# original data frame
X_train = pd.DataFrame(mms.fit_transform(X_train),index=fs.raw_train_id_df.index)
X_train.columns = numeric_predictors

X_test = X_test.loc[:,numeric_predictors]

# impute missinvg values
X_test = imp.transform(X_test)

# Apply min/max transform and 
# convert to data frame,  ensure index values match original data frame
X_test = pd.DataFrame(mms.transform(X_test),index=fs.raw_test_id_df.index)
X_test.columns = numeric_predictors


# save new feature set
fs.saveFeatureSet(X_train, y_train, X_test)

#%%