# -*- coding: utf-8 -*-
"""Continuous ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_9uOrm5Jv85FtUv8i7pXieSyknvvnJue
"""

import numpy as np
import pandas as pd
import numpy as np
train = pd.read_csv()
test = pd.read_csv()

label = ['speed']
del_col = ['speed','flow']

train_label = train[label]
train_feat = train.drop(del_col,axis=1)

test_label = test[label]
test_feat = test.drop(del_col,axis=1)

train_feat = np.array(train_feat)
train_label = np.array(train_label)
test_feat = np.array(test_feat)
test_label = np.array(test_label)

from sklearn.svm import SVR
import numpy as np
SVR = SVR(gamma='scale', C=1000, epsilon=0.1)
SVR.fit(train_feat, train_label)

model_pre = SVR.predict(test_feat)
test_preNreal5 = pd.DataFrame()
test_preNreal5['real'] = test_label.flatten()
test_preNreal5['pre'] = model_pre
test_preNreal5.to_csv('SVM_wow.csv',index=False) 

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn import metrics
y_true = test_preNreal5['real']
y_pred = test_preNreal5['pre']


print(np.sqrt(metrics.mean_squared_error(y_true,y_pred)))
print(mean_absolute_percentage_error(y_true, y_pred))

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators= 200,random_state= 42)#n_estimators= 200,random_state=42
RF.fit(train_feat, train_label)

model_pre = RF.predict(test_feat)
test_preNreal5 = pd.DataFrame()
test_preNreal5['real'] = test_label.flatten()
test_preNreal5['pre'] = model_pre
test_preNreal5.to_csv('RF_wow.csv',index=False) 

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn import metrics
y_true = test_preNreal5['real']
y_pred = test_preNreal5['pre']


print(np.sqrt(metrics.mean_squared_error(y_true,y_pred)))
print(mean_absolute_percentage_error(y_true, y_pred))

from sklearn.neural_network import MLPRegressor
ANN = MLPRegressor(learning_rate_init=0.001,batch_size=20,tol=0.01,learning_rate='constant',hidden_layer_sizes=(1000,),solver='adam')
ANN.fit(train_feat, train_label)

model_pre = ANN.predict(test_feat)
test_preNreal5 = pd.DataFrame()
test_preNreal5['real'] = test_label.flatten()
test_preNreal5['pre'] = model_pre
test_preNreal5.to_csv('ANN_wow.csv',index=False) 

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
from sklearn import metrics

y_true = test_preNreal5['real']
y_pred = test_preNreal5['pre']


print(np.sqrt(metrics.mean_squared_error(y_true,y_pred)))
print(mean_absolute_percentage_error(y_true, y_pred))

import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.1, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=3000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =42, nthread = -1)
model_xgb.fit(train_feat, train_label)

model_pre = model_xgb.predict(test_feat)
test_preNreal5 = pd.DataFrame()
test_preNreal5['real'] = test_label.flatten()
test_preNreal5['pre'] = model_pre
test_preNreal5.to_csv('XGB_wow.csv',index=False) 

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = test_preNreal5['real']
y_pred = test_preNreal5['pre']


print(np.sqrt(metrics.mean_squared_error(y_true,y_pred)))
print(mean_absolute_percentage_error(y_true, y_pred))

from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.3,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)
GBoost.fit(train_feat, train_label)

model_pre = GBoost.predict(test_feat)
test_preNreal5 = pd.DataFrame()
test_preNreal5['real'] = test_label.flatten()
test_preNreal5['pre'] = model_pre
test_preNreal5.to_csv('GBDT_wow.csv',index=False) 

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = test_preNreal5['real']
y_pred = test_preNreal5['pre']


print(np.sqrt(metrics.mean_squared_error(y_true,y_pred)))
print(mean_absolute_percentage_error(y_true, y_pred))