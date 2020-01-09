# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:35:40 2019

@author: August
"""

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
#from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from scipy import stats
import random
random.seed(42)
from import_file2 import *

## Desired prediction-attribute is moved as to be the last attribute
# In X the attribute order is: ['age' 'gender' 'direct_bilirubin' 'tot_proteins' 'albumin' 'ag_ratio' 'sgpt' 'sgot' 'alkphos']
## Edits attributenames to fit new data-matrix

### tot_bilirubin is removed for predicting dir_bilirubin
#Xr = X
#predictAttributeNum = 3 #direct_billirubin
#attributeNamesN = attributeNames

### tot_bilirubin is not removed
Xr = np.delete(X,2,1)
predictAttributeNum = 2 #direct_billirubin
attributeNamesN = np.delete(attributeNames,2,0)





#print(attributeNamesN)


X = np.zeros_like(Xr)
X[:,int(len(Xr[1])-1)]=Xr[:,predictAttributeNum]
X[:,0:predictAttributeNum]=Xr[:,0:predictAttributeNum]
X[:,predictAttributeNum:int(len(Xr[1])-1)]=Xr[:,predictAttributeNum+1:int(len(Xr[1]))]

y = np.array([X[:,len(Xr[1])-1]]).squeeze()
X = np.delete(X,len(Xr[1])-1,1)
N,M = X.shape


attributeNames = np.zeros_like(attributeNamesN)
attributeNames[len(attributeNamesN)-1] = attributeNamesN[predictAttributeNum]
attributeNames[0:predictAttributeNum]=attributeNamesN[0:predictAttributeNum]
attributeNames[predictAttributeNum:len(attributeNamesN)-1]=attributeNamesN[predictAttributeNum+1:len(attributeNamesN)]
print(attributeNames)



## Standardize data
X = stats.zscore(X)

k = 10



def kfold_cv(data: Data, k: int):

  #log("Running %i-fold cross validation for linear regression" % k)
  lambdas = np.logspace(-3, 3, 30)
  nobs_per_batch = data.x.shape[0] // k
  train_losses = np.empty((k, len(lambdas)))
  losses = np.empty((k, len(lambdas)))
  for i in range(k):
    idcs = np.zeros(data.x.shape[0], dtype=np.bool)
    idcs[i*nobs_per_batch:(i+1)*nobs_per_batch] = 1
    inner_train = Data(
      data.x[idcs],
      data.y[idcs],
    )
    inner_test = Data(
      data.x[~idcs],
      data.y[~idcs],
    )
    
    model_train_losses = np.empty(len(lambdas))
    model_losses = np.empty(len(lambdas))
    for k1, cfg in enumerate(lambdas):
      w = lr_l2(inner_train.x, inner_train.y, cfg)
      yhat = inner_test.x @ w
      # lm = LinearRegression(fit_intercept=False)
      # lm.fit(inner_train.x, inner_train.y)
      # yhat = lm.predict(inner_test.x)
      # print(inner_test.y.mean(), yhat.mean(), w)
      train_loss = ((inner_train.y - inner_train.x @ w)**2).sum() / inner_train.y.size
      model_train_losses[k1] = train_loss
      loss = ((inner_test.y - yhat)**2).sum() / inner_test.y.size
      model_losses[k1] = loss
    train_losses[i] = model_train_losses
    losses[i] = model_losses
    
  losses = losses.mean(axis=0)
  train_losses = train_losses.mean(axis=0)

  plt.plot(lambdas, losses)
  # plt.plot(configs, train_losses)
  plt.xscale("log")
  plt.yscale("log")
  plt.ylim(10**-2)
  plt.grid(True)
  plt.show()