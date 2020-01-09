#Complete classification scripts
import os
import numpy
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from import_file2 import *
import random

#Logistic packages:
import sklearn.linear_model as lm
from sklearn import model_selection
from scipy import stats
random.seed(42)

XC = np.zeros_like(X)
XC[:,9]=X[:,3]
XC[:,0:3]=X[:,0:3]
XC[:,2:9]=X[:,4:11]
y = XC[:,[9]]             
XC = XC[:,0:9]           
N, M = XC.shape
C = 2

# Normalize data
XC = stats.zscore(XC)

### Desired prediction-attribute is moved as to be the last attribute
### tot_bilirubin is removed for predicting dir_bilirubin
#Xr = np.delete(X,2,1)
#
## In X the attribute order is: ['age' 'gender' 'gender' 'direct_bilirubin' 'tot_proteins' 'albumin' 'ag_ratio' 'sgpt' 'sgot' 'alkphos']
#predictAttributeNum = 2 #direct_billirubin
#
#X = np.zeros_like(Xr)
#X[:,int(len(Xr[1])-1)]=Xr[:,predictAttributeNum]
#X[:,0:predictAttributeNum]=Xr[:,0:predictAttributeNum]
#X[:,predictAttributeNum:int(len(Xr[1])-1)]=Xr[:,predictAttributeNum+1:int(len(Xr[1]))]
#
#y = np.array([X[:,len(Xr[1])-1]]).squeeze()
#
#
#X = np.delete(X,len(Xr[1])-1,1)
#N,M = X.shape
#
### Edits attributenames to fit new data-matrix
#attributeNamesN = np.delete(attributeNames,2,0)
#attributeNamesN = np.delete(attributeNamesN,len(attributeNamesN)-1,0)
##attributeNamesN = np.insert(attributeNamesN,1,'gender')
##print(attributeNamesN)
#attributeNames = np.zeros_like(attributeNamesN)
#attributeNames[len(attributeNamesN)-1] = attributeNamesN[predictAttributeNum]
#attributeNames[0:predictAttributeNum]=attributeNamesN[0:predictAttributeNum]
#attributeNames[predictAttributeNum:len(attributeNamesN)-1]=attributeNamesN[predictAttributeNum+1:len(attributeNamesN)]
##print(attributeNames)
#
### Standardize data
#X = stats.zscore(X)

def baselineRegression(X,y):
    y_pred = np.mean(y)
    sample_size = y.shape[0]
    error = np.sum( (y-y_pred)**2 ) / sample_size
    return error 


def baselineRegression_k_cross(X,y):
    # Create crossvalidation partition for evaluation
    K_o_splits = 10
    outer_it = 0
    K_i_splits = 10
    eval_o_reg = np.zeros((K_o_splits))

    CV1 = model_selection.KFold(n_splits=K_o_splits,shuffle=True)
    #StratifiedKfold ensures that there is a reasonable percentage of each class in each split.
    #CV1 = model_selection.StratifiedKFold(n_splits=K_o_splits, shuffle = True)

    
    #Outer k-fold split
    for train_index_o, test_index_o in CV1.split(X,y):
        print('Outer CV1-fold {0} of {1}'.format(outer_it+1,K_o_splits))
        
        X_train_o = X[train_index_o,:]
        y_train_o = y[train_index_o]
        X_test_o = X[test_index_o,:]
        y_test_o = y[test_index_o]
        
        #Inner validation loop

        
        reg_err = baselineRegression(X_train_o,y_train_o)
            
        
        eval_o_reg[outer_it] = reg_err
        
        
        outer_it+=1

    figure()
    boxplot(eval_o_reg)
    xlabel('Baseline Regression')
    ylabel('Cross-validation error [%]')
    show()

    
#------------------------------
baselineRegression_k_cross(XC,y)   