# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:24:53 2019

@author: Mads-
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:04:30 2019

@author: Mads-
"""

#Complete classification scripts
import random
import os
import numpy
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from import_file import *
random.seed(42)
#Logistic packages:
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from scipy import stats

classNames = np.asarray(df.columns)
M, N = X.shape
C = len(classNames)
X = stats.zscore(X)

def crossValidationKNN():
    # Create crossvalidation partition for evaluation
    K_o_splits = 10
    outer_it = 0
    K_i_splits = 10
    model_count = 10

    summed_eval_i = np.zeros((model_count))
    eval_i = np.zeros((model_count))
    eval_o = np.zeros((model_count))
    optimal_lambda = np.zeros((K_o_splits))

    #CV1 = model_selection.KFold(n_splits=K_o_splits,shuffle=True)
    #StratifiedKfold ensures that there is a reasonable percentage of each class in each split.
    CV1 = model_selection.StratifiedKFold(n_splits=K_o_splits, shuffle = True)
    CV2 = model_selection.StratifiedKFold(n_splits=K_i_splits, shuffle = True)
    
    #Outer k-fold split
    for train_index_o, test_index_o in CV1.split(X,y):
        print('Outer CV1-fold {0} of {1}'.format(outer_it+1,K_o_splits))
        
        X_train_o = X[train_index_o,:]
        y_train_o = y[train_index_o]
        X_test_o = X[test_index_o,:]
        y_test_o = y[test_index_o]
        
        #Inner validation loop
        inner_it = 0

        for train_index_i, test_index_i in CV2.split(X_train_o,y_train_o):
            print('Inner CV2-fold {0} of {1}'.format(inner_it+1,K_i_splits))
            X_train_i = X[train_index_i,:]
            y_train_i = y[train_index_i]
            X_test_i = X[test_index_i,:]
            y_test_i = y[test_index_i]
            
            #C specifies the inverse of regularization strength. Small C means high regularization
            for idx in range(model_count):
                reg_term = (1+idx*3)
                
                knclassifier = KNeighborsClassifier(n_neighbors=reg_term);
                knclassifier.fit(X_train_i, y_train_i);
                y_est = knclassifier.predict(X_test_i);
                current_err = 100*( (y_est!=y_test_i).sum().astype(float)/ len(y_test_i))
                
              
                summed_eval_i[idx] += current_err
            
            inner_it += 1
            
        
        eval_i = summed_eval_i * (len(X_test_i)/len(X_train_o))     
        idx = np.argmin(eval_i)
        reg_term = (1+idx*2)
        
        knclassifier = KNeighborsClassifier(n_neighbors=reg_term);
        knclassifier.fit(X_train_o, y_train_o);
        y_est = knclassifier.predict(X_test_o);
        current_err = 100*( (y_est!=y_test_o).sum().astype(float)/ len(y_test_o))
        
        eval_o[outer_it] = current_err
        optimal_lambda[outer_it] = reg_term
        
        
        outer_it+=1
        
    mode_reg, _= numpy.unique(optimal_lambda, return_counts=True)
    
    figure()
    boxplot(eval_o)
    xlabel('KNN')
    ylabel('Cross-validation error [%]')
    show()
    e_gen = np.sum(eval_o) * (len(X_test_o)/ len(X))
    print("KNN generalization error: %f with %s and %i" % (e_gen,'neighbours',mode_reg[0]))
    print(eval_o)

    
crossValidationKNN()