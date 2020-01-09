#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:44:00 2019

@author: Johannesreiche
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:52:11 2019

@author: Johannesreiche
"""
from import_file2 import *
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import random

random.seed(42)

X = X[:,0:(X.shape[1]-1)]
yw = y.reshape(579,1)
y = np.zeros((579,1))
y[:,:] = yw
attributeNames = ['age', 'gender', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins',
 'albumin','ag_ratio', 'sgpt', 'sgot', 'alkphos', 'is_patient']
classNames = ['Is patient', 'Not a patient']
N, M = X.shape
C = len(classNames)
# Normalize data
X = stats.zscore(X);        
    
# Parameters for neural network classifier
max_iter = 10000    

loss_fn = torch.nn.BCELoss() 
n_replicates = 1
errors = []

acc_inner = []
acc_outer = []


def crossValidationANN():
    # Create crossvalidation partition for evaluation
    K_o_splits = 10
    outer_it = 0
    K_i_splits = 10
    model_count = 10

    summed_eval_i = np.zeros((model_count))
    eval_i = np.zeros((model_count))
    eval_o = np.zeros((model_count))
    optimal_hidden_units = np.zeros((K_o_splits))

    CV1 = model_selection.StratifiedKFold(n_splits=K_o_splits, shuffle = True)
    CV2 = model_selection.StratifiedKFold(n_splits=K_i_splits, shuffle = True)
    
    #Outer k-fold split
    for train_index_o, test_index_o in CV1.split(X,y):
        print('Outer CV1-fold {0} of {1}'.format(outer_it+1,K_o_splits))
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_o = torch.tensor(X[train_index_o,:], dtype=torch.float)
        y_train_o = torch.tensor(y[train_index_o], dtype=torch.float)
        X_test_o = torch.tensor(X[test_index_o,:], dtype=torch.float)
        y_test_o = torch.tensor(y[test_index_o], dtype=torch.uint8)
        
        #Inner validation loop
        inner_it = 0

        for train_index_i, test_index_i in CV2.split(X_train_o,y_train_o):
            print('Inner CV2-fold {0} of {1}'.format(inner_it+1,K_i_splits))
            # Extract training and test set for current CV fold, convert to tensors
            X_train_i = torch.tensor(X[train_index_i,:], dtype=torch.float)
            y_train_i = torch.tensor(y[train_index_i], dtype=torch.float)
            X_test_i = torch.tensor(X[test_index_i,:], dtype=torch.float)
            y_test_i = torch.tensor(y[test_index_i], dtype=torch.uint8)
            
            #C specifies the inverse of regularization strength. Small C means high regularization
            for idx in range(model_count):
                n_hidden_units = idx+1
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1),
                    torch.nn.Sigmoid()# n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
            # Train the net on training data
                net, final_loss, learning_curve = train_neural_net(model,
                                                                   loss_fn,
                                                                   X=X_train_i,
                                                                   y=y_train_i,
                                                                   n_replicates=n_replicates,
                                                                   max_iter=max_iter)
                y_sigmoid = net(X_test_i)
                y_test_est = y_sigmoid>.5
                e = y_test_est != y_test_i
                current_err = 100*( (sum(e).type(torch.float)/len(y_test_i)).data.numpy())
                a = 100*( (sum(y_test_est == y_test_i).type(torch.float)/len(y_test_i)).data.numpy())
                errors.append(current_err) # store error rate for current CV fold
                acc_inner.append(a)

              
                summed_eval_i[idx] += current_err
            
            inner_it += 1
            
        
        eval_i = summed_eval_i * (len(X_test_i)/len(X_train_o))     
        idx = np.argmin(eval_i)
        n_hidden_units = idx+1
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), 
                    torch.nn.ReLU(),   
                    torch.nn.Linear(n_hidden_units, 1),
                    torch.nn.Sigmoid()
                    )
                
            # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                          loss_fn,
                                                          X=X_train_o,
                                                          y=y_train_o,
                                                          n_replicates=n_replicates,
                                                          max_iter=max_iter)
        y_sigmoid = net(X_test_o)
        y_test_est = y_sigmoid>.5
        # Determine errors and errors
        e = y_test_est != y_test_o
        a = 100*( (sum(y_test_est == y_test_o).type(torch.float)/len(y_test_o)).data.numpy())
        current_err = 100*( (sum(e).type(torch.float)/len(y_test_o)).data.numpy())
        eval_o[outer_it] = current_err
        acc_outer.append(a)
        errors.append(current_err) # store error rate for current CV fold 
        optimal_hidden_units[outer_it] = n_hidden_units
        
        
        outer_it+=1
        
    mode_reg, _= np.unique(optimal_hidden_units, return_counts=True)
    e_gen = np.sum(eval_o) * (len(X_test_o)/ len(X))
    
#
    print("ANN with variable hidden units and one layer")
    print("ANN accuracy: %f with %s and %i" % (np.max(a),'number of layers',mode_reg[0]))
    print("ANN generalization error: %f with %s and %i" % (e_gen,'number of layers',mode_reg[0]))
    print(optimal_hidden_units)
    print(acc_outer)
    print(eval_o)
    

    
crossValidationANN()