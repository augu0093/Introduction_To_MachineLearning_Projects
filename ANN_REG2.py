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

attributeNames = ['age', 'gender', 'tot_proteins',
 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos', 'is_patient', 'direct_bilirubin']

XC = np.zeros_like(X)
XC[:,9]=X[:,3]
XC[:,0:3]=X[:,0:3]
XC[:,2:9]=X[:,4:11]
y = XC[:,[9]]             
XC = XC[:,0:9]           
N, M = XC.shape
C = 2

X = np.copy(XC)

# Normalize data
X = stats.zscore(X);        
    
# Parameters for neural network classifier
max_iter = 10000    

loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss    # 
n_replicates = 1
errors = []
def crossValidationANN():
    # Create crossvalidation partition for evaluation
    K_o_splits = 10
    outer_it = 0
    K_i_splits = 10
    model_count = 10

    summed_eval_i = np.zeros((model_count))
    eval_i = np.zeros((model_count))
    eval_o = np.zeros((K_o_splits))
    optimal_hidden_units = np.zeros((K_o_splits))

    CV1 = model_selection.KFold(n_splits=K_o_splits,shuffle=True)
    CV2 = model_selection.KFold(n_splits=K_i_splits, shuffle = True)
    
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
            
            for idx in range(model_count):
                if idx == 0:
                    n_hidden_units = idx+1
                else:
                    n_hidden_units = idx*5
                model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), 
                    torch.nn.ReLU(),   
                    torch.nn.Linear(n_hidden_units, 1), 
                    )
                loss_fn = torch.nn.MSELoss() 


                net, final_loss, learning_curve = train_neural_net(model,
                                                                   loss_fn,
                                                                   X=X_train_i,
                                                                   y=y_train_i,
                                                                   n_replicates=n_replicates,
                                                                   max_iter=max_iter)

                y_test_est = net(X_test_i)
                se = (y_test_est.float()-y_test_i.float())**2 
                mse = (sum(se).type(torch.float)/len(y_test_i)).data.numpy() 
                errors.append(mse) 
                
              
                summed_eval_i[idx] += mse
            
            inner_it += 1
            
        
        eval_i = summed_eval_i * (len(X_test_i)/len(X_train_o))
        print(eval_i)
        idx = np.argmin(eval_i)
        print(idx)
        n_hidden_units = idx+1
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), 
                    torch.nn.ReLU(),  
                    torch.nn.Linear(n_hidden_units, 1), 
                    )
                
        net, final_loss, learning_curve = train_neural_net(model,
                                                          loss_fn,
                                                          X=X_train_o,
                                                          y=y_train_o,
                                                          n_replicates=n_replicates,
                                                          max_iter=max_iter)
        y_test_est = net(X_test_o)
        # Determine errors and errors
        se = (y_test_est.float()-y_test_o.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test_o)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
        
        
        eval_o[outer_it] = mse
        optimal_hidden_units[outer_it] = n_hidden_units
        
        
        outer_it+=1
        
    mode_reg, _= np.unique(optimal_hidden_units, return_counts=True)
    
    e_gen = np.sum(eval_o) * (len(X_test_o)/ len(X))
    print("ANN reggresion with variable hidden units and one layer")
    print("ANN generalization error: %f with %s and %i" % (e_gen,'hidden units',mode_reg[0]))
    print(eval_o)
    print(optimal_hidden_units)

    
crossValidationANN()