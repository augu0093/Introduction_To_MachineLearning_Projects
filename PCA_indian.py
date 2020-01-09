#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johannesreiche
"""

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
import scipy.linalg as linalg
import numpy as np
from import_file import * 


# Calculating mean and std for each attribute
mean_d = []
std_d = []
for i in range(11):
    if i == 1 or i == 2:
        mean_d.append(0)
        std_d.append(1)
    else:
        mean_d.append(np.mean(X[:,i]))
        std_d.append(np.std(X[:,i]))

mean_d = np.reshape(mean_d, (1,11))
std_d = np.reshape(std_d, (1,11))

# Standardize the data 
# subtract mean column values and divide by the standard deviation column values
Xc = X.copy()
for j in range(11):
    Xc[:,j] = (X[:,j] - mean_d[0,j])/std_d[0,j] 
        

# Singular-Value-Decomposition
# S = Eigenvalues, V = Eigenvectors  
U,S,V = linalg.svd(Xc,full_matrices=False)

V = V.T
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

# Project data onto principal component space
Z = Xc @ V

# Plot variance explained
xx = np.linspace(1,11,11)
f = np.ones(11)*0.8
summ = S.sum()
cumsum = 0
total_var_explained = np.zeros(11)
for i in range(11):    
    cumsum = cumsum + rho[i]
    total_var_explained[i]=cumsum
figure()
plt.plot(xx,f, color = 'black', linestyle='dashed', linewidth=1.5)
plt.plot(xx,total_var_explained)
plt.scatter(xx,total_var_explained)
plt.plot(xx,rho)
plt.scatter(xx,rho)
plt.grid()
legend(['80 %', 'Cumulative', 'Individual'])
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained value');
plt.savefig('VarianceEx.png')
plt.show();

n = [1,2]

N,M = Xc.shape
C = len(n)

attributeNames = ['Liver patient', 'Non-liver patient']
f = figure()
title('Indian patients vectors projected on PCs')
for c in n:
    # select indices belonging to class c:
    class_mask = (y == c)
    plot(Z[class_mask,0], Z[class_mask,1], 'o')
legend(attributeNames)
xlabel('PC1')
ylabel('PC2')
plt.savefig('ProjectionPC.png')