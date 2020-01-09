#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Johannesreiche
"""

from matplotlib.pyplot import boxplot, xticks, ylabel, title, show, savefig, grid

# requires data from exercise 4.2.1
from import_file import *
X1 = np.delete(X, 1, 1)
X1 = np.delete(X1, 1, 1)
A = np.delete(attributeNames,10,0)
A = np.delete(A,1,0)

B1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

boxplot(X1)
grid()
xticks(range(1,10),B1)
savefig('BoxplotOR')
show()

# Calculating mean and std for each attribute
mean_d = []
std_d = []
for i in range(9):
        mean_d.append(np.mean(X1[:,i]))
        std_d.append(np.std(X1[:,i]))

mean_d = np.reshape(mean_d, (1,9))
std_d = np.reshape(std_d, (1,9))

# Standardize the data 
# subtract mean column values and divide by the standard deviation column values
Xc = X1.copy()
for j in range(9):
    Xc[:,j] = (X1[:,j] - mean_d[0,j])/std_d[0,j] 
        

boxplot(Xc[:,0:4])
grid()
xticks(range(1,5),A[0:4])
savefig('BoxplotNew1')
show()


boxplot(Xc[:,4:9])
grid()
xticks(range(1,6),A[4:9])
savefig('BoxplotNew2')
show()
