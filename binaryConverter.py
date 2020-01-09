# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:47:16 2019

@author: August
"""

### This script transforms the Indian-Liver data set to binary values

import numpy as np
from import_file2 import X, attributeNames

## Finds the shape of X
M,N = np.shape(X)

## Create new np.array that is double width for one-out-of-K coding
Xb = np.zeros((M,2*N))

## New attribute names list with one-out-of-K coding:
attributeNamesB = [None]*22
for i in range(len(attributeNames)):
    attributeNamesB[i] = (attributeNames[i]+"_50-100 perc.")
    attributeNamesB[i+11] = (attributeNames[i]+"_0-50 perc.")

## Change name for gender and is_patient
attributeNamesB[1] = attributeNames[1] + "_Female"
attributeNamesB[12] = attributeNames[1] + "_Male"
attributeNamesB[10] = attributeNames[10] + "_Sick"
attributeNamesB[21] = attributeNames[10] + "_Not-Sick"

## Creates list of medain values of data, herefrom the ones for gender and 
## is_patient won't really be used
medians = np.median(X,axis=0)



## For loop that evaluates value of X in comparison to median, if smaller then
## the attributes first collumn (they have 2 each here) is a one (the other
## is a zero), and if bigger then the second column is a one
for col in range(0,N):
    for val in range(0,M):
        #print(X[val,col])
        if X[val,col] < medians[col]:
            Xb[val,col+11] = 1
            #Xb[val,col*2+1] = 0
        elif X[val,col] >= medians[col]:
            #Xb[val,col*2] = 0
            Xb[val,col] = 1

## Gender and is_patient is adjusted because they are already binary..
Xb[:,1] = X[:,1]
Xb[:,12] = np.absolute(X[:,1]-1)

Xb[:,10] = np.absolute(X[:,10]-2)
Xb[:,21] = X[:,10]-1






