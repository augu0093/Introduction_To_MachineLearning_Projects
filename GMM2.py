#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:10:35 2019

@author: Johannesreiche
"""

from matplotlib.pyplot import figure, plot, legend, xlabel, show, savefig
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from toolbox_02450 import clusterval
from scipy import stats
import random
from import_file2 import *
random.seed(42)

attributeNames = ['age', 'gender', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins',
 'albumin','ag_ratio', 'sgpt', 'sgot', 'alkphos', 'is_patient']
classNames = ['Is patient', 'Not a patient']
N, M = X.shape
C = len(classNames)
y = y.astype(dtype='uint8')


# Range of K's to try
KRange = range(1,15)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

cds = []        
# extract cluster centroids (means of gaussians)

cls = []
for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        cds.append(gmm.means_)
        cls.append(gmm.predict(X))
        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)
            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()

# Plot results

figure(1); 
plot(KRange, CVE,'-ok')
xlabel('K')
savefig('GMM_with_y.png')
show()

Rand = np.zeros((1,))
Jaccard = np.zeros((1,))
NMI = np.zeros((1,))
Rand[0], Jaccard[0], NMI[0] = clusterval(y,cls[8]) 

print("\n Rand: ", Rand, "Jaccard: ", Jaccard, "NMI: ", NMI)


print("Lowest score: ", np.min(CVE))
print("Highest score: ", np.max(CVE))

print("Cluster means: ", cds[8])