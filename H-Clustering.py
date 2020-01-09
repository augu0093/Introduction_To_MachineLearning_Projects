#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:57:19 2019

@author: Johannesreiche
"""

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
from scipy.io import loadmat
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from toolbox_02450 import clusterval
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
import scipy.linalg as linalg
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from import_file2 import *
from scipy import stats

" Perform PCA for the 2D-plot "
def PCA(X):
    Xc = X[:,0:-1]
    Xc = stats.zscore(X);  
    U,S,V = linalg.svd(Xc,full_matrices=False)
    V = V.T
    Z = Xc @ V
    return Z,X

y = y.astype(dtype='uint8')
Z1,X1 = PCA(X)

" Perform hierarchical clustering on data matrix "
" with linkage function = complete and euclidean distance measure "

Method = 'complete'
Metric = 'euclidean'

Z = linkage(X1,method=Method,metric=Metric)
Rand = np.zeros((1,))
Jaccard = np.zeros((1,))
NMI = np.zeros((1,))

" Compute and display clusters by thresholding the dendrogram "
Maxclust = 9
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
Rand[0], Jaccard[0], NMI[0] = clusterval(y,cls) 
print("\n Rand: ", Rand, "Jaccard: ", Jaccard, "NMI: ", NMI)
clusterplot(Z1, cls.reshape(cls.shape[0],1), y=y)
xlabel('PC1')
ylabel('PC2')
plt.savefig('H-clustering_PCA.png')

" Display dendrogram "
max_display_levels=8
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)
plt.savefig('H-clustering_dendo.png')
show()

print('Did H-clustering')