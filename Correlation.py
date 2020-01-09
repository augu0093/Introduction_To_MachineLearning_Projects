
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)
from pandas.plotting import scatter_matrix
from import_file import *

## Extracts attribute names
attributeNames = np.asarray(df.columns)
attributeNames = np.delete(attributeNames,[1,10], axis=0)
M = len(attributeNames)
C = M

# requires data from exercise 4.2.1
#from ex4_2_1 import *
X = np.delete(X,[1,2],axis=1)
figure(figsize=(M+10,M+10))

## Calculates correlation matrix for dataset
corr = np.corrcoef(X, rowvar=False)
corr = np.around(corr, decimals = 2)

## Prints out scatterplots combined with correlation  values
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        #suptitle('This', fontsize=16)
        plt.title(corr[m1,m2])
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
#legend(attributeNames)
show()