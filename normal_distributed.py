import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import numpy as np
from scipy.linalg import svd

os.chdir(r"C:\Users\Mads-\Documents\Universitet\3. Semester\02450 Introduktion til machine learning")
print(os.getcwd())
from import_file import import_and_processing_of_file
from import_file import check_package_setup

check_package_setup()
X,y,df = import_and_processing_of_file(r"C:\Users\Mads-\Documents\Universitet\3. Semester\02450 Introduktion til machine learning")

list_ = [0,3,4,5,6,7,8,9,10]
print(list_)




def normal_dis(X,y,df):
    attributeNames = np.asarray(df.columns)
    print(attributeNames)
    for j in range(3):
        fig = plt.figure()
        for i,index in enumerate(list_):
            mu, std = norm.fit(X[:,index])
            #Plot histogram
            plt.hist(X[:,index], bins=25, density=True, alpha=0.6)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            
            
            if(i<1):
                plt.title(attributeNames[index],fontsize=20)
            else:
                plt.title(attributeNames[index-1],fontsize=20)
            plt.show()
            
            
def normal_dis_log(X,y,df):
    attributeNames = np.asarray(df.columns)
    print(attributeNames)
    for j in range(3):
        for i,index in enumerate(list_):
            mu, std = norm.fit(X[:,index])
#            if index in ([3,4,5,6,7]):
#                X_new = np.log(X[:,index])
#                mu, std = norm.fit(X_new)
#            else: 
#                X_new = X[:,index]
#                mu, std = norm.fit(X_new)
            X_new = np.log(X[:,index])
            mu, std = norm.fit(X_new)
            #Plot histogram
            plt.hist(X_new, bins=25, density=True, alpha=0.6)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            
            
            if(i<1):
                plt.title(attributeNames[index],fontsize=20)
            else:
                plt.title(attributeNames[index-1],fontsize=20)
            plt.show()
            
def PCA_try():
    attributeNames = np.asarray(df.columns)
    N = len(y)
    M = len(attributeNames)
    Y = X - np.ones((N,1))*X.mean(axis=0)

    # PCA by computing SVD of Y
    U,S,V = svd(Y,full_matrices=False)
    
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 
    
    threshold = 0.9
    
    # Plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()
PCA_try()
#normal_dis(X,y,df)
#normal_dis(X,y,df)          

#print(df['gender']=='Male')

