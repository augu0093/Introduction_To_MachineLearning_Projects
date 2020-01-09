#Complete classification scripts
import os
import random
import numpy
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
from import_file import *

#Logistic packages:
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from scipy import stats
random.seed(42)
classNames = np.asarray(df.columns)
M, N = X.shape
C = len(classNames)
X = stats.zscore(X)

######### 
#Baseline
def baselineClassification(X,y):
    unique, counts = numpy.unique(y, return_counts=True)
    sample_size = y.shape[0]
    error = 1-(counts[0]/sample_size)
    return error

  

#-----------------

def baselineClass(X,y):
    # Create crossvalidation partition for evaluation
    K_o_splits = 10
    outer_it = 0
    K_i_splits = 10

    eval_o_class = np.zeros((K_o_splits))


    #CV1 = model_selection.KFold(n_splits=K_o_splits,shuffle=True)
    #StratifiedKfold ensures that there is a reasonable percentage of each class in each split.
    CV1 = model_selection.StratifiedKFold(n_splits=K_o_splits, shuffle = True)

    
    #Outer k-fold split
    for train_index_o, test_index_o in CV1.split(X,y):
        print('Outer CV1-fold {0} of {1}'.format(outer_it+1,K_o_splits))
        X_train_o = X[train_index_o,:]
        y_train_o = y[train_index_o]
        X_test_o = X[test_index_o,:]
        y_test_o = y[test_index_o]
        
        #Inner validation loop
        inner_it = 0

        
        class_err = baselineClassification(X_train_o,y_train_o)
            
        
        eval_o_class[outer_it] = class_err

        
        
        outer_it+=1

    
    figure()
    boxplot(eval_o_class)
    xlabel('Baseline for classification')
    ylabel('Cross-validation error [%]')
    show()
    
#------------------------------
#baselineClass(X,y)   



def crossValidationLogReg():
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
            lowest_err = 100
            optimal_reg = 999
            for idx in range(model_count):
                reg_term = (0.01+idx*0.1)
                model = lm.logistic.LogisticRegression(C=reg_term,penalty='l2')
                model = model.fit(X_train_i, y_train_i)
                y_logreg = model.predict(X_test_i)
                current_err = 100*(y_logreg!=y_test_i).sum().astype(float)/len(y_test_i)
                
                summed_eval_i[idx] += current_err
            
            inner_it += 1
            
        
        eval_i = summed_eval_i * (len(X_test_i)/len(X_train_o))     
        idx = np.argmin(eval_i)
        reg_term = (0.01+idx*0.1)
        model = lm.logistic.LogisticRegression(C=reg_term,penalty='l2')
        model = model.fit(X_train_o, y_train_o)
        y_logreg = model.predict(X_test_o)
        current_err = 100*(y_logreg!=y_test_o).sum().astype(float)/len(y_test_o)
        
        eval_o[outer_it] = current_err
        optimal_lambda[outer_it] = reg_term
        
        
        outer_it+=1
        
    mode_reg, _= numpy.unique(optimal_lambda, return_counts=True)
    
    figure()
    boxplot(eval_o)
    xlabel('Logistic Regression')
    ylabel('Cross-validation error [%]')
    show()
    e_gen = np.sum(eval_o) * (len(X_test_o)/ len(X))
    print("Logistic regression generalization error: %f with %s and %f" % ((e_gen),'l2-norm',mode_reg[0]))

    
crossValidationLogReg()