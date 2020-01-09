import sklearn.tree
import sklearn.linear_model

from toolbox_02450 import *
# requires data from exercise 1.5.1
from import_file2 import *
alpha = 0.05
rho = 1/100

" Statistical evalutaion on regression models "

# Test errors from two-level crossvalidation
r_ANN = np.array([3.63, 8.48, 9.91, 7.03, 3.21, 14.83, 6.78, 2.73, 4.04, 5.55])
r_reg = np.array([7.79, 7.08, 7.99, 7.33, 7.71, 7.36, 6.93, 6.48, 7.28, 10.08])
r_base = np.array([7.72, 8.37, 8.18, 7.16, 7.99, 8.35, 7.78, 7.76, 8.01, 7.80])

print("\nStatistical evalutaion on regression models setupII")
###########################################################################
" Comparison of ANN and Linear regression "

r_ANN_Reg = r_ANN-r_reg
# Initialize parameters and run test appropriate for setup II
p_setupII_1, CI_setupII_1 = correlated_ttest(r_ANN_Reg, rho, alpha=alpha)

print("\n95 % CI for ANN_Reg and Lin_Reg", CI_setupII_1)
print("p-value for ANN_Reg and Lin_Reg comparison", p_setupII_1)

###########################################################################
" Comparison of ANN and Baseline "
r_ANN_Base = r_ANN-r_base
# Initialize parameters and run test appropriate for setup II
p_setupII_2, CI_setupII_2 = correlated_ttest(r_ANN_Base, rho, alpha=alpha)

print("\n95 % CI for ANN_Reg and Baseline", CI_setupII_2)
print("p-value for ANN_Reg and Baseline comparison", p_setupII_2)

###########################################################################
" Comparison of Linear regression and Baseline "
r_Reg_Base = r_reg-r_base
# Initialize parameters and run test appropriate for setup II
p_setupII_3, CI_setupII_3 = correlated_ttest(r_Reg_Base, rho, alpha=alpha)

print("\n95 % CI for Lin_Reg and Baseline", CI_setupII_3)
print("p-value for Lin_Reg and Baseline comparison", p_setupII_3)
###########################################################################



" Statistical evalutaion on classification models "

# Test errors from two-level crossvalidation
c_KNN = np.array([25.4, 25.4, 32.2, 28.8, 27.6, 24.6, 35.1, 28.1, 28.1, 29.8])
c_log = np.array([25.4, 37.3, 27.1, 23.7, 25.9, 29.8, 29.8, 29.8, 29.8, 28.1])
c_base = np.array([28.5, 28.5, 28.5, 28.5, 28.4, 28.5, 28.5, 28.5, 28.5, 28.5])

print("\nStatistical evalutaion on classification models setupII")
###########################################################################
" Comparison of KNN and Logistic regression "

c_KNN_Log = c_KNN-c_log
# Initialize parameters and run test appropriate for setup II
p_setupII_11, CI_setupII_11 = correlated_ttest(c_KNN_Log, rho, alpha=alpha)

print("\n95 % CI for KNN and Log_Reg", CI_setupII_11)
print("p-value for KNN and Log_Reg comparison", p_setupII_11)

###########################################################################
" Comparison of KNN and Baseline "

c_KNN_Base = c_KNN-c_base
# Initialize parameters and run test appropriate for setup II
p_setupII_22, CI_setupII_22 = correlated_ttest(c_KNN_Base, rho, alpha=alpha)

print("\n95 % CI for KNN and Baseline", CI_setupII_22)
print("p-value for KNN and Baseline comparison", p_setupII_22)

###########################################################################
" Comparison of Logistic regression and Baseline "

c_Log_Base = c_log-c_base
# Initialize parameters and run test appropriate for setup II
p_setupII_33, CI_setupII_33 = correlated_ttest(c_Log_Base, rho, alpha=alpha)

print("\n95 % CI for Log_Reg and Baseline", CI_setupII_33)
print("p-value for Log_Reg and Baseline comparison", p_setupII_33)
###########################################################################
