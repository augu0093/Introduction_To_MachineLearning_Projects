#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:14:32 2019

@author: Johannesreiche
"""

import numpy as np
import pandas as pd
import os
import toolbox_02450

###Check for correct setup
def check_package_setup():
    print(toolbox_02450.__version__)
    print(os.getcwd())
    if os.getcwd() != r"C:\Users\Mads-\Documents\Universitet\3. Semester\02450 Introduktion til machine learning":
        os.chdir(r"C:\Users\Mads-\Documents\Universitet\3. Semester\02450 Introduktion til machine learning")

###################################
    

## General setup function. That imports datasets, handle errors, and performs one out of k encoding.    
def import_and_processing_of_file(filelocation):
    #Set working directory
    #set_working_directory(filelocation)
    #Import dataset. token = True if file was found, else False.
    df, attributeNames,token = import_file()    
    # handle errors
    if token != False:
        df_clean = error_handling(df,True)
        X,y,df = extract_X_y_df(df_clean)
        X = one_hot_encoding(X)
    return X,y,df

def import_file():
    filename = 'Indian Liver Patient Dataset (ILPD).csv'
    try: 
        df = pd.read_csv(filename)
        attributeNames = np.asarray(df.columns)
        return df, attributeNames, True
    except FileNotFoundError:
        print("File is not found")
        return 0, 0, False

    
def set_working_directory(filelocation):
    if os.getcwd() != filelocation:
        if filelocation != None:
            os.chdir(filelocation)  
            print("Directory is now: \n ",filelocation)

#Filelocation kan kopieres fra stifinderen. For at få rigtige format af stien, så brug r' foran stien.
    #Eksempelvis r' C:\Users\Mads-\Documents\Universitet\3. Semester\02450 Introduktion til machine learning'
def error_handling(dataframe,option):
    if option == True:
        #Replace nan values in alkphos column
        dataframe.replace("nan", np.NaN)
        # drop rows with missing values
        dataframe.dropna(inplace=True)
        return dataframe
    
def extract_X_y_df(df):
        row,column = np.shape(df)
        raw_data = np.asarray(df)
        raw_data1 = df.to_numpy()
        range_attribute = range(column)
        X = raw_data[:, range_attribute]
        y = raw_data[:,column-1]
        y1 = raw_data1[:,column-1]
        
        y_new = np.zeros(len(y))
        for i in range(len(y)):
            y_new[i] = y[i]
        y = y_new 
        return X,y, df
    
def one_hot_encoding(X):
    row,column = np.shape(X)
    Y = np.zeros((row,column))
    for j in range(column):
        for i in range(row):
            if j == 0:
                Y[i,j] = X[i,j]
            elif j == 1:
                if X[i,j] == 'Female':
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
            else:
                Y[i,j] = X[i,j]
    return Y


X,y,df = import_and_processing_of_file(r"")
attributeNames = np.asarray(df.columns)
#print(y*y)
## Import file
## Error handling
## One of out k-coding on gender