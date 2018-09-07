# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:11:12 2018

@author: Dzejkot
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('!BoreliozAnaliza.csv')
X = pd.DataFrame(dataset.iloc[:, :-3].values)
y = pd.DataFrame(dataset.iloc[:, 63].values)

#Taking care of missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis = 0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])
'''

#Encoding categorical data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder() 
y = labelencoder_X.fit_transform(y)
'''