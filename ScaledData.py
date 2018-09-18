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
dataset = pd.read_csv('LymeSurveyPreproc.csv')
X = pd.DataFrame(dataset.iloc[:, 2:-1].values)
X = X.drop([26], axis='columns')

y = pd.DataFrame(dataset.iloc[:, 28].values)

#Taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis = 0)
imputer = imputer.fit(X.iloc[:, 25:26])
X.iloc[:, 25:26] = imputer.transform(X.iloc[:, 25:26])

imputer2 = Imputer(missing_values='NaN',strategy='mean', axis = 0)
imputer2 = imputer2.fit(X.iloc[:, 0:1])
X.iloc[:, 0:1] = imputer.transform(X.iloc[:, 0:1])


imputerY = Imputer(missing_values='NaN',strategy='mean', axis = 0)
imputerY = imputerY.fit(y.iloc[:, 0:1])
y.iloc[:, 0:1] = imputer.transform(y.iloc[:, 0:1])

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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



