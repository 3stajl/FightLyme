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


from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PREDICTING

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 156, kernel_initializer = 'uniform', activation = 'relu', input_dim = 156))

# Adding the second hidden layer
classifier.add(Dense(units = 312, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(np.array(X_train), np.array(y_train), batch_size = 32, epochs = 1000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

score = classifier.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


