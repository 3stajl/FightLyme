# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 01:30:09 2019

@author: Dzejkot
"""

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

df = pd.read_csv('BoaData.csv')
pd.set_option('display.max_columns', None)

df.head()
len(df.columns)
#removing irrelevant columns
df = df.drop(['Unnamed: 64', 'Unnamed: 65'], axis='columns')

#shortening columns names
colnames = 'record_time,blood,infection,residence,start_treat'.split(',')
treatments = 'doxy,ilads,buhner,cowden,liposomal,other_herbs,vitaminD,supp,oil,kambo,plasma,sugar-free,gluten-free,dairy-free,bioresonance,antimicrobial,oxygen,cannabis_oil,binaural'.split(',')
colnames.extend(treatments)
stimulants = 'tobacco,alcohol,coffee,marijuana,other_stim'.split(',')
colnames.extend(stimulants)
colnames.extend(['num_antibiotics', 'method_antibiotics'])
symptoms = 'depression,unreal,irritability,sleep,conc,stupor,memory,stiffness,breath,fatigue,numb,strength,ache,arthralgia,headache,facial_muscle,abdominal,gastro_reflux,eye,ringing,light,sound,toothache,rash,hair,chest_pain,bladder,libido,weight,fever,Parkinson,muscle_decay'.split(',')
colnames.extend(symptoms)
colnames.append('effective')
df.columns = colnames

#dropping empty or with 'effective- NaN' column and duplicated rows:
df = df.drop([21,22,28,30,50,68,105,106,110,122])

#dealing  with Nan
df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: pd.factorize(x)[0])

X=df[['start_treat','doxy','ilads','buhner','cowden','liposomal','other_herbs','vitaminD','supp','oil','sugar-free','gluten-free','dairy-free','bioresonance','antimicrobial','oxygen','cannabis_oil','binaural','tobacco','alcohol','coffee','marijuana','other_stim','num_antibiotics','method_antibiotics']].values
y=df['effective'].values

#Future Scaling
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='NaN',strategy='mean')
imputer = imputer.fit(X)
X = imputer.transform(X)

imputerY = SimpleImputer(missing_values='NaN',strategy='mean')
imputerY = imputerY.fit(y)
y = imputer.transform(y)
'''




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


import keras
from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y)


'''
model = DecisionTreeRegressor(max_depth=10)
cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
'''



model = RandomForestRegressor(max_depth=15, n_estimators=25, n_jobs=8)
model.fit(X,y_binary)
feats = {}
for feature, importance in zip(df[['start_treat','doxy','ilads','buhner','cowden','liposomal','other_herbs','vitaminD','supp','oil','sugar-free','gluten-free','dairy-free','bioresonance','antimicrobial','oxygen','cannabis_oil','binaural','tobacco','alcohol','coffee','marijuana','other_stim','num_antibiotics','method_antibiotics']], model.feature_importances_):
    feats[feature] = importance #add the name/value pair 
scores = cross_val_score(model, X, y_binary, cv=3, scoring='neg_mean_absolute_error')
np.mean(scores), np.std(scores)



#model.fit(X,y_binary)
MostImportant = model.feature_importances_
'''
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(df.columns, model.feature_importances_):
   feats[feature] = importance #add the name/value pair 
'''

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
model.predict(X)
y_pred = model.predict(X)

'''
import keras
from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size = 0.2, random_state = 0)

'''

'''
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 156, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))

# Adding the second hidden layer
classifier.add(Dense(units = 312, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(np.array(X_train), np.array(y_train), batch_size = 15, epochs = 1000)
# Part 3 - Making predictions and evaluating the model
'''

'''
# Predicting the Test set results
y_pred = classifier.predict(X_test)

score = classifier.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

'''

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 156, kernel_initializer = 'uniform', activation = 'relu', input_dim = 25))

# Adding the second hidden layer
classifier.add(Dense(units = 312, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(np.array(X_train), np.array(y_train), batch_size = 15, epochs = 1000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

score = classifier.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




