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

df = pd.read_csv('BoaData.csv')
df.head()