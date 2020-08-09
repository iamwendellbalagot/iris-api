#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier


# In[23]:


class ModelBuilder:
    """
        desription here
    """
    def __init__(self):
        self.df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
        try:
            self.model = joblib.load('models/iris.model')
        except:
            self.model = None
    def fit(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        
        self.model = RandomForestClassifier(max_depth=5).fit(X,y)
        joblib.dump(self.model, 'models/iris.model')
        print('Finished Training')
    def predict(self, params):
        
        if not os.path.isfile('models/iris.model'):
            print('Please fit the model before calling using predict method!')
        if len(params) !=4:
            print('Expected 4 pramaters, but got {0}'.format(len(params)))
        else:
            prediction = self.model.predict([params])
        return prediction[0]

