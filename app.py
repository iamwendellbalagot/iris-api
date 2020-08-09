#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from ModelBuilder.modelBuilder import ModelBuilder


# In[ ]:


app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    @staticmethod
    def post():
        data = request.get_json()
        sepal_length = data['sepal_length']
        sepal_width = data['sepal_width']
        petal_length = data['petal_length']
        petal_width = data['petal_length']
        #create a instance of modelBuilder
        model = ModelBuilder()
        
        #if there's no model, then train
        if not os.path.isfile('models/iris.model'):
            model.fit()
        else:
            prediction = model.predict([sepal_length, sepal_width, petal_length, petal_width])
            response = jsonify({
                'Input': {
                    'sepal_length' : sepal_length,
                    'sepal_width' : sepal_width,
                    'petal_length' : petal_length,
                    'petal_width' : petal_width
                },
                'Prediction': prediction
            })
            return response
            
api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(debug=True)

