# -*- coding: utf-8 -*-
"""
Wrapper for the ResNet classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
A class for building, fitting and predicting using a ResNet Deep Neural Network
"""

#Import keras packages for DL models
from turtle import width
import numpy as np
import keras as keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Add
from keras.wrappers.scikit_learn import KerasClassifier
from keras.engine.topology import Layer

import tensorflow as tf
jobs = 12 # it means number of cores
from keras import backend as K
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=jobs,
                                                                                            inter_op_parallelism_threads=jobs)))
tf.config.run_functions_eagerly(True)


# Define the residual block as a new layer
class Residual(Layer):
    def __init__(self, width, drop, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.width = width
        self.drop = drop

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = x
        x = BatchNormalization()(x)
        x = Dense(self.width, activation='relu')(x)
        x = Dropout(self.drop)(x)
        x = Dense(self.width)(x)
        x = Dropout(self.drop)(x)
        residual = Add()([x, first_layer])
        x = Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def create_ResNet(input_dim, output_dim, n_residual, width, drop):
    # Create the ResNet topology
    # Based on: Revisiting Deep Learning Models for Tabular Data. Yury Gorishniy et al.
    clf = Sequential()
    clf.add(Dense(width, activation='relu', input_dim = input_dim))
    # Add residual layers
    for i in range (1, n_residual):
        clf.add(Residual(width, drop))
    clf.add(Dense(output_dim, activation='softmax'))

    #Compile the model
    clf.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
        
    #print(clf.summary())
    return clf



class ResNet():
    
    def __init__(self, **kwargs):
        """
        A Residual Neural Network.
        """   
        
        self.clf = KerasClassifier(build_fn   = create_ResNet,
                                   input_dim  = kwargs.get('input_dim'),
                                   output_dim = kwargs.get('output_dim'), 
                                   n_residual  = kwargs.get('n_residual'),
                                   width  = kwargs.get('width'),
                                   drop   = kwargs.get('drop'),
                                   epochs     = kwargs.get('epochs'),
                                   batch_size = kwargs.get('batch_size'))   
        
        self.num_classes = kwargs.get('output_dim')
        
    
    def fit(self, X, y = None):
        """
        Fit ResNet.
        """
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        self.clf.history_ResNet = self.clf.fit(X,y,verbose=0) 
        
    def predict(self, X):
        """
        Predict class for X.
        """
        return(self.clf.predict(X))


    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        proba = self.clf.predict_proba(X)
        proba = np.where(proba > 0.00001, proba, 0.00001)
        return(proba)

        
    def get_params(self, **kwargs):
        """
        Get parameters for the ResNet.
        """
        return self.clf.get_params(**kwargs)
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        return self.clf.set_params(**params)

    def get_history(self, **kwargs):
        """
        Get history results of the CNN.
        """
        
        return self.clf.history_ResNet
