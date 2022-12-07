# -*- coding: utf-8 -*-
"""
Wrapper for the Convolutional Neural Network (CNN) classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas

A class for building, fitting and predicting using a Convolutional Neural Network (CNN)
"""

#Import keras packages for DL models
import numpy as np
import keras as keras
from keras import layers
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD


def create_CNN(input_dim, output_dim):
    # Create the CNN classifier
    clf = Sequential()
    clf.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_dim,1)))
    clf.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
    clf.add(Dropout(0.5))
    clf.add(layers.MaxPooling1D(pool_size=2))
    clf.add(layers.Flatten())
    clf.add(Dense(100, activation='relu'))
    clf.add(Dense(output_dim, activation='softmax'))
    
    # Compile the model
    clf.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #print(clf.summary())

    return clf

class CNN():
    
    def __init__(self, **kwargs):
        """
        A Convolutional Neural Network (CNN).
        """   
        
        self.clf = KerasClassifier(build_fn=create_CNN,
                                   input_dim = kwargs.get('input_dim'),
                                   output_dim = kwargs.get('output_dim'), 
                                   epochs = kwargs.get('epochs'),
                                   batch_size = kwargs.get('batch_size'))   
        
        self.num_classes = kwargs.get('output_dim')
        
    
    def fit(self, X, y = None):
        """
        Fit CNN.
        """

        # Fix for CNN
        x1 = X.shape[0]
        x2 = X.shape[1]
        X = X.to_numpy().reshape(x1, x2, 1)

        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        self.clf.history_DNN = self.clf.fit(X,y) 
        
    def predict(self, X):
        """
        Predict class for X.
        """
        # Fix for CNN
        x1 = X.shape[0]
        x2 = X.shape[1]
        X = X.to_numpy().reshape(x1, x2, 1)
        return(self.clf.predict(X))


    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        # Fix for CNN
        x1 = X.shape[0]
        x2 = X.shape[1]
        X = X.to_numpy().reshape(x1, x2, 1)
        proba = self.clf.predict_proba(X)
        proba = np.where(proba > 0.00001, proba, 0.00001)
        return(proba)

        
    def get_params(self, **kwargs):
        """
        Get parameters for the CNN.
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
        
        return self.clf.history_DNN