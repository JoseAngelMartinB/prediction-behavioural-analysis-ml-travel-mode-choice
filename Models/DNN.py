# -*- coding: utf-8 -*-
"""
Wrapper for the Deep Neural Network classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas

A class for building, fitting and predicting using a Deep Multilayer Perceptron (Deep MLP)
"""

#Import keras packages for Deep MLP model
import numpy as np
import keras as keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD


def create_DMLP(input_dim, output_dim, depth, width, drop):
    # Create the Deep MLP classifier
    clf = Sequential()
    clf.add(Dense(width, activation='relu', input_dim = input_dim))
    for i in range (1,depth):
        clf.add(Dense(width, activation='relu'))
        clf.add(Dropout(drop))
    clf.add(Dense(output_dim, activation='softmax'))
    
    #clf.add(Dense(64, activation='relu', input_dim = input_dim))
    #clf.add(Dropout(0.5))
    #clf.add(Dense(64, activation='relu'))
    #clf.add(Dropout(0.5))
    #clf.add(Dense(64, activation='relu'))
    #clf.add(Dropout(0.5))
    #clf.add(Dense(output_dim, activation='softmax'))

    #Compile the model
    clf.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
        
    #print(clf.summary())
    return clf

class DNN():
    
    def __init__(self, **kwargs):
        """
        A Deep Multilayer Perceptron (Deep MLP).
        """
        
        self.clf = KerasClassifier(build_fn=create_DMLP,
                                   input_dim = kwargs.get('input_dim'),
                                   output_dim = kwargs.get('output_dim'), 
                                   depth  = kwargs.get('depth'),
                                   width  = kwargs.get('width'),
                                   drop   = kwargs.get('drop'),
                                   epochs = kwargs.get('epochs'),
                                   batch_size = kwargs.get('batch_size')) 
        
        self.num_classes = kwargs.get('output_dim')
        
    
    def fit(self, X, y = None):
        """
        Fit Deep MLP.
        """
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        self.clf.history_DNN = self.clf.fit(X,y,verbose=0) 
        
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
        Get parameters for the MLP.
        """

        return self.clf.get_params(**kwargs)
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        return self.clf.set_params(**params)
    

    def get_history(self, **kwargs):
        """
        Get history results of the MLP.
        """
        
        return self.clf.history_DNN