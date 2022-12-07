# -*- coding: utf-8 -*-
"""
Wrapper for the Multilayer Perceptron (MLP) classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas

A class for building, fitting and predicting using a Multilayer Perceptron (MLP)
"""

# Import packages 
from sklearn.neural_network import MLPClassifier
import numpy as np

class NN():
    
    def __init__(self, **kwargs):
        """
        A Multilayer Perceptron (MLP).
        """
        # The special syntax **kwargs in function definitions in python is used
        # to pass a keyworded, variable-length argument list.

        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
            del kwargs["n_jobs"]
        
        # Create the RF classifier
        self.clf = MLPClassifier(**kwargs)
        
        
    def fit(self, X, y = None):
        """
        Fit MLP.
        """ 
        self.clf.history_NN = self.clf.fit(X,y) 
        
        
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
        return self.clf.history_NN
        