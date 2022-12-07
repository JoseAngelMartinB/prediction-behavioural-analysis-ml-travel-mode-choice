#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for the Random Forest classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

# Import packages
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest():
    
    def __init__(self, **kwargs):
        """
        A random forest classifier.
        """
        # The special syntax **kwargs in function definitions in python is used
        # to pass a keyworded, variable-length argument list.
        
        # Create the RF classifier
        self.clf = RandomForestClassifier(**kwargs)
        
        
    def fit(self, X, y=None):
        """
        Fit estimator.
        """
        self.clf.fit(X, y)
        
        
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
        Get parameters for this estimator.
        """
        return self.clf.get_params(**kwargs)
    
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        return self.clf.set_params(**params)