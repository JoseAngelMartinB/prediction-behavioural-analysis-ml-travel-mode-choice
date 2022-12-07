#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for the Support Vector Machine classifier.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

# Import packages
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import pipeline
import numpy as np
from sklearn.kernel_approximation import Nystroem

class SVM():
    
    def __init__(self, **kwargs):
        """
        A Support Vector Machine classifier.
        """
        # The special syntax **kwargs in function definitions in python is used
        # to pass a keyworded, variable-length argument list.

        if "n_jobs" in kwargs:
            n_jobs = kwargs["n_jobs"]
            del kwargs["n_jobs"]

        if "kernel" in kwargs:
            self.kernel = kwargs["kernel"]
            del kwargs["kernel"]

        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]
            del kwargs["gamma"]

        if "nystrom_components" in kwargs:
            self.nystrom_components = kwargs["nystrom_components"]
            del kwargs["nystrom_components"]
        
        # Create the RF classifier
        #linear_SVC = LinearSVC(**kwargs)
        #calibrated_classifier = CalibratedClassifierCV(base_estimator=linear_SVC, cv=5, n_jobs=n_jobs)
        #nystrom_kernel = Nystroem(kernel=self.kernel, n_components=self.nystrom_components, gamma=self.gamma)
        #self.clf = pipeline.Pipeline(
        #    [("feature_map", nystrom_kernel), ("svm", calibrated_classifier)]
        #)

        linear_SVC = LinearSVC(**kwargs)
        nystrom_kernel = Nystroem(kernel=self.kernel, n_components=int(self.nystrom_components*4/5), gamma=self.gamma)
        kernel_SVC = pipeline.Pipeline(
            [("feature_map", nystrom_kernel), ("svm", linear_SVC)]
        )
        self.clf = CalibratedClassifierCV(base_estimator=kernel_SVC, cv=5, n_jobs=n_jobs)

        
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
        