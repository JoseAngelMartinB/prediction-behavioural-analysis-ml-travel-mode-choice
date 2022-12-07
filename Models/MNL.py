#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for the Multinomial Logit Model using Biogeme.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""


# Import packages
from re import S
from ssl import VERIFY_X509_STRICT
import pandas as pd
import numpy as np
import os

# Import Biogeme modules
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import *


class MNL():

    def __init__(self, beta_params, utility_functions, model="logit"):
        """
        A Multinomial Logit Model
        """

        self.beta_params = beta_params
        self.utility_functions = utility_functions
        self.alternatives = list(utility_functions.keys())
        self.model = model


    def _preprocess_X(self, X):   
        # Fix column names errors
        X.columns = X.columns.str.replace(r'.', '_', regex=True)
        return X


    def fit(self, X, y=None):
        """
        Fit estimator.
        """
        X = X.copy()        
        X = self._preprocess_X(X)
        X["choice"] = y
        
        # Initializate Biogeme database
        database = db.Database('X_train', X)
        globals().update(database.variables)
        
        # Construct the model parameters
        for beta in self.beta_params:
            exec("{} = Beta('{}', 0, None, None, 0)".format(beta, beta), globals())

        # Define utility functions
        for utility_idx in self.utility_functions.keys():
            exec("V_{} = {}".format(utility_idx, self.utility_functions[utility_idx]), globals())

        # Assign utility functions to utility indices
        exec("V_dict = {}", globals())
        for utility_idx in self.utility_functions.keys():
            exec("V_dict[{}] = V_{}".format(utility_idx, utility_idx), globals())
        self.V_dict = V_dict

        # Associate the availability conditions with the alternatives
        exec("av = {}", globals())
        for utility_idx in self.utility_functions.keys():
            exec("av[{}] = 1".format(utility_idx), globals())
        
        # Definition of the model. This is the contribution of each
        # observation to the log likelihood function.
        logprob = models.loglogit(V_dict, av, choice)

        # Create the Biogeme object
        self.biogeme = bio.BIOGEME(database, logprob)
        self.biogeme.modelName = 'Biogeme-model'

        # Calculate the null log likelihood for reporting.
        self.biogeme.calculateNullLoglikelihood(av)

        # Estimate the parameters
        self.results = self.biogeme.estimate()

        # Get the results in a pandas table
        self.pandasResults = self.results.getEstimatedParameters()

        # Remove Biogeme files
        os.remove(self.results.data.htmlFileName) 
        os.remove(self.results.data.pickleFileName) 
        os.remove('__' + self.results.data.pickleFileName.replace('.pickle', '.iter',))


    def predict(self, X):
        """
        Predict class for X.
        """
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)


    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        X = X.copy()        
        X = self._preprocess_X(X)

        database = db.Database('X_test', X)
        globals().update(database.variables)

        # Update previous estimated beta variables
        globals().update(self.results.getBetaValues())

        V = np.zeros((X.shape[0], len(self.alternatives)))
        for alt in self.alternatives:
            V[:, alt] = database.valuesFromDatabase(eval(self.utility_functions[alt]))

        P = np.zeros((X.shape[0], len(self.alternatives)))
        for alt in self.alternatives:
            P[:, alt] = np.exp(V[:, alt]) / np.sum(np.exp(V), axis=1)

        proba = np.where(P > 0.00001, P, 0.00001)
        return proba
    
    
    def get_params(self, **kwargs):
        """
        Get parameters for this estimator.
        """
        # No need for hyperparameters
        pass 
    
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        # No need for hyperparameters
        pass