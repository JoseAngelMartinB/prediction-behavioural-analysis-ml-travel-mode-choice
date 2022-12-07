#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tunning of the Machine Learning methods for the synthetic datasets.

Authors:
- José Ángel Martín-Baos
- Ricardo García-Ródenas
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
"""

# Import packages
import pandas as pd  # For file input/output
import scipy
import time
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hyperopt
import warnings

# Import the Classification models
from Models.RandomForest import RandomForest
from Models.SVM import SVM
from Models.DNN import DNN
from Models.CNN import CNN
from Models.NN import NN
from Models.ResNet import ResNet
from Models.XGBoost import XGBoost
#from MNL import MNL

import random
from collections import Counter, defaultdict


# Load the data
simulation_dir = "Data/Simulated_V2/"  #"Data/Simulated/"
samples_subpath = "samples/"
adjusted_hyperparms_dir = "Data/adjusted-hyperparameters/SimulatedData_V2/"  #"Data/adjusted-hyperparameters/SimulatedData/"
mode_var = "y"
random_samples = 30
X_n_cols = 6
n_alternatives = 3
sample_size = 10000
dataframe_reduction = 0.01
hyperparameters_file = "SimulatedData_hyperparameters"
reset_crossval_indices = 0 # Set to 0 for reproducibility of the experiment over multiple executions

# Read the simulation summary file and extract the different models
summary = pd.read_csv(simulation_dir + 'Summary.csv')
models = summary[["model", "function", "beta_a", "beta_b"]].drop_duplicates(keep='first', inplace=False)


###########################   Set the parameters   ############################
# Number of iterations of the random search
n_iter = 1000
#n_iter = 1 # Uncomment for quick experiments
CV = 5 # Number of cross-validation

hyperparameters_file = hyperparameters_file +'_'+ str(n_iter) + '.csv'

# Set the hyperparameters search space
hyperparameters = {"RF" : {"n_estimators": hyperopt.pyll.scope.int(hyperopt.hp.quniform('n_estimators', 1, 200,1)),
                                     "max_features": hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_features', 2, X_n_cols,1)),
                                     "max_depth": hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_depth', 3, 10,1)),
                                     "min_samples_leaf": hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_samples_leaf', 1, 20,1)),
                                     "min_samples_split": hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_samples_split', 2, 20,1)),
                                     "criterion": hyperopt.hp.choice('criterion', ["gini", "entropy"]),
                                     },
                   "SVM" : {"kernel": hyperopt.hp.choice('kernel', ['rbf']),
                            "gamma": hyperopt.hp.loguniform('gamma', -6.90, 0), # interval: [0.001, 1]
                            "nystrom_components": hyperopt.hp.choice('nystrom_components', [int(sample_size*(CV-1)/CV * 0.025)]),
                            "C": hyperopt.hp.loguniform('C', -2.30, 2.30) # interval: [0.1, 10]
                            },
                   "NN"  : {"hidden_layer_sizes": hyperopt.pyll.scope.int(hyperopt.hp.quniform('hidden_layer_sizes', 10, 500,1)),
                            "activation" : hyperopt.hp.choice('activation', ["tanh"]), # TODO: Consider other activation functions (ReLU, LeakyReLU, etc.)
                            "solver" : hyperopt.hp.choice('solver', ["lbfgs","sgd","adam"]),
                            "learning_rate_init": hyperopt.hp.uniform('learning_rate_init', 0.0001, 1),
                            "learning_rate" : hyperopt.hp.choice('learning_rate', ["adaptive"]),
                            "max_iter": hyperopt.hp.choice('max_iter', [10000000]),
                            "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                            "tol" : hyperopt.hp.choice('tol', [1e-3]),
                           },
                   "DNN"  : {"input_dim": hyperopt.hp.choice('input_dim', [X_n_cols]),
                             "output_dim": hyperopt.hp.choice('output_dim', [n_alternatives]),
                             "depth": hyperopt.hp.choice('depth', [2,3,4,5,6,7,8,9,10]),
                             "width": hyperopt.hp.choice('width', [25,50,100,150,200]),
                             "drop": hyperopt.hp.choice('drop', [0.1, 0.01, 1e-5]),
                             # TODO: Consider adding the activation functions for the hidden layers (thanh, ReLU, LeakyReLU, etc.)
                             "epochs": hyperopt.pyll.scope.int(hyperopt.hp.quniform('epochs', 50, 200,1)),
                             "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                           },
                   "CNN"  : {"input_dim": hyperopt.hp.choice('input_dim', [X_n_cols]), # nº columnas
                             "output_dim": hyperopt.hp.choice('output_dim', [n_alternatives]),
                             "epochs": hyperopt.pyll.scope.int(hyperopt.hp.quniform('epochs', 50, 200,1)),
                             "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                           },
                    "ResNet":{"input_dim": hyperopt.hp.choice('input_dim', [X_n_cols]),
                             "output_dim": hyperopt.hp.choice('output_dim', [n_alternatives]),
                             "n_residual": hyperopt.hp.choice('n_residual', [2,3,4]),
                             "width": hyperopt.hp.choice('width', [25,50,100,150,200]),
                             "drop": hyperopt.hp.choice('drop', [0.1, 0.01, 1e-5]),
                             "epochs": hyperopt.pyll.scope.int(hyperopt.hp.quniform('epochs', 50, 100,1)),
                             "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                           },
                   "XGBoost" : {'max_depth': hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_depth', 1, 14,1)),
                                'gamma': hyperopt.hp.loguniform('gamma', -9.21, 1.61), # interval: [0.0001, 5]
                                'min_child_weight': hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_child_weight', 1, 100, 1)),
                                'max_delta_step': hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_delta_step', 0, 10, 1)),
                                'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
                                'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
                                'colsample_bylevel': hyperopt.hp.uniform('colsample_bylevel', 0.5, 1),
                                'reg_alpha': hyperopt.hp.loguniform('reg_alpha', -9.21, 2.30), # interval: [0.0001, 10]
                                'reg_lambda': hyperopt.hp.loguniform('reg_lambda', -9.21, 2.30), # interval: [0.0001, 10]
                                'n_estimators': hyperopt.pyll.scope.int(hyperopt.hp.quniform('n_estimators', 1, 6000, 1)),
                            },
                   }


model_type_to_class = { "RF": RandomForest,
                        "SVM": SVM,
                        "NN": NN,
                        "DNN": DNN,
                       #"ResNet": ResNet,
                        "XGBoost": XGBoost,
                        }

STATIC_PARAMS = {'n_jobs': 12}

###############################################################################



def objective(space):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # Ignore deprecation warnings

    # Create the classifier
    params = {**space, **STATIC_PARAMS}
    clf = model_type_to_class[classifier](**params)

    # Applying k-Fold Cross Validation
    loss = 0
    N_sum = 0

    for iteration in range(0, len(train_indices)):
        # Obtain training and testing data for this iteration (split of de k-Fold) by joining datasets
        X_train = X_crossval_train[iteration]
        X_test = X_crossval_test[iteration]
        y_train = y_crossval_train[iteration]
        y_test = y_crossval_test[iteration]

        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)

        # Cross-Entropy Loss
        sum = 0
        i = 0
        for sel_mode in y_test.values:
            sum = sum + np.log(proba[i,sel_mode])
            i += 1
        N = i - 1
        loss += -sum  # Original: (-sum/N) * N
        N_sum += N

    loss = loss / N_sum
    return {'loss': loss, 'status': hyperopt.STATUS_OK}


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

## Obtain datasets for K-Fold cross validation (the same fold splits are used across all the iterations for all models)
train_indices = []
test_indices = []
splits = list(split(list(range(0, random_samples)), CV))
for j in range(0, CV):
    training_splits = list(range(0,CV))
    training_splits.remove(j)
    aux_indices = []
    for k in training_splits:
        aux_indices = aux_indices + splits[k]
    train_indices.append(aux_indices)
    test_indices.append(splits[j])


total_it = models.shape[0]
increment = total_it/25
count = 0
for index, model in models.iterrows():
    print(
        "[" + "="*round((count + 1) / increment) + " "*round((total_it-count-1)/increment) + "]   " + str(
            round((count+1)/total_it*100, 2)) + "%")
    print("Loading {}-{}-{}-{}".format(model["model"],model["function"],model["beta_a"],model["beta_b"]))
    count += 1

    # Load all the datasets generated for the same model
    simulations = []
    for sim in range(index, index+random_samples):
        simulations.append(pd.read_csv(simulation_dir + samples_subpath + summary.loc[sim, "name_train"]))

    # Construct the training and test dataframes for each fold
    X_crossval_train = []
    X_crossval_test = []
    y_crossval_train = []
    y_crossval_test = []
    for iteration in range(0, len(train_indices)):
        frames_train = []
        for f in range(0, len(train_indices[iteration])):
            frames_train.append(simulations[train_indices[iteration][f]])
        X_aux = pd.concat(frames_train, ignore_index=True)
        _, X_sample, _, y_sample = train_test_split(X_aux.loc[:, X_aux.columns != mode_var], X_aux["y"],
                                                    stratify=X_aux["y"],
                                                    test_size=dataframe_reduction,
                                                    random_state=2022)
        X_sample = X_sample.reset_index(drop=True)
        y_sample = y_sample.reset_index(drop=True)
        X_crossval_train.append(X_sample)
        y_crossval_train.append(y_sample)

        frames_test = []
        for f in range(0, len(test_indices[iteration])):
            frames_test.append(simulations[test_indices[iteration][f]])
        X_aux = pd.concat(frames_test, ignore_index=True)
        _, X_sample, _, y_sample = train_test_split(X_aux.loc[:, X_aux.columns != mode_var], X_aux["y"],
                                                    stratify=X_aux["y"],
                                                    test_size=dataframe_reduction,
                                                    random_state=2022)
        X_sample = X_sample.reset_index(drop=True)
        y_sample = y_sample.reset_index(drop=True)
        X_crossval_test.append(X_sample)
        y_crossval_test.append(y_sample)


    # Read a previous adjusted-hyperparameters datafile or create a new one
    model_dir = "{}-{}-{}-{}/".format(model["model"],model["function"],model["beta_a"],model["beta_b"])
    try:
        adjusted_hyperparameters_file = pd.read_csv(adjusted_hyperparms_dir + model_dir + hyperparameters_file, index_col=0)
        best_hyperparameters = adjusted_hyperparameters_file.to_dict()
    except (OSError, IOError) as e:
        print("Creating new best_hyperparameters structure...")
        best_hyperparameters = {}
        # Create model dir if it does not exists
        if not os.path.exists(adjusted_hyperparms_dir + model_dir):
            os.makedirs(adjusted_hyperparms_dir + model_dir)


    for classifier in model_type_to_class.keys():
        print("\n--- %s" % classifier)
        time_ini = time.perf_counter()

        trials = hyperopt.Trials()
        best_classifier = hyperopt.fmin(fn=objective,
                                        space=hyperparameters[classifier],
                                        algo=hyperopt.tpe.suggest,
                                        max_evals=n_iter,
                                        trials=trials)

        elapsed_time = time.perf_counter() - time_ini
        print("Tiempo ejecucción: %f" % elapsed_time)

        best_hyperparameters[classifier] = best_classifier
        best_hyperparameters[classifier]['_best_loss'] = trials.best_trial["result"]["loss"]
        best_hyperparameters[classifier]['_best_GMPCA'] = np.exp(-trials.best_trial["result"]["loss"])
        best_hyperparameters[classifier]['_elapsed_time'] = elapsed_time

        # Partially store the results (the best hyperparameters)
        best_hyperparameters_df = pd.DataFrame(best_hyperparameters)
        best_hyperparameters_df.to_csv(adjusted_hyperparms_dir + model_dir + hyperparameters_file, sep=',', index=True)



