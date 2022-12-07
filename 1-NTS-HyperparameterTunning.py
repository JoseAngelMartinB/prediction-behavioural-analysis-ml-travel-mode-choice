#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tunning of the Machine Learning methods for the NTS dataset.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

# Import packages
import pandas as pd  # For file input/output
import scipy
import time
import numpy as np
import pickle

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
from Models.XGBoost import XGBoost
#from MNL import MNL

import random
from collections import Counter, defaultdict


# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices



# Load the data
dataset_prefix = "nts_data"
data_dir = "Data/Datasets/preprocessed/"
adjusted_hyperparms_dir = "Data/adjusted-hyperparameters/"
dataset_name = dataset_prefix+"_train.csv"
mode_var = "mode_main"
individual_id = 'individual_id'
hyperparameters_file = dataset_prefix+"_hyperparameters"
crossval_pickle_file = data_dir+dataset_prefix+"_hyperparams_crossval.pickle"
reset_crossval_indices = 0 # Set to 0 for reproducibility of the experiment over multiple executions

scaled_fetures = ['distance', 'density', 'age', 'cars', 'bicycles', 'diversity', 'green', 'temp', 'precip', 'wind',
                  'income_cat', 'education_cat']

train = pd.read_csv(data_dir + dataset_name, sep=',')

# Divide the dataset into charasteristics and target variable
X = train.loc[:, train.columns != mode_var]
y = train[mode_var]

# Reduce dataset size to reduce computational cost of the hyperparameter estimation
_, X_sample, _, y_sample = train_test_split(X, y,
                                            stratify=y,
                                            test_size=0.25,
                                            random_state=2022)
X_sample = X_sample.reset_index(drop=True)
y_sample = y_sample.reset_index(drop=True)

# Extract the individual ID to later group observations using it
groups = np.array(X_sample[individual_id].values)
X_sample = X_sample.drop(columns=individual_id)

X_n_cols = X_sample.shape[1]
n_alternatives = y.nunique()


###########################   Set the parameters   ############################
# Number of iterations of the random search
n_iter = 1000
#n_iter = 10 # Uncomment for quick experiments
CV = 5 # Number of cross-validation

hyperparameters_file = hyperparameters_file +'_'+ str(n_iter) + '.csv'

# Set the hyperparameters search space
hyperparameters = {"RF" : {"n_estimators": hyperopt.pyll.scope.int(hyperopt.hp.quniform('n_estimators', 1, 200,1)),
                                     "max_features": hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_features', 2, X_sample.shape[1],1)),
                                     "max_depth": hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_depth', 3, 10,1)),
                                     "min_samples_leaf": hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_samples_leaf', 1, 20,1)),
                                     "min_samples_split": hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_samples_split', 2, 20,1)),
                                     "criterion": hyperopt.hp.choice('criterion', ["gini", "entropy"]),
                                     },
                   "SVM" : {"kernel": hyperopt.hp.choice('kernel', ['rbf']),
                            "gamma": hyperopt.hp.loguniform('gamma', -6.90, 0), # interval: [0.001, 1]
                            "nystrom_components": hyperopt.hp.choice('nystrom_components', [int(X_sample.shape[0]*(CV-1)/CV * 0.1)]),
                            "C": hyperopt.hp.loguniform('C', -2.30, 2.30), # interval: [0.1, 10]
                            "class_weight": hyperopt.hp.choice('class_weight', ['balanced']),
                            },
                   "NN"  : {"hidden_layer_sizes": hyperopt.pyll.scope.int(hyperopt.hp.quniform('hidden_layer_sizes', 10, 500,1)),
                            "activation" : hyperopt.hp.choice('activation', ["tanh"]),  # TODO: Consider other activation functions (ReLU, LeakyReLU, etc.)
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
                             "epochs": hyperopt.pyll.scope.int(hyperopt.hp.quniform('epochs', 50, 200,1)),
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
                        "CNN": CNN,
                        #"ResNet": ResNet,
                        "XGBoost": XGBoost,
                        }

STATIC_PARAMS = {'n_jobs': 12}

###############################################################################


## Obtain datasets for K-Fold cross validation (the same fold splits are used across all the iterations for all models)
train_indices = []
test_indices = []

try:
    train_indices, test_indices = pickle.load(open(crossval_pickle_file, "rb"))
    if reset_crossval_indices == 1: # Reset the indices
        raise FileNotFoundError
except (OSError, IOError) as e:
    print("Recomputing Cross-val indices...")
    for (train_index, test_index) in stratified_group_k_fold(X_sample, y_sample, groups, k=CV):
        train_indices.append(train_index)
        test_indices.append(test_index)
    pickle.dump([train_indices, test_indices], open(crossval_pickle_file, "wb"))


def objective(space):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # Ignore deprecation warnings

    # Create the classifier
    params = {**space, **STATIC_PARAMS}
    clf = model_type_to_class[classifier](**params)

    # Applying k-Fold Cross Validation
    loss = 0
    N_sum = 0

    for iteration in range(0, len(train_indices)):
        # Obtain training and testing data for this iteration (split of de k-Fold)
        X_train, X_test = X_sample.loc[train_indices[iteration]], X_sample.loc[test_indices[iteration]]
        y_train, y_test = y_sample.loc[train_indices[iteration]], y_sample.loc[test_indices[iteration]]

        # Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train[scaled_fetures])
        X_train.loc[:, scaled_fetures] = scaler.transform(X_train[scaled_fetures])
        X_test.loc[:, scaled_fetures] = scaler.transform(X_test[scaled_fetures])

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


#%%

# Read a previous adjusted-hyperparameters datafile or create a new one
try:
    adjusted_hyperparameters_file = pd.read_csv(adjusted_hyperparms_dir + hyperparameters_file, index_col=0)
    best_hyperparameters = adjusted_hyperparameters_file.to_dict()
except (OSError, IOError) as e:
    print("Creating new best_hyperparameters structure...")
    best_hyperparameters = {}

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
    best_hyperparameters_df.to_csv(adjusted_hyperparms_dir + hyperparameters_file, sep=',', index=True)
