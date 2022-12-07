#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the synthetic datasets used in the paper.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

import sys
import os
import numpy as np
import pandas as pd


###########################   Set the parameters   ############################
simulation_dir = "../Data/Simulated/"
samples_subpath = "samples/"

random_samples = 30
sample_size = 10000
n_alternatives = 3
attribute_range = [0, 1]
logit_model_list = ["logit", "probit"]
lambda_list = [np.sqrt(1/12)]
#SIGMA = np.sqrt(2/5)
function_list = ["linear", "CD"] # The utility functions
beta_a = 1
beta_b_list = [1, 2, 0.5]
###############################################################################


def computeChoice(U):
    y = np.argmax(U, axis=1)
    return y


def GenerateSample(model, lamb, function, beta_b):
    Min = attribute_range[0]
    Max = attribute_range[1]

    # Generate a matrix containing the simulated attributes a and b for all alternatives
    A = np.round(np.random.uniform(Min, Max, size=(sample_size, n_alternatives)),6)
    B = np.round(np.random.uniform(Min, Max, size=(sample_size, n_alternatives)),6)

    # Construct the deterministic part of the utility function
    V = 0
    if function == "linear":
        V = beta_a*A + beta_b*B
    elif function == "CD":
        V = np.power(A, beta_a) * np.power(B, beta_b)
    elif function == "minimum":
        V = np.minimum(beta_a*A, beta_b*B)

    # Generate the stochastic part of the utility function
    if model == "logit":
        epsilon = np.random.gumbel(loc=0.0, scale=lamb, size=(sample_size, n_alternatives))
    elif model == "probit":
        epsilon = np.random.normal(loc=0.0, scale=lamb, size=(sample_size, n_alternatives))
    else:
        raise Exception('Unknown value model = %s' % model)

    # Utility function
    U = V + epsilon

    # Compute the vector of selected alternatives
    y = computeChoice(U)
    y_deterministic = computeChoice(V)

    # Generate the data matrix
    data = np.concatenate((A, B, np.asmatrix(y).transpose()),
                          axis=1)

    # Generate the colum names
    cols = []
    for alt in range(0, n_alternatives):
        cols.append('a.'+str(alt))
    for alt in range(0, n_alternatives):
        cols.append('b.'+str(alt))
    cols.append('y')

    # Combine the generated data
    simulated_data = pd.DataFrame(data = data,
                                  columns = cols)
    simulated_data['y'] = simulated_data.y.astype('int64') # Convert y to int

    maximum_accuracy = (((y == y_deterministic).sum() / sample_size)  * 100).round(2)

    simulation = {'data': simulated_data,
                  'maximum_accuracy': maximum_accuracy
                  }

    return simulation




##############################   Main program   ###############################
if not os.path.exists(simulation_dir+samples_subpath):
    os.makedirs(simulation_dir+samples_subpath)

summary = pd.DataFrame(data=None,
                       columns=['name_train', 'name_test',
                                'maximum_accuracy_train', 'maximum_accuracy_test',
                                'random_samples', 'sample_size', 'n_alternatives',
                                'model', 'lambda', 'function', 'beta_a', 'beta_b'])

it = 0
total_it = random_samples*len(logit_model_list)*len(lambda_list)*len(function_list)*len(beta_b_list)
increment = total_it/50

for model in logit_model_list:
    for lamb in lambda_list:
        for function in function_list:
            for beta_b in beta_b_list:
                for i in range(0, random_samples):
                    it += 1
                    sys.stdout.write("\r[" + "=" * round(it / increment) +  " " * round((total_it - it)/ increment) +
                                     "]   " +  str(round(it / total_it * 100, 2)) + "%")
                    sys.stdout.flush()
                    sample_train = GenerateSample(model, lamb, function, beta_b)
                    sample_test = GenerateSample(model, lamb, function, beta_b)

                    train_file_name = "train_%d.csv" % it
                    test_file_name = "test_%d.csv" % it

                    sample_summary = {'name_train': train_file_name,
                                      'name_test': test_file_name,
                                      'maximum_accuracy_train': sample_train['maximum_accuracy'],
                                      'maximum_accuracy_test': sample_test['maximum_accuracy'],
                                      'random_samples': random_samples,
                                      'sample_size': sample_size,
                                      'n_alternatives': n_alternatives,
                                      'model': model,
                                      'lambda': lamb,
                                      'function': function,
                                      'beta_a': beta_a,
                                      'beta_b': beta_b
                                      }

                    summary = summary.append(pd.DataFrame(sample_summary,
                                                          index=[it-1]))

                    # Store the samples
                    sample_train['data'].to_csv(simulation_dir+samples_subpath+train_file_name,
                                                index=False, sep=',')
                    sample_test['data'].to_csv(simulation_dir+samples_subpath+test_file_name,
                                               index=False, sep=',')


# Store the summary file
summary.to_csv(simulation_dir+'Summary.csv', index=False, sep=',')
