#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess the Optima dataset.

Reference:
Atasoy, B., A. Glerum, and M. Bierlaire (2013). Attitudes towards mode choice in Switzerland. 
disP - The Planning Review 49(2), 101-117.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

# Import packages
import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations

# Import sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 


# Configure
pd.set_option('display.max_columns', 150)


## 1. Load the dataset
optima = pd.read_csv("Data/datasets/optima.csv", sep='\t')


## 2. Preprocess the dataset.
# 2.1 Feature selection
features_ext = ['ID', 'TimePT', 'TimeCar', 'MarginalCostPT', 'CostCarCHF', 'distance_km',
                'Gender', 'age', 'NbChild', 'NbCar', 'NbMoto', 'NbBicy', 'OccupStat'] 
choice_var = 'Choice'
total_var_ext = features_ext.copy()
total_var_ext.append(choice_var)
dataset_ext = optima[total_var_ext]

# 2.2. Filter data
# Remove all rows with a nan
dataset_ext = dataset_ext.dropna()
# Remove all duplicate rows
dataset_ext = dataset_ext.drop_duplicates()
# Exclude observations such that the chosen alternative is -1
dataset_ext = dataset_ext[dataset_ext[choice_var] != -1]

# 2.3. Definition of new variables.
dataset_ext['Gender'] = dataset_ext['Gender'].map({1: 'man', 2: 'woman', -1: 'unreported'})
dataset_ext = pd.get_dummies(dataset_ext, columns=['Gender'])

dataset_ext['OccupStat'] = dataset_ext['OccupStat'].map({1: 'fulltime', 2: 'notfulltime', 3: 'notfulltime', 4: 'notfulltime', 5: 'notfulltime', 6: 'notfulltime', 7: 'notfulltime', 8: 'notfulltime', 9: 'notfulltime', -1: 'notfulltime'})
dataset_ext = pd.get_dummies(dataset_ext, columns=['OccupStat'])

# 2.4 Divide between train and test sets
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
split = splitter.split(dataset, groups=dataset['ID'])
train_inds, test_inds = next(split)

train_set_ext, test_set_ext = dataset_ext.iloc[train_inds], dataset_ext.iloc[test_inds]


## 3. Store the train and test datasets
train_set_ext.to_csv(r'Data/datasets/preprocessed/optima_ext_train.csv', sep=',', index=False)
test_set_ext.to_csv(r'Data/datasets/preprocessed/optima_ext_test.csv', sep=',', index=False)
