#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess the NTS dataset.

Reference: 
Onderzoek Verplaatsingen in Nederland (2014). Onderzoeksbeschrijving ovin 2010-2014.
Data archiving and networked services (dans). 
https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:54132/tab/1

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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 


# Configure
pd.set_option('display.max_columns', 150)


## 1. Load the dataset
nts_data = pd.read_csv("Data/Datasets/nts_data.csv", sep=',')


## 2. Preprocess the dataset.
# 2.1. Random sample
nts_data = nts_data.sample(n=100000, replace=False, random_state=1996)
nts_data = nts_data.reset_index(drop=True)

# 2.2. Transform the categorical variables
# Label Encoder
license_encoder = LabelEncoder()
male_encoder = LabelEncoder()
weekend_encoder = LabelEncoder()

license_cat = nts_data['license']
male_cat = nts_data['male']
weekend_cat = nts_data['weekend']

license_cat = license_encoder.fit_transform(license_cat)
male_cat = license_encoder.fit_transform(male_cat)
weekend_cat = weekend_encoder.fit_transform(weekend_cat)

# Label Binarizer
ethnicity_encoder = LabelBinarizer()
ethnicity_cat = nts_data['ethnicity']
ethnicity_cat_1hot = ethnicity_encoder.fit_transform(ethnicity_cat)

y_cat = nts_data['mode_main']
y_cat = y_cat.map({'walk' : 0, 'bike' : 1, 'pt' : 2, 'car' : 3})

income_cat = nts_data['income']
income_cat = income_cat.map({'less20' : 2, '20to40' : 3, 'more40' : 4})

education_cat = nts_data['education']
education_cat = education_cat.map({'lower' : 1, 'middle' : 2 , 'higher' : 3})

# Infer the individualID from the data
# individual IDs can be imputed by enumerating the unique combinations of postcode area (using the diversity and green
# features in the data) and socio-economic data (combination of age, gender, ethnicity, education, income, cars,
# bikes, and driving license ownership). If two people in the survey data from the same postcode area have the same
# socio-economic details they will be grouped as the same person.
individual_id = nts_data.groupby(["diversity","green", "age", "male", "ethnicity", "education", "income", "cars",
                                  "bicycles", "license"]).ngroup()

# Create a new dataframe
data_trans = nts_data.drop(columns=['mode_main', 'license', 'male', 'weekend', 'education', 'ethnicity', 'income'], axis='columns')

data_trans = pd.concat([data_trans,
                           pd.DataFrame(y_cat, dtype=int, columns=['mode_main']),
                           pd.DataFrame(individual_id, dtype=int, columns=['individual_id']),
                           pd.DataFrame(license_cat, dtype=int, columns=['license_cat']),
                           pd.DataFrame(male_cat, dtype=int, columns=['male_cat']),
                           pd.DataFrame(weekend_cat, dtype=int, columns=['weekend_cat']),
                           pd.DataFrame(ethnicity_cat_1hot, dtype=int, columns=ethnicity_encoder.classes_),
                           pd.DataFrame(np.asarray(income_cat), dtype=int, columns=['income_cat']),
                           pd.DataFrame(np.asarray(education_cat), dtype=int, columns=['education_cat'])
                          ], axis=1)

# 2.3 Divide between train and test sets
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
split = splitter.split(data_trans, groups=data_trans['individual_id'])
train_inds, test_inds = next(split)
train_set, test_set = data_trans.iloc[train_inds], data_trans.iloc[test_inds]


## 3. Store the train and test datasets
train_set.to_csv(r'Data/datasets/preprocessed/nts_data_train.csv', sep=',', index=False)
test_set.to_csv(r'Data/datasets/preprocessed/nts_data_test.csv', sep=',', index=False)