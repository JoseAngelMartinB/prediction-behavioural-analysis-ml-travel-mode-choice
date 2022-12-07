#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess the LPMC dataset (London Passenger Mode Choice).

Reference: 
Hillel, T., Elshafie, M. Z. E. B. and Jin, Y. (2018), 
'Recreating passenger mode choice-sets for transport simulation: 
A case study of London, UK', 171(1), 29-42.

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


# Configure
pd.set_option('display.max_columns', 150)


## 1. Load the dataset
LPMC_data = pd.read_csv("Data/Datasets/LPMC.csv", sep=',')


## 2. Preprocess the dataset.
# 2.1. Transform the categorical variables
# One-hot encoding for purpose, faretype and fueltype
purpose_df = pd.get_dummies(LPMC_data['purpose'], prefix='purpose')
LPMC_data = LPMC_data.join(purpose_df)
LPMC_data.drop(columns=['purpose'], inplace=True)

#faretype_df = pd.get_dummies(LPMC_data['faretype'], prefix='faretype')
#faretype_df = faretype_df.rename(columns={"faretype_16+": "faretype_16over"}) # Remove + symbol from variable
#LPMC_data = LPMC_data.join(faretype_df)
LPMC_data.drop(columns=['faretype'], inplace=True)

LPMC_data['fueltype'] = LPMC_data['fueltype'].map({'Petrol_Car':'Petrol', 'Petrol_LGV':'Petrol', 'Diesel_Car':'Diesel', 'Diesel_LGV':'Diesel', 'Hybrid_Car':'Hybrid', 'Average_Car':'Average'})
fueltype_df = pd.get_dummies(LPMC_data['fueltype'], prefix='fueltype')
LPMC_data = LPMC_data.join(fueltype_df)
LPMC_data.drop(columns=['fueltype'], inplace=True)

# Convert choice from categorical variable to numeric one
LPMC_data['travel_mode'] = LPMC_data['travel_mode'].map({'walk':0, 'cycle':1, 'pt':2, 'drive':3})

# Remove unused columns
LPMC_data.drop(columns=['trip_id', 'person_n', 'trip_n', 'travel_year', 'travel_month', 'travel_date', 'bus_scale', 'dur_pt_total', 'dur_pt_int_total', 'cost_driving_fuel', 'cost_driving_con_charge', 'driving_traffic_percent'], inplace=True)

# 2.2 Divide between train and test sets
# Take years 2012-2014 for training (aprox. 70% of data) and 2014-2015 for testing (aprox. 30% of data)
# Survey year: 1 (April 2012-March 2013), 2 (13/14) or 3 (14/15)
train_set = LPMC_data[LPMC_data['survey_year'].isin([1, 2])]
test_set = LPMC_data[LPMC_data['survey_year'].values == 3]

train_set.drop(columns=['survey_year'], inplace=True)
test_set.drop(columns=['survey_year'], inplace=True)

print("Length of train: %d\nLength of test: %d" % (train_set.shape[0], test_set.shape[0]))


## 3. Store the train and test datasets
train_set.to_csv(r'Data/Datasets/preprocessed/LPMC_train.csv', sep=',', index=False)
test_set.to_csv(r'Data/Datasets/preprocessed/LPMC_test.csv', sep=',', index=False)