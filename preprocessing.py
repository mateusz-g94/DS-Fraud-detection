#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:38:28 2019

@author: thatone

Data preprocessing. 

"""
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os

FILE_PATH = './data/creditcard.csv'
PATH_EXP = './data'

# Read CSV file
raw_data = pd.read_csv(FILE_PATH, index_col = 0)

# Check missing values 
print(raw_data.isnull().sum())
print('Any missing values: ', raw_data.isnull().values.any())

# Check column types
print(raw_data.dtypes)

# Rename target column
raw_data.rename(columns={"Class" : "target"}, inplace = True)

# Save file
raw_data.to_csv(os.path.join(PATH_EXP, 'creditcard_prep.csv'))

# Split into X and y 
x_cols = [col for col in raw_data.columns if col != 'target']
raw_data_x = raw_data[x_cols]
raw_data_y = raw_data[['target']]

# Split data into test and train
raw_data_y['target'].value_counts()

# Ther are only 492 ones. I want +-100 ones in test.
spl = StratifiedShuffleSplit(n_splits = 2, test_size = 0.2, random_state = 7)
for train_index, test_index in spl.split(raw_data_x, raw_data_y):
    x_train, x_test = raw_data_x.iloc[train_index], raw_data_x.iloc[test_index]
    y_train, y_test = raw_data_y.iloc[train_index], raw_data_y.iloc[test_index]
    
# Save files
x_train.to_csv(os.path.join(PATH_EXP, 'x_train.csv'))    
y_train.to_csv(os.path.join(PATH_EXP, 'y_train.csv')) 
x_test.to_csv(os.path.join(PATH_EXP, 'x_test.csv')) 
y_test.to_csv(os.path.join(PATH_EXP, 'y_test.csv')) 
    