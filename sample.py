#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:22:39 2019

@author: thatone

Split data based on imbalanced sampling techniques.

"""
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler      # 1) undersampling
from imblearn.under_sampling import NearMiss                # 2) undersampling
from imblearn.over_sampling import SMOTE, ADASYN            # 3) 4) oversampling

FILE_PATH_X = './data/x_train.csv'
FILE_PATH_Y = './data/y_train.csv'
PATH_EXP = './data'

X = pd.read_csv(FILE_PATH_X, index_col = 0)
y = pd.read_csv(FILE_PATH_Y, index_col = 0)

samplers = [(RandomUnderSampler(ratio = 0.15), 'rus'),\
            (NearMiss(ratio = 0.15), 'nm'),\
            (SMOTE(ratio = 0.15),'smote'),\
            (ADASYN(ratio = 0.15), 'adasyn')]

for sampler in samplers:
    print('Sampler', sampler[1], 'START')
    X_resampled, y_resampled = sampler[0].fit_resample(X,y)
    df_X_resampled = pd.DataFrame(X_resampled, columns = X.columns)
    df_y_resampled = pd.DataFrame(y_resampled, columns = y.columns)
    name_x = 'x_train_' + sampler[1] + '.csv'
    name_y = 'y_train_' + sampler[1] + '.csv'
    df_X_resampled.to_csv(os.path.join(PATH_EXP, name_x))
    df_y_resampled.to_csv(os.path.join(PATH_EXP, name_y))
    print(df_y_resampled['target'].value_counts())
    
