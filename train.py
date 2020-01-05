#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:29:33 2019

@author: thatone

Train models.

NOTATKI:
    - przetrenowac drugi model XGBoost vs Adabost?
    - napisac drugi opt bayes
    - rozwioj ClasifEval

"""

import pandas as pd
from utils import bayes_opt_random_forest, bayes_opt_xgboost
from plot_utils import plot_numeric_var 
from sklearn.metrics import confusion_matrix
from DataScience.utilities import StoreModels

data_names = ['nm', 'rus','adasyn', 'smote']
models_dw = StoreModels()

for name in data_names:
    FILE_PATH_X = './data/x_train_' + name + '.csv'
    FILE_PATH_Y = './data/y_train_' + name + '.csv'
    X = pd.read_csv(FILE_PATH_X, index_col = 0)
    y = pd.read_csv(FILE_PATH_Y, index_col = 0)
    best_model = bayes_opt_random_forest(X = X, y = y['target'], init_round = 15, opt_round = 25, cv = 3)
    models_dw.save_model(best_model, 'run_for_' + name)
    best_model = bayes_opt_xgboost(X = X, y = y['target'], init_round = 2, opt_round = 2, cv = 3)
    models_dw.save_model(best_model, 'xgb_' + name)
