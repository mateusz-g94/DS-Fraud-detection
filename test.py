#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:07:03 2019

@author: thatone

Classifier evaluation on train and test. 

"""

import pandas as pd
from DataScience.classification.classifiereval import ClassifierEval
from DataScience.utilities import StoreModels

# Read data 
FILE_PATH_X_TEST = './data/x_test.csv'
FILE_PATH_Y_TEST = './data/y_test.csv'
FILE_PATH_X_TRAIN = './data/x_train.csv'
FILE_PATH_Y_TRAIN = './data/y_train.csv'

X_test = pd.read_csv(FILE_PATH_X_TEST, index_col = 0)
y_test = pd.read_csv(FILE_PATH_Y_TEST, index_col = 0)
X_train = pd.read_csv(FILE_PATH_X_TRAIN, index_col = 0)
y_train = pd.read_csv(FILE_PATH_Y_TRAIN, index_col = 0)

# Open database storing trained models
models_dw = StoreModels() 

# Create params for eval object
params = {}

# For models create params to evaluate
#for model in ['run_for_nm', 'run_for_rus', 'run_for_adasyn', 'run_for_smote']:
for model in models_dw.get_models_list():
    params[model] = (models_dw.load_model(model), {'train' : (X_train, y_train), 'test' : (X_test, y_test)})
    
# Open eval object
eval_obj = ClassifierEval(params = params)

# Eval
eval_obj.lift_chart(orientation = 'set')
eval_obj.lift_chart(orientation = 'model')
eval_obj.lift_chart(orientation = 'set', chart_type = 'captured response')
eval_obj.lift_chart(orientation = 'model', chart_type = 'captured response')
eval_obj.lift_chart(orientation = 'set', chart_type = 'response', scale = 1000)
eval_obj.lift_chart(orientation = 'model', chart_type = 'response', scale = 1000)
eval_obj.roc_chart()
eval_obj.roc_chart(orientation = 'set')
eval_obj.precision_recall_chart(x_thresshold = False)
eval_obj.precision_recall_chart(x_thresshold = True)
eval_obj.precision_recall_chart(orientation = 'set', x_thresshold = False)
eval_obj.score_hist_chart()
eval_obj.conf_matrix_chart(0.70)
