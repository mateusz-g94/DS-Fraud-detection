#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:38:05 2019

@author: thatone

Utilities.

"""

from bayes_opt import BayesianOptimization
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def bayes_opt_random_forest(X, y, init_round = 15, opt_round = 25, cv = 3):
    
    def train_random_forest(n_estimators, max_depth, min_samples_leaf, cv = cv):
        params = {}
        params['n_estimators'] = int(round(n_estimators))
        params['max_depth'] = int(round(max_depth))
        params['min_samples_leaf'] = int(round(min_samples_leaf))
        params['n_jobs'] = 3
        clf = RandomForestClassifier(**params)
        scores = cross_val_score(clf, X, y, cv = cv, scoring = make_scorer(average_precision_score, greater_is_better = True))
        return sum(scores)/len(scores)
    
    optimizer = BayesianOptimization(train_random_forest, {'n_estimators' : (50, 500),
                                                           'max_depth' : (2,10),
                                                           'min_samples_leaf' : (1,30)}, random_state = 77)
    
    optimizer.maximize(init_points = init_round, n_iter = opt_round)
    params_best = {k : int(v) for k, v in optimizer.max['params'].items()}
    clf_best = RandomForestClassifier(**params_best)
    clf_best.fit(X,y)
    return clf_best

def bayes_opt_xgboost(X, y, init_round = 15, opt_round = 25, cv = 3):
    
    def train_xgboost(n_estimators, max_depth, eta, subsample, cv = cv):
        params = {}
        params['n_estimators'] = int(round(n_estimators))
        params['max_depth'] = int(round(max_depth))
        params['eta'] = eta
        params['subsample'] = subsample
        params['n_jobs'] = -1
        clf = XGBClassifier(**params)
        scores = cross_val_score(clf, X, y, cv = cv, scoring = make_scorer(average_precision_score, greater_is_better = True))
        return sum(scores)/len(scores)
    
    optimizer = BayesianOptimization(train_xgboost, {'eta' : (0,1),
                                                     'subsample' : (0.2, 1),
                                                     'n_estimators' : (50, 500),
                                                     'max_depth' : (2,10)}, random_state = 77)
    
    optimizer.maximize(init_points = init_round, n_iter = opt_round)
    params_best = {k : int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in optimizer.max['params'].items()}
    clf_best = XGBClassifier(**params_best)
    clf_best.fit(X,y)
    return clf_best
