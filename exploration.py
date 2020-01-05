#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:59:25 2019

@author: thatone

Exploration analysis.

"""
from plot_utils import plot_numeric_var,  plot_settings, plot_corr
import pandas as pd
import os

FILE_PATH = './data/creditcard_prep.csv'
GRP_PATH = 'grp'

# Read CSV file
data = pd.read_csv(FILE_PATH, index_col = 0)

# Visualizations
plot_settings()
plot_numeric_var(data = data, var = 'V1', target_name = 'target', trunct_level = 0.2)

# Plot for all variables and save
def plot_histograms(trunct_level = 0.2):
    if not os.path.exists(GRP_PATH):
        os.mkdir(GRP_PATH)
    for col in [col for col in data.columns if col != 'target']:
        path = os.path.join(GRP_PATH, 'hist_' + str(col) + '.jpg')
        plot_numeric_var(data = data, var = col, target_name = 'target', save_path = path, trunct_level = trunct_level)

plot_histograms()
plot_corr(data = data, exclude = ['target', 'Time'], save_path = os.path.join(GRP_PATH, 'corr.jpg'))
# Variables arent correlated, because it's a result of PCA. 
