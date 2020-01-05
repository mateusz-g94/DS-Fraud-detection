#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:30:12 2019

@author: thatone

Vizualization functions.

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

def plot_settings(size = [15,11]):
    plt.style.use('seaborn-darkgrid')
    sns.set(style = 'darkgrid')
    plt.rcParams["figure.figsize"] = size
    sns.set(rc={'figure.figsize' : size})

def plot_numeric_var(data, var, target_name, trunct_level = 0, save_path = None):
    
    def truncate_histogram(data, var, trunct_level, target_name):
        data_temp = data[[var, target_name]].loc[:].dropna().sort_values(var).reset_index(drop=True)
        temp = data_temp.shape[0] * trunct_level / 100
        temp_min = temp / 2
        temp_max = data_temp.shape[0] - temp / 2
        return data_temp.loc[temp_min : temp_max]
    
    if trunct_level != 0:
        data_hist = truncate_histogram(data = data, var = var, trunct_level = trunct_level, target_name = target_name)
    else:
        data_hist = data
    
    fig = plt.figure(tight_layout = False)
    gs = gridspec.GridSpec(2,2)
    
    ax = fig.add_subplot(gs[0, :])
    bins = np.linspace(int(data_hist[var].min()), int(np.ceil(data_hist[var].max())), 175)
    ax.hist(data_hist[var].where(data_hist[target_name] == 0).dropna(), bins = bins, alpha = 0.5, label = str(target_name) + ' = 0', density = True)    
    ax.hist(data_hist[var].where(data_hist[target_name] == 1).dropna(), bins = bins, alpha = 0.5, label = str(target_name) + ' = 1', density = True)  
    ax.set_ylabel('%')
    ax.set_xlabel(var)
    ax.set_xlim(data_hist[var].min(), data_hist[var].max())
    ax.legend(loc = 'upper right')
    
    ax = fig.add_subplot(gs[1,0])
    sns.boxplot(y = target_name, x = var, data = data, orient = 'h', ax = ax, boxprops = dict(alpha = .5))
    plt.axvline(data_hist[var].min(), color = 'gray', linestyle = '--')
    plt.axvline(data_hist[var].max(), color = 'gray', linestyle = '--')
    
    ax = fig.add_subplot(gs[1,1])
    count_nan = 100 * (len(data[var]) - data[var].count())/len(data[var])
    vals = [count_nan, 100 - count_nan]
    ax.pie(vals, radius = 1, labels = ['Missing values', 'Data'], autopct = '%1.1f%%', wedgeprops = dict(alpha = .5, width = 0.3, edgecolor = 'w'))
    ax.axis('equal')
    
    fig.align_labels()
    fig.tight_layout()
    
    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
    
def plot_corr(data, exclude = None, save_path = None):
    if exclude != None:
        cols = [col for col in data.columns if col not in exclude]
    else:
        cols = data.columns
    corr = data[cols].corr()
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0, square = True, linewidths = .5, cbar_kws = {"shrink" : .5})
    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    