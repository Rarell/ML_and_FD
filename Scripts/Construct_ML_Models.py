#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:48:10 2022

@author: stuartedris

This script contains all the codework for building the ML models used in the 
FD identification via ML models project.     

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory

Current ML models include:
- 

TODO:
- Ensure there are no function name conflicts with other scripts
- Fill out the build_model() function
- Add RFs
- Add SVMs
- Add NNs:
    ANNs
    CNNs
    RNNs
    Others
- Test reworked code
"""


#%%
##############################################

# Library impots
import os, sys, warnings
import gc
import argparse
import pickle
import numpy as np
import multiprocessing as mp
import pathos.multiprocessing as pmp
from joblib import parallel_backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy.special import gamma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib import patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics


#%%
##############################################

def build_sklean_model(args):
    '''
    Build a ML model from the sklearn package
    
    Inputs:
    :param args: Argparse arguments
    '''
    
    if (args.ml_model.lower() == 'rf') | (args.ml_model.lower() == 'random_forest'):
        model = build_rf_model(args)
    
    return model


def build_keras_model():
    '''
    Build a ML model (a nueral network) from the keras package
    '''
    
    model = 10
    
    return model


#%%
##############################################

# Function to make a RF model
def build_rf_model(args):
    '''
    Construct a random forest model using sklearn
    
    Inputs:
    :param args: Argparse arguments
    '''
    
    model = ensemble.RandomForestClassifier(n_estimators = args.n_trees, 
                                            criterion = args.tree_criterion,
                                            max_depth = args.tree_depth,
                                            max_features = args.tree_max_features,
                                            bootstrap = args.feature_importance,
                                            oob_score = args.feature_importance,
                                            verbose = args.verbose)
                                            #n_jobs = )
    
    return model