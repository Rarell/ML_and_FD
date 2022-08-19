#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:52:45 2021

##############################################################
# File: ML_and_FD.py
# Version: 1.0.0
# Author: Stuart Edris (sgedris@ou.edu)
# Description:
#     This is the main script for the employment of machine learning to identify flash drought study.
#     This script takes in indces calculated from the Calculate_Indices script (training data) and the 
#     identified flash drought in the Calculate_FD script (label data) and identifies flash drought
#     using those indices (minus the index used to calculate flash drought). Several models are employed
#     (decision trees to set the process up, boosted trees, random forests, SVMs, and nueral networks).
#     Models are run for each flash drought identification method. Output results are given in the final
#     models, ROC curves, tables of performance statistics, weights (contribution of each index), etc.
#
# 
#
# Version History: 
#   1.0.0 - 12/24/2021 - Initial reformatting of the script to use a 'version' setting (note the version number is not in the script name, so this is not a full version version control)
#   1.1.0 - 12/25/2021 - Implemented code to split the dataset into regions in the DetermineParameters and CreateSLModel functions
#   1.2.0 - 12/30/2021 - Modified the main functions to write outputs to .txt file instead of output to the terminal (easier to save everything)
#   1.3.0 - 1/13/2022 - Modified the ModelPredictions function to encorporate multiple regions.
#
# Inputs:
#   - Data files for FD indices and identified FD (.nc format)
#
# Outputs:
#   - A number of figure showing the results of the SL algorithms
#   - Several outputs (in the terminal) showing performance metrics for the SL algorithms
#
# To Do:
#   - Fix SVMs
#   - Turn all features into functions to be called
#   - Update documentation
#   - Ensure there are no function name conflicts with other scripts
#   - Main function that parses raw data, calculates the index, and calculates FD, if the respective files do not exist
#       - Parse data into 1 growing season per fold
#       - Parse data spatially into 5 degrees by 5 degrees (might change in future)
#       - Add learning curve calculations and figures
#       - Add mean ROC Curve calculations and maps
#       - Add AUC maps
#       - ADD Accuracy and potentially other maps
#       - Add model architecture if appliclable
#       - Add predictive climatology and years and associated errors
#   - Might have a separate script for model building
#   - Add argparse function
#   - Might try a more effective approach to parallel processing for increased computation speed
#   - Test reworked code
#
# Bugs:
#   - SVMs will freeze the code
#
# Notes:
#   - All Python libraries are the lateset versions of the release date.
#   - This script assumes it is being running in the 'ML_and_FD_in_NARR' directory
#   - Several of the comments for this program are based on the Spyder IDL, which can separate the code into different blocks of code, or 'cells'
#
###############################################################

"""

# lat1d = lat.reshape(I*J, order = 'F')
# lon1d = lon.reshape(I*J, order = 'F')

# # data is NFeatures x time x lat x lon 
# lat_labels = np.arange(-90, 90+5, 5)
# lon_labels = np.arange(-180, 180+5, 5)

# I = lat_labels.size
# J = lon_labels.size

# # Split the data into regions.
# data_split = []
# for i in range(I-1):
#     for j in range(J-1):
#         ind = np.where( ((lat_labels[i] >= lat1d) & (lat_labels[i+1] <= lat1d)) & ((lon_labels[j] >= lon1d) & (lon_labels[j+1] <= lon1d)) )[0]
        
#         # Not all datasets are global; remove sets where there is no data
#         if len(ind) < 1: 
#             continue
        
        
#         data_split.append(data_stacked[:,:,ind])



# Load in data
#  Consider normalizing to be between 0 & 1
# Split in training, validation, and test sets
# Split into spatial regions
# Loop over all methods:
    # For each region:
        # Loop over all rotations:
            # Perform experiment
            # Save model
            # Save training, validation, and results to a list (Results include metrics & predictions)
            # Save learning curves to a list
        # Average (and get variation) the list into a part of a results map
        # Save part of results map to into a piece of a whole results map (whole_mape[ind] = results)
        # Average learning curves together into a spatial list
    # Save whole results map(s) to a pickle file for figures
    # Average (and get variation) learning curves together and save for figures
        
            
    
#%%
##############################################

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

# Function to load in pickle files
def load_ml_data(fname, path = './Data'):
    '''
    Load in a pickle dataset that has been split into different folds
    
    Inputs:
    :param fname: Filename of the dataset being loaded
    :param path: Path leading to the file being loaded
    
    Outputs:
    :param data: Loaded dataset
    '''
    
    # Load the data
    with open("%s\%s"%(path, fname), "rb") as fp:
        data = pickle.load(fp)
        
    return data

#%%
##############################################

# Function to separate 4D data into training, validation, and testing datasets
def split_data(data, ntrain_folds, rotation, normalize = False):
    '''
    Split a data set into a training, validation, and test datasets based on folds
    
    Inputs:
    :param data: Input data to be split. Must be in a Nfeate/Nmethod x time x space x fold format
    :param ntrain_folds: Number of folds to include in the training set
    :param rotation: The rotation of k-fold to use
    :param normalize: Boolean indicating whether to normalize the data to range from 0 to 1
    
    Outputs:
    :param train: Training dataset
    :param validation: Validation dataset
    :param test: Testing dataset
    '''
    
    # Initialize some values
    N, T, IJ, Nfolds = data.shape
    
    data_norm = data
    
    # Normalize the data?
    if normalize:
        for n in range(Nfolds):
            # Note the training set is T (K), ET, PET, P, and soil moisture, which are all, theoretically > 0
            max_value = np.nanmax(data[n,:,:,:])
            #min_value = np.nanmin(data)
            #mean_value = np.nanmean(data)
            
            data_norm[n,:,:,:] = data_norm[n,:,:,:]/max_value
            
    # Determine the training, validation, and test folds
    train_folds = int((np.arange(ntrain_folds) + rotation) % Nfolds)
    validation_folds = int((np.array[ntrain_folds] + rotation) % Nfolds)
    test_folds = int((np.array([ntrain_folds]) + 1 + rotation) % Nfolds)
    
    # Collect the training, validation, and test data
    # train = np.ones((N, T*ntrain_folds, IJ))
    # for n, fold in enumerate(train_folds):
    #     ind_start = n*T
    #     ind_end = (n+1)*T
    #     train[:,ind_start:ind_end,:] = data_norm[:,:,:,fold]
    
    train = np.concatenate([data_norm[:,:,:,fold] for fold in train_folds], axis = 1)
        
    validation = data_norm[:,:,:,validation_folds]
    test = data_norm[:,:,:,test_folds]
    
    #### This is a test to ensure the split is correct.
    print(train.shape, validation.shape, test.shape)
    
    return train, validation, test


#%%
##############################################

# Function to conduct a single experiment
def execute_single_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname):
    '''
    Run a single ML experiment and save the model
    
    Inputs:
    :param args: Argparse arguments
    :param train_in: Input training dataset used to train the model
    :param valid_in: Input validation dataset to help validate the model
    :param test_in: Input testing dataset to test the model
    :param train_out: Output training dataset used to train the model
    :param valid_out: Output validation dataset to help validate the model
    :param test_out: Output testing dataset to test the model
    :param model_fname: The filename to save the model to
    
    Outputs:
    :param results: Dictionary results from the ML model, including predictions, performance metrics, and learning curves
    '''
    
    # Build the model
    model = build_model(train_in, valid_in, test_in, train_out, valid_out, test_out, args)
    
    # Train the model
    model.train()
    
    # Collect model results
    results = {}
    
    # Model predictions
    results['train_predict'] = model.predict(train_in)
    results['valid_predict'] = model.predict(valid_in)
    results['test_predict'] = model.predict(test_in)
    
    # Model performance
    results['train_eval'] = model.evaluate(train_in, train_out)
    results['valid_eval'] = model.evaluate(valid_in, valid_out)
    results['test_eval'] = model.evaluate(test_in, test_out)
    
    # Learning curves/model history
    results['history'] = model.history
    
    # Save the model
    model.save(model_fname)
        
    return results


#%%
##############################################

# Function to conduct all experiments for 1 ML model (for all rotations, methods, and regions)
def execute_all_exp(args):
    '''
    Run multiple ML experiments for all regions and all rotations
    
    Inputs:
    :param args: Argparse arguments
    '''
    
    ############ 6 args inputs used that are not defined
    
    # Determine the directory of the data
    dataset_dir = '%s/%s'%(args.dataset, args.model)
    
    # Load the data
    # Data is Nfeatures/Nmethods x time x space x fold
    data_in = load_ml_data(args.input_data_fname, path = dataset_dir)
    data_out = load_ml_data(args.output_data_fname, path = dataset_dir)
    
    # Make the rotations
    Nmethods, T, IJ, Nfold  = data_in.shape
    rotation = np.arange(Nfold)
    
    # Make the latitude/longitude labels for individual regions
    lat_labels = np.arange(-90, 90+5, 5)
    lon_labels = np.arange(-180, 180+5, 5)
    
    # Load and reshape lat/lon data
    # Load lat and lon data
    lat = load2D_nc('lat_%s.nc'%args.model, sname = 'lat', path = dataset_dir)
    lon = load2D_nc('lat_%s.nc'%args.model, sname = 'lon', path = dataset_dir)
    
    # Correct the longitude?
    if args.correct_lon:
        for n in range(len(lon[:,0])):
            ind = np.where(lon[n,:] > 0)[0]
            lon[n,ind] = -1*lon[n,ind]
            
    # Reshape the latitude/longitude data
    I, J = lat.shape
    lat1d = lat.reshape(I*J, order = 'F')
    lon1d = lon.reshape(I*J, order = 'F')
    
    I = lat_labels.size
    J = lon_labels.size
    
    
    # Split the data into regions.
    data_in_split = []
    data_out_split = []
    lat_lab = []
    lon_lab = []
    for i in range(I-1):
        for j in range(J-1):
            ind = np.where( ((lat_labels[i] >= lat1d) & (lat_labels[i+1] <= lat1d)) & ((lon_labels[j] >= lon1d) & (lon_labels[j+1] <= lon1d)) )[0]
            
            # Not all datasets are global; remove sets where there is no data
            if len(ind) < 1: 
                continue
            
            lat_lab.append(lat_labels[i])
            lon_lab.append(lon_labels[j])
            
            data_in_split.append(data_in[:,:,ind,:])
            data_out_split.append(data_out[:,:,ind,:])
            
    
    # Loop over all FD methods to conduct the set of experiments for all of them
    for method in range(Nmethods):
        # Initialize results
        pred_train = np.ones((T, IJ))
        pred_valid = np.ones((T, IJ))
        pred_test = np.ones((T, IJ))
        
        eval_train = np.ones((IJ))
        eval_valid = np.ones((IJ))
        eval_test = np.ones((IJ))
        
        learn_curves = np.ones((T, IJ))
        
        # Begin looping and performing an experiment over all regions
        for n, (region_in, region_out) in enumerate(zip(data_in_split, data_out_split)):
            # Find where the current latitude and longitude values are
            ind = np.where( ((lat_lab[n] >= lat1d) & (lat_lab[n+1] <= lat1d)) & ((lon_lab[n] >= lon1d) & (lon_lab[n+1] <= lon1d)) )[0]
            
            # Initialize some lists
            ptrain = []
            pvalid = []
            ptest = []
            
            etrain = []
            evalid = []
            etest = []
            
            lc = []
            # For each region, perform an experiment for each rotation; obtain a statistical sample
            for rot in rotation:
                
                # Split the data into training, validation, and test sets
                train_in, valid_in, test_in = split_data(region_in, args.ntrain_fold, rot, normalize = args.normalize)
                train_out, valid_out, test_out = split_data(region_out, args.train_fold, rot, normalize = False) # Note the label data is already binary
                
                # Generate the model filename
                model_fname = generate_model_fname(args, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5], rot)
                
                # Perform the experiment
                # The ML model is saved in this step
                results = execute_single_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname)
                
                # Collect the results for the rotation
                ptrain.append(results['train_predict'])
                pvalid.append(results['valid_predict'])
                ptest.append(results['test_predict'])
                
                etrain.append(results['train_eval'])
                evalid.append(results['valid_eval'])
                etest.append(results['test_eval'])
                
                lc.append(results['history'])
                
            # At the end of the experiments for each rotation, stack the reults into a single array (per variable) and average along the rotation axis
            pred_train[:,ind] = np.nanmean(np.stack(ptrain, axis = -1), axis = -1)
            pred_valid[:,ind] = np.nanmean(np.stack(pvalid, axis = -1), axis = -1)
            pred_test[:,ind] = np.nanmean(np.stack(ptest, axis = -1), axis = -1)
            
            eval_train[ind] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
            eval_valid[ind] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
            eval_test[ind] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
            
            learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
            
        # At the end of the experiments for each region, collect the results and save the results
        results_fname = generate_results_fname()
        
        results = {}
        
        # Model predictions
        results['train_predict'] = pred_train
        results['valid_predict'] = pred_valid
        results['test_predict'] = pred_test
        
        # Model performance
        results['train_eval'] = eval_train
        results['valid_eval'] = eval_valid
        results['test_eval'] = eval_test
        
        # Learning curves/model history
        results['history'] = learn_curves
        
        # Save the results
        with open("%s/%s"%(dataset_dir,results_fname), "wb") as fp:
            pickle.dump(results, fp)
    






