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
#   - Add feature importance to execute_all_exp and execute_exp
#   - Main function
#       - Add learning curve calculations and figures
#       - Add mean ROC Curve calculations and maps
#       - Add AUC maps
#       - ADD Accuracy and potentially other maps
#       - Add model architecture if appliclable
#       - Add predictive climatology and years and associated errors
#       - Add prediction time series
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

##### Build_model and analyze_results scripts
            
    
#%%
##############################################

# Library imports
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

# Import a custom script
from Raw_Data_Processing import *
from Calculate_Indices import *
from Calculate_FD import *



#%%
##############################################

# argument parser
def create_ml_parser():
    '''
    Create argument parser
    '''
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ML Calculations', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    
    parser.add_argument('--dataset', type=str, default='/Users/stuartedris/desktop/PhD_Research_ML_and_FD/ML_and_FD_in_NARR/Data', help='Data set directory')
    
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--model', type=str, default='narr', help='Reanalysis model the dataset(s) came from')
    parser.add_argument('--ml_model', type=str, default='rf', help='Type of ML model used to conduct experiment(s)')
    parser.add_argument('--normalize', action='store_true', help='Normalize feature data to range from 0 to 1 before training')
    
    
    # Specific experiment configuration
    parser.add_argument('--ntrain_folds', type=int, default=3, help='Number of training folds')
    parser.add_argument('--rotation', type=int, default=0, help='Rotation in the k-fold validation. Only used for conducting a single experiment.')
    
    parser.add_argument('--correct_lon', action='store_true', help='Correct longitude values of the raw dataset')
    parser.add_argument('--keras', action='store_true', help='Use the keras package to make the ML model. Must be usd for NNs, and must be not used otherwise.')
    parser.add_argument('--feature_importance', action='store_true', help='Collect the importance of each feature from the experiment. Is not available for all ML methods.')
    
    
    # Random forest parameters


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
    
    train_folds = np.sort(train_folds)
    train = np.concatenate([data_norm[:,:,:,fold] for fold in train_folds], axis = 1)
        
    validation = data_norm[:,:,:,validation_folds]
    test = data_norm[:,:,:,test_folds]
    
    #### This is a test to ensure the split is correct.
    print(train.shape, validation.shape, test.shape)
    
    return train, validation, test


#%%
##############################################
# Functions to generate file names

def generate_model_fname(model, ml_model, method, rotation, lat_labels, lon_labels):
    '''
    Generate a filename for a ML model to be saved to. Models are differentiated by lat/lon region, rotation, ML model, reanalysis model trained on, FD method
    
    Inputs:
    :param model: Reanalyses model the ML model is trained on
    :param ml_model: The ML model being saved
    :param method: The FD identification method used for labels
    :param rotation: Current rotation in the k-fold validation
    :param lat_labels: List of latitudes that encompass the region
    :param lon_labels: List of longitudes that encompass the region
    
    Outputs:
    :param fname: The filename the ML model will be saved to
    '''
    
    # Create the filename
    fname = '%s_%s_%s_%s_%s-%slat_%s_%slon'%(model, 
                                             ml_model, 
                                             method, 
                                             rotation, 
                                             lat_labels[0], lat_labels[1], 
                                             lon_labels[0], lon_labels[1])
    
    return fname
    
def generate_results_fname(model, ml_model, method):
    '''
    Generate a filaname the results of a ML model will be saved to. Results are differentiated by reanalysis trained on, ML model, FD method
    
    Inputs:
    :param model: Reanalyses model the ML model is trained on
    :param ml_model: The ML model being saved
    :param method: The FD identification method used for labels
    
    Outputs:
    :param fname: The filename the ML results will be saved to
    '''
    
    # Create the filename
    fname = '%s_%s_%s_results.pkl'%(model, ml_model, method)
    
    return fname
    
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
    if args.keras:
        model = build_keras_model(train_in, valid_in, test_in, train_out, valid_out, test_out, args)
        
        # Train the model
        model.train()
        
    else:
        model = build_sklearn_model(train_in, valid_in, test_in, train_out, valid_out, test_out, args)
        
        # Train the model
        model.fit(train_in, train_out)
        
    
    # Collect model results
    results = {}
    
    # Model predictions
    if args.keras:
        results['train_predict'] = model.predict(train_in)
        results['valid_predict'] = model.predict(valid_in)
        results['test_predict'] = model.predict(test_in)
    else:
        results['train_predict'] = model.predict_proba(train_in)[:,1]
        results['valid_predict'] = model.predict_proba(valid_in)[:,1]
        results['test_predict'] = model.predict_proba(test_in)[:,1]
    
    # Model performance
    if args.keras:
        results['train_eval'] = model.evaluate(train_in, train_out)
        results['valid_eval'] = model.evaluate(valid_in, valid_out)
        results['test_eval'] = model.evaluate(test_in, test_out)
    else:
        pass # FILL with sklearn method of evaluation
    
    # Learning curves/model history
    if args.keras:
        results['history'] = model.history
     
    # Collect the feature importance
    if args.feature_importance:
        if args.keras:
            pass
        else:
            results['feature_importance'] = model.feature_importances_
    
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
    
    # List of FD identification methods
    methods = ['christian', 'nogeura', 'pendergrass', 'liu', 'otkin']
    
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
        
        pred_train_var = np.ones((T, IJ))
        pred_valid_var = np.ones((T, IJ))
        pred_test_var = np.ones((T, IJ))
        
        eval_train_var = np.ones((IJ))
        eval_valid_var = np.ones((IJ))
        eval_test_var = np.ones((IJ))
        
        learn_curves_var = np.ones((T, IJ))
        
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
                train_in, valid_in, test_in = split_data(region_in, args.ntrain_folds, rot, normalize = args.normalize)
                train_out, valid_out, test_out = split_data(region_out, args.ntrain_folds, rot, normalize = False) # Note the label data is already binary
                
                # Generate the model filename
                model_fbase = generate_model_fname(args.model, args.ml_model, methods[method], rot, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
                model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, method, model_fbase)
                
                # Perform the experiment
                # The ML model is saved in this step
                results = execute_single_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname)
                
                # Collect the results for the rotation
                ptrain.append(results['train_predict'])
                pvalid.append(results['valid_predict'])
                ptest.append(results['test_predict'])
                
                if args.keras:
                    etrain.append(results['train_eval'])
                    evalid.append(results['valid_eval'])
                    etest.append(results['test_eval'])
                
                lc.append(results['history'])
                
            ### Add variation as well
            # At the end of the experiments for each rotation, stack the reults into a single array (per variable) and average along the rotation axis
            pred_train[:,ind] = np.nanmean(np.stack(ptrain, axis = -1), axis = -1)
            pred_valid[:,ind] = np.nanmean(np.stack(pvalid, axis = -1), axis = -1)
            pred_test[:,ind] = np.nanmean(np.stack(ptest, axis = -1), axis = -1)
            
            if args.keras:
                eval_train[ind] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
                eval_valid[ind] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
                eval_test[ind] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
            
            learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
            
            
            pred_train_var[:,ind] = np.nanstd(np.stack(ptrain, axis = -1), axis = -1)
            pred_valid_var[:,ind] = np.nanstd(np.stack(pvalid, axis = -1), axis = -1)
            pred_test_var[:,ind] = np.nanstd(np.stack(ptest, axis = -1), axis = -1)
            
            if args.keras:
                eval_train_var[ind] = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
                eval_valid_var[ind] = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
                eval_test_var[ind] = np.nanstd(np.stack(etest, axis = -1), axis = -1)
            
            learn_curves_var[:,ind] = np.nanstd(np.stack(lc, axis = -1), axis = -1)
            
        # At the end of the experiments for each region, collect the results and save the results
        results_fbase = generate_results_fname(args.model, args.ml_model, methods[method])
        results_fname = '%s/%s'%(dataset_dir, results_fbase)
        
        results = {}
        
        # Model predictions
        results['train_predict'] = pred_train
        results['valid_predict'] = pred_valid
        results['test_predict'] = pred_test
        
        results['train_predict_var'] = pred_train_var
        results['valid_predict_var'] = pred_valid_var
        results['test_predict_var'] = pred_test_var
        
        # Model performance
        results['train_eval'] = eval_train
        results['valid_eval'] = eval_valid
        results['test_eval'] = eval_test
        
        results['train_eval_var'] = eval_train_var
        results['valid_eval_var'] = eval_valid_var
        results['test_eval_var'] = eval_test_var
        
        # Learning curves/model history
        results['history'] = learn_curves
        results['history_var'] = learn_curves_var
        
        # Save the results
        with open("%s/%s"%(dataset_dir,results_fname), "wb") as fp:
            pickle.dump(results, fp)
    


#%%
##############################################

# Function to conduct an experiment for 1 rotation and 1 ML model and 1 FD method (for all regions)
def execute_exp(args):
    '''
    Run multiple ML experiments for all regions and all rotations
    
    Inputs:
    :param args: Argparse arguments
    
    Outputs:
    :param results: The results of the experiment
    '''
    
    # List of FD identification methods
    methods = ['christian']
    
    # Determine the directory of the data
    dataset_dir = '%s/%s'%(args.dataset, args.model)
    
    # Load the data
    # Data is Nfeatures/Nmethods x time x space x fold
    data_in = load_ml_data(args.input_data_fname, path = dataset_dir)
    data_out = load_ml_data(args.output_data_fname, path = dataset_dir)
    
    # Make the rotations
    Nmethods, T, IJ, Nfold  = data_in.shape
    
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
            


    # Initialize results
    pred_train = np.ones((T, IJ))
    pred_valid = np.ones((T, IJ))
    pred_test = np.ones((T, IJ))
    
    eval_train = np.ones((IJ))
    eval_valid = np.ones((IJ))
    eval_test = np.ones((IJ))
    
    learn_curves = np.ones((T, IJ))
    
    pred_train_var = np.ones((T, IJ))
    pred_valid_var = np.ones((T, IJ))
    pred_test_var = np.ones((T, IJ))
    
    eval_train_var = np.ones((IJ))
    eval_valid_var = np.ones((IJ))
    eval_test_var = np.ones((IJ))
    
    learn_curves_var = np.ones((T, IJ))
    
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

            
        # Split the data into training, validation, and test sets
        train_in, valid_in, test_in = split_data(region_in, args.ntrain_folds, args.rotation, normalize = args.normalize)
        train_out, valid_out, test_out = split_data(region_out, args.train_folds, args.rotation, normalize = False) # Note the label data is already binary
        
        # Generate the model filename
        model_fbase = generate_model_fname(args.model, args.ml_model, methods[0], args.rotation, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
        model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, method, model_fbase)
        print(model_fname)
        
        # Perform the experiment
        # The ML model is saved in this step
        results = execute_single_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname)
        
        # Collect the results for the rotation
        ptrain.append(results['train_predict'])
        pvalid.append(results['valid_predict'])
        ptest.append(results['test_predict'])
        
        if args.keras:
            etrain.append(results['train_eval'])
            evalid.append(results['valid_eval'])
            etest.append(results['test_eval'])
        
        lc.append(results['history'])
            
    # At the end of the experiments for each rotation, stack the reults into a single array (per variable) and average along the rotation axis
    pred_train[:,ind] = np.nanmean(np.stack(ptrain, axis = -1), axis = -1)
    pred_valid[:,ind] = np.nanmean(np.stack(pvalid, axis = -1), axis = -1)
    pred_test[:,ind] = np.nanmean(np.stack(ptest, axis = -1), axis = -1)
    
    if args.keras:
        eval_train[ind] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
        eval_valid[ind] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
        eval_test[ind] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
    
    learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
    
    
    pred_train_var[:,ind] = np.nanstd(np.stack(ptrain, axis = -1), axis = -1)
    pred_valid_var[:,ind] = np.nanstd(np.stack(pvalid, axis = -1), axis = -1)
    pred_test_var[:,ind] = np.nanstd(np.stack(ptest, axis = -1), axis = -1)
    
    if args.keras:
        eval_train_var[ind] = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
        eval_valid_var[ind] = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
        eval_test_var[ind] = np.nanstd(np.stack(etest, axis = -1), axis = -1)
    
    learn_curves_var[:,ind] = np.nanstd(np.stack(lc, axis = -1), axis = -1)
        
    # At the end of the experiments for each region, collect the results and save the results
    results_fbase = generate_results_fname(args.model, args.ml_model, methods[0])
    results_fname = '%s/%s'%(dataset_dir, results_fbase)
    print(results_fname)
    
    results = {}
    
    # Model predictions
    results['train_predict'] = pred_train
    results['valid_predict'] = pred_valid
    results['test_predict'] = pred_test
    
    results['train_predict_var'] = pred_train_var
    results['valid_predict_var'] = pred_valid_var
    results['test_predict_var'] = pred_test_var
    
    # Model performance
    results['train_eval'] = eval_train
    results['valid_eval'] = eval_valid
    results['test_eval'] = eval_test
    
    results['train_eval_var'] = eval_train_var
    results['valid_eval_var'] = eval_valid_var
    results['test_eval_var'] = eval_test_var
    
    # Learning curves/model history
    results['history'] = learn_curves
    results['history_var'] = learn_curves_var
    
    # Save the results
    with open("%s/%s"%(dataset_dir,results_fname), "wb") as fp:
        pickle.dump(results, fp)
        
    return results



