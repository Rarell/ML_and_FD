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
#   2.0.1 - 9/05/2022 - Major modifications to the code structure and experiment design. Each growing season is now a fold, regions are based on 5 degree x 5 
#                       degree sections. sklearn section has been tested and is working.
#
# Inputs:
#   - Data files for FD indices and identified FD (.nc format)
#
# Outputs:
#   - A number of figure showing the results of the SL algorithms
#   - Several outputs (in the terminal) showing performance metrics for the SL algorithms
#
# To Do:
#   - Main function
#       - Add learning curve calculations and figures
#       - Add mean ROC Curve calculations and maps
#       - Add AUC maps
#       - ADD Accuracy and potentially other maps
#       - Add model architecture if appliclable
#       - Add predictive climatology and years and associated errors
#       - Add prediction time series
#   - Add argparse arguments for other ML models
#   - Might try a more effective approach to parallel processing for increased computation speed
#   - Add an if __name__ == __main__ section
#   - Keras models have not been tested
#
# Bugs:
#   - 
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
# import pathos.multiprocessing as pmp
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

# Import custom scripts
from Raw_Data_Processing import *
from Calculate_Indices import *
from Calculate_FD import *
from Construct_ML_Models import *
from Create_Figures import *



#%%
##############################################

# argument parser
def create_ml_parser():
    '''
    Create argument parser
    '''
    
    # To add: args.time_series, args.climatology_plot, args.case_studies, args.case_study_years
              
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ML Calculations', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--label', type=str, default='rf', help='Experiment label')
    
    parser.add_argument('--dataset', type=str, default='/Users/stuartedris/desktop/PhD_Research_ML_and_FD/ML_and_FD_in_NARR/Data', help='Data set directory')
    
    parser.add_argument('--input_data_fname', type=str, default='fd_input_features.pkl', help='Filename of the input data')
    parser.add_argument('--output_data_fname', type=str, default='fd_output_labels.pkl', help='Filename of the target output data')
    
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--ra_model', type=str, default='narr', help='Reanalysis model the dataset(s) came from')
    parser.add_argument('--ml_model', type=str, default='rf', help='Type of ML model used to conduct experiment(s)')
    parser.add_argument('--normalize', action='store_true', help='Normalize feature data to range from 0 to 1 before training')
    parser.add_argument('--remove_nans', action='store_true', help='Replace NaNs with 0s. Must be done for sklearn models')
    
    parser.add_argument('--learning_curve', action='store_true', help="Plot the models' average learning curve as part of the results")
    parser.add_argument('--confusion_matrix_plots', action='store_true', help='Make several skill score plots based on a confusion matrix as part of results')
    parser.add_argument('--time_series', action='store_true', help='Plot the ML predicted time series as part of the results')
    parser.add_argument('--climatology_plot', action='store_true', help='Plot the ML predicted climatology as part of the results')
    parser.add_argument('--case_studies', action='store_true', help='Plot a set of case study maps (predicted FD for certain years) as part of the results')
    parser.add_argument('--case_study_years', type=int, nargs='+', default=[1988, 2000, 2011, 2012, 2017, 2019], help='List of years to make case studies for')
    parser.add_argument('--globe', action='store_true', help='Plot global dataset (otherwise plot CONUS)')
    
    
    # Specific experiment configuration
    parser.add_argument('--ntrain_folds', type=int, default=3, help='Number of training folds')
    parser.add_argument('--rotation', type=int, nargs='+', default=[0, 1], help='Rotation in the k-fold validation. Only used for conducting a single experiment.')
    
    parser.add_argument('--correct_lon', action='store_true', help='Correct longitude values of the raw dataset')
    parser.add_argument('--keras', action='store_true', help='Use the keras package to make the ML model. Must be usd for NNs, and must be not used otherwise.')
    parser.add_argument('--feature_importance', action='store_true', help='Collect the importance of each feature from the experiment. Is not available for all ML methods.')
    
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy', 'auc'], help='Metric(s) used to evaluate the ML model')
    parser.add_argument('--roc_curve', action='store_true', help='Calculate and plot ROC curves for the ML models')
    parser.add_argument('--class_weight', type=float, default=None, help='Class imbalance of non-FD events to FD events')
    
    
    # Random forest parameters
    parser.add_argument('--n_trees', type=int, default=100, help='Number of trees in the random forest')
    parser.add_argument('--tree_criterion', type=str, default='gini', help='Function used to judge the best split in each tree')
    parser.add_argument('--tree_depth', type=int, default=5, help='Maximum depth of each tree in the forest')
    parser.add_argument('--tree_max_features', type=str, default='sprt', help='Method of determining the maximum number of features for each split')
    
    # Support vector machine parameters 
    parser.add_argument('--svm_regularizer', type=str, default='l2', help='Must be l1 or l2. The L1 or L2 regularizer used to regularize the SVM.')
    parser.add_argument('--svm_regularizer_value', type=float, default=1e-4, help='The value of the L1/L2 regularization parameter used for the SVM.')
    parser.add_argument('--svm_loss', type=str, default='squared_hinge', help='Loss function for the SVMs. Must be hinge or squared_hinged.')
    parser.add_argument('--svm_stopping_err', type=float, default=1e-3, help='The stopping error to end SVM training')
    parser.add_argument('--svm_max_iter', type=float, default=1000, help='Maximum number of iterations/epochs before ending SVM training')
    parser.add_argument('--svm_intercept', action='store_true', help='Also train an intercept parameter when training the SVMs.')
    parser.add_argument('--svm_intercept_scale', type=float, default=1, help='Scaling factor of the SVM intercepts.')
    parser.add_argument('--svm_kernel', type=str, default='linear', help='Kernel type used to solve the SVMs.')
    
    # Ada Boosted tree
    parser.add_argument('--ada_n_estimators', type=int, default = 50, help='Number of estimators in the Ada Boosted trees.')
    parser.add_argument('--ada_learning_rate', type=float, default=1e-2, help='Learning rate for Ada Boosted trees')
    
    return parser


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
    with open("%s/%s"%(path, fname), "rb") as fp:
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
    train_folds = (np.arange(ntrain_folds) + rotation) % Nfolds
    validation_folds = int((np.array([ntrain_folds]) + rotation) % Nfolds)
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
    # print(train.shape, validation.shape, test.shape)
    
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
    #fname = '%s_%s_%s_%s_%s-%slat_%s_%slon'%(model, 
    #                                               ml_model, 
    #                                               method, 
    #                                               rotation, 
    #                                               lat_labels[0], lat_labels[1], 
    #                                               lon_labels[0], lon_labels[1])
    
    fname = '%s_%s_%s_rot_%s_lat_%s-%s_lon_%s_%s'%(model, 
                                                   ml_model, 
                                                   method, 
                                                   rotation, 
                                                   lat_labels[0], lat_labels[1], 
                                                   lon_labels[0], lon_labels[1])
    
    return fname
    
def generate_results_fname(model, ml_model, method, keras):
    '''
    Generate a filaname the results of a ML model will be saved to. Results are differentiated by reanalysis trained on, ML model, FD method
    
    Inputs:
    :param model: Reanalyses model the ML model is trained on
    :param ml_model: The ML model being saved
    :param method: The FD identification method used for labels
    :param keras: Boolean indicating whether a file name for a keras model is being saved
    
    Outputs:
    :param fname: The filename the ML results will be saved to
    '''
    
    # Create the filename
    fname = '%s_%s_%s_results'%(model, ml_model, method)
    
    if keras:
        return fname
    else:
        return '%s.pkl'%fname
    
    
    
#%%
##############################################

# Functions to conduct a single experiment
def execute_sklearn_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname):
    '''
    Run a single ML experiment from the sklearn package and save the model
    
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
    
    # sklearn models only accept 2D inputs and 1D outputs
    # Reshape data
    NV, T, IJ = train_in.shape
    NV, Tt, IJ = valid_in.shape

    train_in = train_in.reshape(NV, T*IJ, order = 'F')
    valid_in = valid_in.reshape(NV, Tt*IJ, order = 'F')
    test_in = test_in.reshape(NV, Tt*IJ, order = 'F')

    train_out = train_out.reshape(T*IJ, order = 'F')
    valid_out = valid_out.reshape(Tt*IJ, order = 'F')
    test_out = test_out.reshape(Tt*IJ, order = 'F')

    # Build the model
    model = build_sklearn_model(args)

    # Train the model
    model.fit(train_in.T, train_out)
    
    
    # Collect model results
    results = {}
    
    # Model predictions
    if (args.ml_model.lower() == 'svm') | (args.ml_model.lower() == 'support_vector_machine'):
        # Note SVMs do not have a predict_proba option
        results['train_predict'] = model.predict(train_in.T).reshape(T, IJ, order = 'F')
        results['valid_predict'] = model.predict(valid_in.T).reshape(Tt, IJ, order = 'F')
        results['test_predict'] = model.predict(test_in.T).reshape(Tt, IJ, order = 'F')
        
        # Check if the model learns to only predict 0s
        only_zeros = np.nansum(results['train_predict']) == 0
    else:
        # Check if the model learns to only predict 0s
        only_zeros = model.predict_proba(train_in.T).shape[1] <= 1
        
        if only_zeros:
            results['train_predict'] = np.zeros((T, IJ))
            results['valid_predict'] = np.zeros((Tt, IJ))
            results['test_predict'] = np.zeros((Tt, IJ))

        else:
            results['train_predict'] = model.predict_proba(train_in.T)[:,1].reshape(T, IJ, order = 'F')
            results['valid_predict'] = model.predict_proba(valid_in.T)[:,1].reshape(Tt, IJ, order = 'F')
            results['test_predict'] = model.predict_proba(test_in.T)[:,1].reshape(Tt, IJ, order = 'F')
        
    # Model performance

    eval_train = []
    eval_valid = []
    eval_test = []

    for metric in args.metrics:
        train_pred = model.predict(train_in.T)
        valid_pred = model.predict(valid_in.T)
        test_pred = model.predict(test_in.T)

        if metric == 'accuracy':
            e_train = metrics.accuracy_score(train_out, train_pred)
            e_valid = metrics.accuracy_score(valid_out, valid_pred)
            e_test = metrics.accuracy_score(test_out, test_pred)

        elif metric == 'auc':
            train_pred = model.decision_function(train_in.T)
            valid_pred = model.decision_function(valid_in.T)
            test_pred = model.decision_function(test_in.T)
            
            e_train = np.nan if only_zeros else metrics.roc_auc_score(train_out, train_pred)
            e_valid = np.nan if (np.nansum(valid_out) == 0) | (np.nansum(valid_out) == valid_out.size) else metrics.roc_auc_score(valid_out, valid_pred)
            e_test = np.nan if (np.nansum(test_out) == 0) | (np.nansum(test_out) == test_out.size) else metrics.roc_auc_score(test_out, test_pred)

        elif metric == 'precision':
            e_train = metrics.precision_score(train_out, train_pred)
            e_valid = metrics.precision_score(valid_out, valid_pred)
            e_test = metrics.precision_score(test_out, test_pred)

        elif metric == 'recall':
            e_train = metrics.recall_score(train_out, train_pred)
            e_valid = metrics.recall_score(valid_out, valid_pred)
            e_test = metrics.recall_score(test_out, test_pred)

        elif metric == 'f1_score':
            e_train = metrics.f1_score(train_out, train_pred)
            e_valid = metrics.f1_score(valid_out, valid_pred)
            e_test = metrics.f1_score(test_out, test_pred)
            
        elif metric == 'mse':
            e_train = metrics.mean_squared_error(train_out, train_pred)
            e_valid = metrics.mean_squared_error(valid_out, valid_pred)
            e_test = metrics.mean_squared_error(test_out, test_pred)
            
        elif metric == 'mae': # Might also add Cross-entropy
            e_train = metrics.mean_absolute_error(train_out, train_pred)
            e_valid = metrics.mean_absolute_error(valid_out, valid_pred)
            e_test = metrics.mean_absolute_error(test_out, test_pred)
            
        elif metric == 'tss':
            # To get the true skill score, determine the true positives, true negatives, and false positives (summed over time and the region)
            tp_train = np.nansum(np.where((train_pred == 1) & (train_out == 1), 1, 0))
            tp_valid = np.nansum(np.where((valid_pred == 1) & (valid_out == 1), 1, 0))
            tp_test = np.nansum(np.where((test_pred == 1) & (test_out == 1), 1, 0))
            
            tn_train = np.nansum(np.where((train_pred == 0) & (train_out == 0), 1, 0))
            tn_valid = np.nansum(np.where((valid_pred == 0) & (valid_out == 0), 1, 0))
            tn_test = np.nansum(np.where((test_pred == 0) & (test_out == 0), 1, 0))
            
            fp_train = np.nansum(np.where((train_pred == 1) & (train_out == 0), 1, 0))
            fp_valid = np.nansum(np.where((valid_pred == 1) & (valid_out == 0), 1, 0))
            fp_test = np.nansum(np.where((test_pred == 1) & (test_out == 0), 1, 0))
            
            # Obtain the TPR and FPR
            tpr_train = tp_train/(tp_train + fp_train)
            tpr_valid = tp_valid/(tp_valid + fp_valid)
            tpr_test = tp_test/(tp_test + fp_test)
            
            fpr_train = fp_train/(tn_train + fp_train)
            fpr_valid = fp_valid/(tn_valid + fp_valid)
            fpr_test = fp_test/(tn_test + fp_test)
            
            # The true skill score (TSS) is the TPR - FPR
            e_train = tpr_train - fpr_train
            e_valid = tpr_valid - fpr_valid
            e_test = tpr_test - fpr_test


        eval_train.append(e_train)
        eval_valid.append(e_valid)
        eval_test.append(e_test)
        
    # Collect information for the ROC curve
    if args.roc_curve:
        train_pred = model.decision_function(train_in.T)
        valid_pred = model.decision_function(valid_in.T)
        test_pred = model.decision_function(test_in.T)
        
        thresh = np.arange(0, 2, 1e-4)
        thresh = np.round(thresh, 4)
        results['fpr_train'] = np.ones((thresh.size)) * np.nan
        results['tpr_train'] = np.ones((thresh.size)) * np.nan
        results['fpr_valid'] = np.ones((thresh.size)) * np.nan
        results['tpr_valid'] = np.ones((thresh.size)) * np.nan
        results['fpr_test'] = np.ones((thresh.size)) * np.nan
        results['tpr_test'] = np.ones((thresh.size)) * np.nan
        
        fpr_train, tpr_train, thresh_train = metrics.roc_curve(train_out, train_pred)
        
        fpr_valid, tpr_valid, thresh_valid = metrics.roc_curve(valid_out, valid_pred)
        
        fpr_test, tpr_test, thresh_test = metrics.roc_curve(test_out, test_pred)
        
        # Note this step is needed to ensure the ROC curves have the same length across all rotations and regions
        for n, t in enumerate(thresh):
            # Place each tpr/fpr with its corresponding threshold. Perform an average if there are multiple thresholds in in a small range.
            ind_train = np.where(t == np.round(thresh_train, 4))[0]
            ind_valid = np.where(t == np.round(thresh_valid, 4))[0]
            ind_test = np.where(t == np.round(thresh_test, 4))[0]
            
            results['fpr_train'][n] = np.nanmean(fpr_train[ind_train])
            results['tpr_train'][n] = np.nanmean(tpr_train[ind_train])
            
            results['fpr_valid'][n] = np.nanmean(fpr_valid[ind_valid])
            results['tpr_valid'][n] = np.nanmean(tpr_valid[ind_valid])
            
            results['fpr_test'][n] = np.nanmean(fpr_test[ind_test])
            results['tpr_test'][n] = np.nanmean(tpr_test[ind_test])
        
            
    results['train_eval'] = eval_train
    results['valid_eval'] = eval_valid
    results['test_eval'] = eval_test

    # Collect the feature importance
    if args.feature_importance:
        results['feature_importance'] = np.array(model.feature_importances_)
    
    # Save the model
    with open('%s.pkl'%model_fname, 'wb') as fn:
        pickle.dump(model, fn)
        
    return results
    

def execute_keras_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname):
    '''
    Run a single ML experiment from the sklearn package and save the model
    
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
    model = build_keras_model(train_in, valid_in, args)
        
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
     
    # Collect the feature importance
    if args.feature_importance:
        pass

    
    # Save the model
    model.save(model_fname)
        
    return results
    
    
    
    
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
    
    # Execute the experiment based on whether it is an sklearn model or NN
    if args.keras:
        results = execute_keras_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname)
        
    else:
        results = execute_sklearn_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, model_fname)
    
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
    dataset_dir = '%s/%s'%(args.dataset, args.ra_model)
    
    print('Loading data...')
    
    # Load the data
    # Data is Nfeatures/Nmethods x time x space x fold
    data_in = load_ml_data(args.input_data_fname, path = dataset_dir)
    data_out = load_ml_data(args.output_data_fname, path = dataset_dir)
    
    # Remove NaNs?
    if args.remove_nans:
        data_in[np.isnan(data_in)] = 0
        data_out[np.isnan(data_out)] = 0
    
    print('Input size (NVariables x time x space x NFolds):', data_in.shape)
    print('Output size (NMethods x time x space x NFolds):', data_out.shape)
    
    # Make the rotations
    NVar, T, IJ, Nfold  = data_in.shape
    Nmethods = data_out.shape[0]
    rotation = np.arange(Nfold)
    
    # Make the latitude/longitude labels for individual regions
    lat_labels = np.arange(-90, 90+5, 5)
    lon_labels = np.arange(-180, 180+5, 5)
    
    # Load and reshape lat/lon data
    # Load lat and lon data
    
    # Load example data with subsetted lat/lon data
    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
    lat = et['lat']; lon = et['lon']
    
    # Correct the longitude?
    if args.correct_lon:
        print('Correcting longitude...')
        for n in range(len(lon[:,0])):
            ind = np.where(lon[n,:] > 0)[0]
            lon[n,ind] = -1*lon[n,ind]
            
    # Reshape the latitude/longitude data
    I, J = lat.shape
    lat1d = lat.reshape(I*J, order = 'F')
    lon1d = lon.reshape(I*J, order = 'F')
    
    I_lab = lat_labels.size
    J_lab = lon_labels.size
    
    
    # Split the data into regions.
    print('Splitting the data into regions...')
    
    data_in_split = []
    data_out_split = []
    lat_lab = []
    lon_lab = []
    for i in range(I_lab-1):
        for j in range(J_lab-1):
            ind = np.where( ((lat1d >= lat_labels[i]) & (lat1d <= lat_labels[i+1])) & ((lon1d >= lon_labels[j]) & (lon1d <= lon_labels[j+1])) )[0]
            
            # Not all datasets are global; remove sets where there is no data
            if len(ind) < 1: 
                continue
            
            lat_lab.append(lat_labels[i])
            lon_lab.append(lon_labels[j])
            
            data_in_split.append(data_in[:,:,ind,:])
            data_out_split.append(data_out[:,:,ind,:])
            
    # Save the lat/lon labels used for future use (needed to load the models later on)
    with open('%s/lat_lon_labels.pkl'%(dataset_dir), 'wb') as fn:
        pickle.dump(lat_lab, fn)
        pickle.dump(lon_lab, fn)
        
    print('There are %d regions.'%len(data_in_split))
            
    # Collect the lat/lon labels used in the regions
    ind_lat = np.where( (lat_labels >= lat_lab[0]) & (lat_labels <= lat_lab[-1]) )[0]
    ind_lon = np.where( (lon_labels >= lon_lab[0]) & (lon_labels <= lon_lab[-1]) )[0]
    
    lat_labels = lat_labels[ind_lat]
    lon_labels = lon_labels[ind_lon]
            
    I_lab = lat_labels.size
    J_lab = lon_labels.size
    
    # Loop over all FD methods to conduct the set of experiments for all of them
    for method in range(Nmethods):
        
        results_fbase = generate_results_fname(args.ra_model, args.label, methods[method], args.keras)
        results_fname = '%s/%s'%(dataset_dir, results_fbase)
        
        # Determine if this experiment has already been done
        if os.path.exists(results_fname):
            # Processed file does exist: continue
            print("File %s already exists"%feature_fname)
            continue
        
        # Initialize results
        pred_train = np.ones((args.ntrain_folds*T, I*J)) * np.nan
        pred_valid = np.ones((T, I*J)) * np.nan
        pred_test = np.ones((T, I*J)) * np.nan
        
        pred_train_var = np.ones((args.ntrain_folds*T, I*J)) * np.nan
        pred_valid_var = np.ones((T, I*J)) * np.nan
        pred_test_var = np.ones((T, I*J)) * np.nan
        
        
        eval_train = np.ones((I_lab*J_lab)) * np.nan
        eval_valid = np.ones((I_lab*J_lab)) * np.nan
        eval_test = np.ones((I_lab*J_lab)) * np.nan
        
        eval_train_var = np.ones((I_lab*J_lab)) * np.nan
        eval_valid_var = np.ones((I_lab*J_lab)) * np.nan
        eval_test_var = np.ones((I_lab*J_lab)) * np.nan
        
        
        if args.roc_curve:
            fpr_train = []
            fpr_valid = []
            fpr_test = []

            fpr_train_var = []
            fpr_valid_var = []
            fpr_test_var = []

            tpr_train = []
            tpr_valid = []
            tpr_test = []

            tpr_train_var = []
            tpr_valid_var = []
            tpr_test_var = []
        
        
        if args.feature_importance:
            feature_import = np.ones((I_lab,J_lab,NVar)) * np.nan
            feature_import_var = np.ones((I_lab,J_lab,NVar)) * np.nan

        
        if args.keras:
            learn_curves = np.ones((T, I*J)) * np.nan
            learn_curves_var = np.ones((T, I*J)) * np.nan
        
        # Begin looping and performing an experiment over all regions
        for n, (region_in, region_out) in enumerate(zip(data_in_split, data_out_split)):
            if n%10 == 0:
                print('Training a %s for the %dth region with the %s method...'%(args.ml_model, n+1, methods[method]))
            
            # Find where the current latitude and longitude values are
            ind = np.where( ((lat1d >= lat_lab[n]) & (lat1d <= lat_lab[n]+5)) & ((lon1d >= lon_lab[n]) & (lon1d <= lon_lab[n]+5)) )[0]
            
            # Initialize some lists
            ptrain = []
            pvalid = []
            ptest = []
            
            etrain = []
            evalid = []
            etest = []
            
            lc = []
            fi = []
            
            fpr_train_tmp = []
            fpr_valid_tmp = []
            fpr_test_tmp = []

            tpr_train_tmp = []
            tpr_valid_tmp = []
            tpr_test_tmp = []
            
            # For each region, perform an experiment for each rotation; obtain a statistical sample
            for rot in rotation:
                
                # Split the data into training, validation, and test sets
                train_in, valid_in, test_in = split_data(region_in, args.ntrain_folds, rot, normalize = args.normalize)
                train_out, valid_out, test_out = split_data(region_out, args.ntrain_folds, rot, normalize = False) # Note the label data is already binary
                
                if np.nansum(train_out == 1) == 0:
                    print('No FD in the current training set.')
                    continue
                
                # Generate the model filename
                model_fbase = generate_model_fname(args.ra_model, args.label, methods[method], rot, 
                                                   [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
                model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, method, model_fbase)
                
                # Perform the experiment
                # The ML model is saved in this step
                results = execute_single_exp(args, train_in, valid_in, test_in, 
                                             train_out[method,:,:], valid_out[method,:,:], test_out[method,:,:], model_fname)
                
                # Collect the results for the rotation
                ptrain.append(results['train_predict'])
                pvalid.append(results['valid_predict'])
                ptest.append(results['test_predict'])
                
                etrain.append(np.array(results['train_eval']))
                evalid.append(np.array(results['valid_eval']))
                etest.append(np.array(results['test_eval']))
                
                if args.roc_curve:
                    fpr_train_tmp.append(results['fpr_train'])
                    fpr_valid_tmp.append(results['fpr_valid'])
                    fpr_test_tmp.append(results['fpr_test'])

                    tpr_train_tmp.append(results['tpr_train'])
                    tpr_valid_tmp.append(results['tpr_valid'])
                    tpr_test_tmp.append(results['tpr_test'])
                
                if args.feature_importance:
                    fi.append(results['feature_importance'])
                
                if args.keras:
                    lc.append(results['history'])
               
            # Check if there was any learning
            if len(ptrain) < 1:
                print('No FD was found in that region; no ML models were trained.')
                continue
            
            # At the end of the experiments for each rotation, stack the reults into a single array (per variable) and average along the rotation axis
            if n%10 == 0:
                print('Evaluating the %s for the %dth region with the %s method...'%(args.ml_model, n+1, method))
          
            pred_train[:,ind] = np.nanmean(np.stack(ptrain, axis = -1), axis = -1)
            pred_valid[:,ind] = np.nanmean(np.stack(pvalid, axis = -1), axis = -1)
            pred_test[:,ind] = np.nanmean(np.stack(ptest, axis = -1), axis = -1)
            
            pred_train_var[:,ind] = np.nanstd(np.stack(ptrain, axis = -1), axis = -1)
            pred_valid_var[:,ind] = np.nanstd(np.stack(pvalid, axis = -1), axis = -1)
            pred_test_var[:,ind] = np.nanstd(np.stack(ptest, axis = -1), axis = -1)
            
            eval_train[n] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
            eval_valid[n] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
            eval_test[n] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
            
            eval_train_var[n] = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
            eval_valid_var[n] = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
            eval_test_var[n] = np.nanstd(np.stack(etest, axis = -1), axis = -1)
            
            if args.roc_curve:
                fpr_train.append(np.nanmean(np.stack(fpr_train_tmp, axis = -1), axis = -1))
                fpr_valid.append(np.nanmean(np.stack(fpr_valid_tmp, axis = -1), axis = -1))
                fpr_test.append(np.nanmean(np.stack(fpr_test_tmp, axis = -1), axis = -1))

                fpr_train_var.append(np.nanstd(np.stack(fpr_train_tmp, axis = -1), axis = -1))
                fpr_valid_var.append(np.nanstd(np.stack(fpr_valid_tmp, axis = -1), axis = -1))
                fpr_test_var.append(np.nanstd(np.stack(fpr_test_tmp, axis = -1), axis = -1))

                tpr_train.append(np.nanmean(np.stack(tpr_train_tmp, axis = -1), axis = -1))
                tpr_valid.append(np.nanmean(np.stack(tpr_valid_tmp, axis = -1), axis = -1))
                tpr_test.append(np.nanmean(np.stack(tpr_test_tmp, axis = -1), axis = -1))

                tpr_train_var.append(np.nanstd(np.stack(tpr_train_tmp, axis = -1), axis = -1))
                tpr_valid_var.append(np.nanstd(np.stack(tpr_valid_tmp, axis = -1), axis = -1))
                tpr_test_var.append(np.nanstd(np.stack(tpr_test_tmp, axis = -1), axis = -1))
            
            if args.feature_importance:
                feature_import[ind_lat,ind_lon,:] = np.nanmean(np.stack(fi, axis = -1), axis = -1)
                feature_import_var[ind_lat,ind_lon,:] = np.nanstd(np.stack(fi, axis = -1), axis = -1)
            
            if args.keras:
                learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
                learn_curves_var[:,ind] = np.nanstd(np.stack(lc, axis = -1), axis = -1)
            
        # At the end of the experiments for each region, collect the results and save the results
        print('Saving the results of the %s for the %s method...'%(args.ml_model, methods[method]))
        
        results = {}
        
        # Model predictions
        results['train_predict'] = pred_train.reshape(args.ntrain_folds*T, I, J, order = 'F')
        results['valid_predict'] = pred_valid.reshape(T, I, J, order = 'F')
        results['test_predict'] = pred_test.reshape(T, I, J, order = 'F')
        
        results['train_predict_var'] = pred_train_var.reshape(args.ntrain_folds*T, I, J, order = 'F')
        results['valid_predict_var'] = pred_valid_var.reshape(T, I, J, order = 'F')
        results['test_predict_var'] = pred_test_var.reshape(T, I, J, order = 'F')
        
        # Model performance
        results['train_eval'] = eval_train
        results['valid_eval'] = eval_valid
        results['test_eval'] = eval_test
        
        results['train_eval_var'] = eval_train_var
        results['valid_eval_var'] = eval_valid_var
        results['test_eval_var'] = eval_test_var
        
        # ROC curve (note this these are spatial means, and the mean variation in space)
        if args.roc_curve:
            results['fpr_train'] = np.nanmean(np.stack(fpr_train, axis = -1), axis = -1)
            results['fpr_valid'] = np.nanmean(np.stack(fpr_valid, axis = -1), axis = -1)
            results['fpr_test'] = np.nanmean(np.stack(fpr_test, axis = -1), axis = -1)

            results['fpr_train_var'] = np.nanmean(np.stack(fpr_train_var, axis = -1), axis = -1)
            results['fpr_valid_var'] = np.nanmean(np.stack(fpr_valid_var, axis = -1), axis = -1)
            results['fpr_test_var'] = np.nanmean(np.stack(fpr_test_var, axis = -1), axis = -1)

            results['tpr_train'] = np.nanmean(np.stack(tpr_train, axis = -1), axis = -1)
            results['tpr_valid'] = np.nanmean(np.stack(tpr_valid, axis = -1), axis = -1)
            results['tpr_test'] = np.nanmean(np.stack(tpr_test, axis = -1), axis = -1)

            results['tpr_train_var'] = np.nanmean(np.stack(tpr_train_var, axis = -1), axis = -1)
            results['tpr_valid_var'] = np.nanmean(np.stack(tpr_valid_var, axis = -1), axis = -1)
            results['tpr_test_var'] = np.nanmean(np.stack(tpr_test_var, axis = -1), axis = -1)
        
        # Feature importance
        if args.feature_importance:
            results['feature_importance'] = feature_import
            results['feature_importance_var'] = feature_import_var
        
        # Learning curves/model history
        if args.keras:
            results['history'] = learn_curves
            results['history_var'] = learn_curves_var
        
        # Save the results
        with open("%s"%(results_fname), "wb") as fp:
            pickle.dump(results, fp)
          
    print('Done.')
    


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
    dataset_dir = '%s/%s'%(args.dataset, args.ra_model)
    
    print('Loading data...')
    
    # Load the data
    # Data is Nfeatures/Nmethods x time x space x fold
    data_in = load_ml_data(args.input_data_fname, path = dataset_dir)
    data_out = load_ml_data(args.output_data_fname, path = dataset_dir)
    data_out[data_out <= 0] = 0 # Remove once corrected
    
    # Remove NaNs?
    if args.remove_nans:
        data_in[np.isnan(data_in)] = -995
        data_out[np.isnan(data_out)] = 0
    
    print('Input size (NVariables x time x space x NFolds):', data_in.shape)
    print('Output size (NMethods x time x space x NFolds):', data_out.shape)
    
    # Make the rotations
    NVar, T, IJ, Nfold  = data_in.shape
    Nmethods = data_out.shape[0]
    
    # Make the latitude/longitude labels for individual regions
    lat_labels = np.arange(-90, 90+5, 5)
    lon_labels = np.arange(-180, 180+5, 5)
    
    # Load and reshape lat/lon data
    # Load lat and lon data
    
    # Load example data with subsetted lat/lon data
    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
    lat = et['lat']; lon = et['lon']
    
    # Correct the longitude?
    if args.correct_lon:
        print('Correcting longitude...')
        for n in range(len(lon[:,0])):
            ind = np.where(lon[n,:] > 0)[0]
            lon[n,ind] = -1*lon[n,ind]
            
    # Reshape the latitude/longitude data
    I, J = lat.shape
    lat1d = lat.reshape(I*J, order = 'F')
    lon1d = lon.reshape(I*J, order = 'F')
    
    I_lab = lat_labels.size
    J_lab = lon_labels.size
    
    
    # Split the data into regions.
    print('Splitting the data into regions...')
    
    data_in_split = []
    data_out_split = []
    lat_lab = []
    lon_lab = []
    for i in range(I_lab-1):
        for j in range(J_lab-1):
            ind = np.where( ((lat1d >= lat_labels[i]) & (lat1d <= lat_labels[i+1])) & ((lon1d >= lon_labels[j]) & (lon1d <= lon_labels[j+1])) )[0]
            
            # Not all datasets are global; remove sets where there is no data
            if len(ind) < 1: 
                continue
            
            lat_lab.append(lat_labels[i])
            lon_lab.append(lon_labels[j])
            
            data_in_split.append(data_in[:,:,ind,:])
            data_out_split.append(data_out[:,:,ind,:])
            
    # Save the lat/lon labels used for future use (needed to load the models later on)
    with open('%s/lat_lon_labels.pkl'%(dataset_dir), 'wb') as fn:
        pickle.dump(lat_lab, fn)
        pickle.dump(lon_lab, fn)
          
    print('There are %d regions.'%len(data_in_split))
        
        
    # Collect the lat/lon labels used in the regions
    ind_lat = np.where( (lat_labels >= lat_lab[0]) & (lat_labels <= lat_lab[-1]) )[0]
    ind_lon = np.where( (lon_labels >= lon_lab[0]) & (lon_labels <= lon_lab[-1]) )[0]
    
    lat_labels = lat_labels[ind_lat]
    lon_labels = lon_labels[ind_lon]
            
    I_lab = lat_labels.size
    J_lab = lon_labels.size
    

    # Initialize results
    pred_train = np.ones((args.ntrain_folds*T, I*J)) * np.nan
    pred_valid = np.ones((T, I*J)) * np.nan
    pred_test = np.ones((T, I*J)) * np.nan
    
    pred_train_var = np.ones((args.ntrain_folds*T, I*J)) * np.nan
    pred_valid_var = np.ones((T, I*J)) * np.nan
    pred_test_var = np.ones((T, I*J)) * np.nan
    
    
    eval_train = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    eval_valid = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    eval_test = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    
    eval_train_var = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    eval_valid_var = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    eval_test_var = np.ones((I_lab,J_lab,len(args.metrics))) * np.nan
    
    if args.roc_curve:
        fpr_train = []
        fpr_valid = []
        fpr_test = []
        
        fpr_train_var = []
        fpr_valid_var = []
        fpr_test_var = []
        
        tpr_train = []
        tpr_valid = []
        tpr_test = []
        
        tpr_train_var = []
        tpr_valid_var = []
        tpr_test_var = []
    
    if args.feature_importance:
        feature_import = np.ones((I_lab,J_lab,NVar)) * np.nan
        feature_import_var = np.ones((I_lab,J_lab,NVar)) * np.nan
    
    if args.keras:
        learn_curves = np.ones((T, I*J)) * np.nan
        learn_curves_var = np.ones((T, I*J)) * np.nan
    
    # Begin looping and performing an experiment over all regions
    for n, (region_in, region_out) in enumerate(zip(data_in_split, data_out_split)):
        if n%10 == 0:
                print('Training a %s for the %dth region with the %s method...'%(args.ml_model, n+1, methods[0]))
          
        # Find where the current latitude and longitude values are
        ind = np.where( ((lat1d >= lat_lab[n]) & (lat1d <= lat_lab[n]+5)) & ((lon1d >= lon_lab[n]) & (lon1d <= lon_lab[n]+5)) )[0]
        
        ind_lat = np.where(lat_labels == lat_lab[n])[0]
        ind_lon = np.where(lon_labels == lon_lab[n])[0]
        
        # Initialize some lists
        ptrain = []
        pvalid = []
        ptest = []
        
        etrain = []
        evalid = []
        etest = []
        
        lc = []
        fi = []
        
        fpr_train_tmp = []
        fpr_valid_tmp = []
        fpr_test_tmp = []
        
        tpr_train_tmp = []
        tpr_valid_tmp = []
        tpr_test_tmp = []
        
        # For each region, perform an experiment for several rotations; obtain a statistical sample
        for rot in args.rotation:
            
            # Split the data into training, validation, and test sets
            train_in, valid_in, test_in = split_data(region_in, args.ntrain_folds, rot, normalize = args.normalize)
            train_out, valid_out, test_out = split_data(region_out, args.ntrain_folds, rot, normalize = False) # Note the label data is already binary
            
            if np.nansum(train_out == 1) == 0:
                # print('No FD in the current training set.')
                continue
            
            # Generate the model filename
            model_fbase = generate_model_fname(args.ra_model, args.label, methods[0], rot, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
            model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, methods[0], model_fbase)
            # print(model_fname)
            
            
            # Perform the experiment
            # The ML model is saved in this step
            results = execute_single_exp(args, train_in, valid_in, test_in, 
                                         train_out[0,:,:], valid_out[0,:,:], test_out[0,:,:], model_fname)
            
            # Collect the results for the rotation
            ptrain.append(results['train_predict'])
            pvalid.append(results['valid_predict'])
            ptest.append(results['test_predict'])
            
            etrain.append(results['train_eval'])
            evalid.append(results['valid_eval'])
            etest.append(results['test_eval'])
            
            if args.roc_curve:
                fpr_train_tmp.append(results['fpr_train'])
                fpr_valid_tmp.append(results['fpr_valid'])
                fpr_test_tmp.append(results['fpr_test'])
                
                tpr_train_tmp.append(results['tpr_train'])
                tpr_valid_tmp.append(results['tpr_valid'])
                tpr_test_tmp.append(results['tpr_test'])
            
            if args.feature_importance:
                fi.append(results['feature_importance'])
            
            if args.keras:
                lc.append(results['history'])
            
        # Check if there was any learning
        if len(ptrain) < 1:
            # print('No FD was found in that region; no ML models were trained.')
            continue
            
        # At the end of the experiments for each rotation, stack the reults into a single array (per variable) and average along the rotation axis
        if n%10 == 0:
                print('Evaluating the %s for the %dth region with the %s method...'%(args.ml_model, n+1, methods[0]))
          
        pred_train[:,ind] = np.nanmean(np.stack(ptrain, axis = -1), axis = -1)
        pred_valid[:,ind] = np.nanmean(np.stack(pvalid, axis = -1), axis = -1)
        pred_test[:,ind] = np.nanmean(np.stack(ptest, axis = -1), axis = -1)
        
        eval_train[ind_lat,ind_lon,:] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
        eval_valid[ind_lat,ind_lon,:] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
        eval_test[ind_lat,ind_lon,:] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
        
        eval_train_var[ind_lat,ind_lon,:] = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
        eval_valid_var[ind_lat,ind_lon,:] = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
        eval_test_var[ind_lat,ind_lon,:] = np.nanstd(np.stack(etest, axis = -1), axis = -1)
        
        if args.roc_curve:
            fpr_train.append(np.nanmean(np.stack(fpr_train_tmp, axis = -1), axis = -1))
            fpr_valid.append(np.nanmean(np.stack(fpr_valid_tmp, axis = -1), axis = -1))
            fpr_test.append(np.nanmean(np.stack(fpr_test_tmp, axis = -1), axis = -1))
            
            fpr_train_var.append(np.nanstd(np.stack(fpr_train_tmp, axis = -1), axis = -1))
            fpr_valid_var.append(np.nanstd(np.stack(fpr_valid_tmp, axis = -1), axis = -1))
            fpr_test_var.append(np.nanstd(np.stack(fpr_test_tmp, axis = -1), axis = -1))
            
            tpr_train.append(np.nanmean(np.stack(tpr_train_tmp, axis = -1), axis = -1))
            tpr_valid.append(np.nanmean(np.stack(tpr_valid_tmp, axis = -1), axis = -1))
            tpr_test.append(np.nanmean(np.stack(tpr_test_tmp, axis = -1), axis = -1))
            
            tpr_train_var.append(np.nanstd(np.stack(tpr_train_tmp, axis = -1), axis = -1))
            tpr_valid_var.append(np.nanstd(np.stack(tpr_valid_tmp, axis = -1), axis = -1))
            tpr_test_var.append(np.nanstd(np.stack(tpr_test_tmp, axis = -1), axis = -1))
            
        
        if args.feature_importance:
            feature_import[ind_lat,ind_lon,:] = np.nanmean(np.stack(fi, axis = -1), axis = -1)
            feature_import_var[ind_lat,ind_lon,:] = np.nanstd(np.stack(fi, axis = -1), axis = -1)
        
        if args.keras:
            learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
            learn_curves_var[:,ind] = np.nanstd(np.stack(lc, axis = -1), axis = -1)
        
    # At the end of the experiments for each region, collect the results and save the results
    print('Saving the results of %s for the %s method...'%(args.ml_model, methods[0]))
          
    results_fbase = generate_results_fname(args.ra_model, args.label, methods[0], args.keras)
    results_fname = '%s/%s'%(dataset_dir, results_fbase)
    print(results_fname)
    
    results = {}
    
    # Model predictions
    results['train_predict'] = pred_train.reshape(args.ntrain_folds*T, I, J, order = 'F')
    results['valid_predict'] = pred_valid.reshape(T, I, J, order = 'F')
    results['test_predict'] = pred_test.reshape(T, I, J, order = 'F')
    
    results['train_predict_var'] = pred_train_var.reshape(args.ntrain_folds*T, I, J, order = 'F')
    results['valid_predict_var'] = pred_valid_var.reshape(T, I, J, order = 'F')
    results['test_predict_var'] = pred_test_var.reshape(T, I, J, order = 'F')
    
    # Model performance
    results['train_eval'] = eval_train
    results['valid_eval'] = eval_valid
    results['test_eval'] = eval_test
    
    results['train_eval_var'] = eval_train_var
    results['valid_eval_var'] = eval_valid_var
    results['test_eval_var'] = eval_test_var
    
    results['eval_lon'], results['eval_lat'] = np.meshgrid(lon_labels, lat_labels)
    
    # ROC curve (note this these are spatial means, and the mean variation in space)
    if args.roc_curve:
        results['fpr_train'] = np.nanmean(np.stack(fpr_train, axis = -1), axis = -1)
        results['fpr_valid'] = np.nanmean(np.stack(fpr_valid, axis = -1), axis = -1)
        results['fpr_test'] = np.nanmean(np.stack(fpr_test, axis = -1), axis = -1)
        
        results['fpr_train_var'] = np.nanmean(np.stack(fpr_train_var, axis = -1), axis = -1)
        results['fpr_valid_var'] = np.nanmean(np.stack(fpr_valid_var, axis = -1), axis = -1)
        results['fpr_test_var'] = np.nanmean(np.stack(fpr_test_var, axis = -1), axis = -1)
        
        results['tpr_train'] = np.nanmean(np.stack(tpr_train, axis = -1), axis = -1)
        results['tpr_valid'] = np.nanmean(np.stack(tpr_valid, axis = -1), axis = -1)
        results['tpr_test'] = np.nanmean(np.stack(tpr_test, axis = -1), axis = -1)
        
        results['tpr_train_var'] = np.nanmean(np.stack(tpr_train_var, axis = -1), axis = -1)
        results['tpr_valid_var'] = np.nanmean(np.stack(tpr_valid_var, axis = -1), axis = -1)
        results['tpr_test_var'] = np.nanmean(np.stack(tpr_test_var, axis = -1), axis = -1)
    
    # Feature importance
    if args.feature_importance:
        results['feature_importance'] = feature_import
        results['feature_importance_var'] = feature_import_var
    
    # Learning curves/model history
    if args.keras:
        results['history'] = learn_curves
        results['history_var'] = learn_curves_var
        
    # Save the results
    with open("%s"%(results_fname), "wb") as fp:
        pickle.dump(results, fp)
          
        
    print('Done.')
    return results




#%%
##############################################
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_ml_parser()
    args = parser.parse_args()
    
    # Execute the experiments?
    if np.invert(args.nogo):
        print('Performing experiments...')
        execute_all_exp(args)
    
    # Begin visualizing model performance
    methods = ['christian', 'nogeura', 'pendergrass', 'liu', 'otkin']
    
    # Get the directory of the dataset
    dataset_dir = '%s/%s'%(args.dataset, args.model)
    
    print('Loading results...')
    results = []
    
    for method in methods:
        results_fname = generate_results_fname(args.ra_model, args.label, method, args.keras)
    
        # Load in all the datasets for all methods
        with open('%s/%s'%(dataset_dir, results_fname), 'rb') as fn:
            results.append(pickle.load(fn))
    
    # Note here that results[0] = christian; results[1] = nogeura; results[2] = pendergrass; rsults[3] = liue; results[4] = otkin
    
    # Obtain the latitude and longitude for metrics
    lat = results[0]['eval_lat']; lon = results[0]['eval_lon']


    # Plot the results of the metrics
    print('Plotting results...')
    for met, metric in enumerate(args.metrics):
        # Collect the metrics
        metrics_train = [results[m]['train_eval'][:,:,met] for m in range(len(methods))]
        metrics_valid = [results[m]['valid_eval'][:,:,met] for m in range(len(methods))]
        metrics_test = [results[m]['test_eval'][:,:,met] for m in range(len(methods))]
        
        if (metric == 'mse') | (metric == 'mae'):
            cmin = 0; cmax = 0.5; cint = 0.05
        else:
            cmin = 0; cmax = 1; cint = 0.1
    
        # Plot the metric
        display_metric_map(metrics_train, lat, lon, methods, 
                           metric, cmin, cmax, cint, args.ra_model, 
                           args.label, dataset = 'train', reverse = False, globe = args.globe, path = dataset_dir)
        
        display_metric_map(metrics_valid, lat, lon, methods, 
                           metric, cmin, cmax, cint, args.ra_model, 
                           args.label, dataset = 'valid', reverse = False, globe = args.globe, path = dataset_dir)
        
        display_metric_map(metrics_test, lat, lon, methods, 
                           metric, cmin, cmax, cint, args.ra_model, 
                           args.label, dataset = 'test', reverse = False, globe = args.globe, path = dataset_dir)
        
        # Remove variables at the end to clear space
        del metric_train, metric_valid, metric_test
    
    
    # Plot the ROC curves?
    if args.roc_curve:
        # Collect the ROC curve informations
        print('Making ROC curves...')
        tpr_train = [results[m]['tpr_train'] for m in range(len(methods))]
        tpr_valid = [results[m]['tpr_valid'] for m in range(len(methods))]
        tpr_test = [results[m]['tpr_test'] for m in range(len(methods))]
        
        tpr_var_train = [results[m]['tpr_train_var'] for m in range(len(methods))]
        tpr_var_valid = [results[m]['tpr_valid_var'] for m in range(len(methods))]
        tpr_var_test = [results[m]['tpr_test_var'] for m in range(len(methods))]
        
        fpr_train = [results[m]['fpr_train'] for m in range(len(methods))]
        fpr_valid = [results[m]['fpr_valid'] for m in range(len(methods))]
        fpr_test = [results[m]['fpr_test'] for m in range(len(methods))]
        
        fpr_var_train = [results[m]['fpr_train_var'] for m in range(len(methods))]
        fpr_var_valid = [results[m]['fpr_valid_var'] for m in range(len(methods))]
        fpr_var_test = [results[m]['fpr_test_var'] for m in range(len(methods))]
        
        # Plot the ROC curves
        display_roc_curves(tpr_train, fpr_train, tpr_var_train, fpr_var_train, 
                           methods, args.ra_model, args.label, dataset = 'train', path = dataset_dir)
        
        display_roc_curves(tpr_valid, fpr_valid, tpr_var_valid, fpr_var_valid, 
                           methods, args.ra_model, args.label, dataset = 'valid', path = dataset_dir)
        
        display_roc_curves(tpr_test, fpr_test, tpr_var_test, fpr_var_test, 
                           methods, args.ra_model, args.label, dataset = 'test', path = dataset_dir)
        
        # Remove variables at the end to clear space
        del tpr_train, tpr_valid, tpr_test, tpr_train_var, tpr_valid_var, tpr_test_var
        del fpr_train, fpr_valid, fpr_test, fpr_train_var, fpr_valid_var, fpr_test_var
        
        
        
    # Plot the feature importance?
    if args.feature_importance:
        features = ['T', 'ET', r'\Delta ET', 'PET', r'\Delta PET', 'P', 'SM', r'\Delta SM']
        I, J, NFeature = results[0]['feature_importance'].shape
        
        # Plot a map of feature importance for each features
        print('Making feature importance maps...')
        for n_feature in range(NFeature):
            fi = [results[m]['feature_importance'][:,:,n_feature] for m in range(len(methods))]
            fi_var = [results[m]['feature_importance_var'][:,:,n_feature] for m in range(len(methods))]
            
            display_metric_map(fi, lat, lon, methods, 
                               features[n_feature], 0, 1, 0.05, args.ra_model, 
                               args.label, dataset = 'feature_importance', reverse = False, globe = args.globe, path = dataset_dir)
            
            display_metric_map(fi_var, lat, lon, methods, 
                               features[n_feature], 0, 1, 0.05, args.ra_model, 
                               args.label, dataset = 'feature_importance_variation', reverse = False, globe = args.globe, path = dataset_dir)
            
        # Average the feature importance in space
        fi = [np.nanmean(results[m]['feature_importance'].reshape(I*J, NFeature)) for m in range(len(methods))]
        fi_var = [np.nanmean(results[m]['feature_importance_var'].reshape(I*J, NFeature)) for m in range(len(methods))]
        
        # Create a barplot of the overall feature variation
        print('Making overall feature importance barplot...')
        display_feature_importance(fi, fi_var, features, methods, args.ra_model, args.label, path = dataset_dir)
        
        # Remove variables at the end to clear space
        del fi, fi_var
        
        
    # Plot the learning curve?

    
    # Since some of these files are large, remove them from the namespace to ensure conserve memory
    del results, lat, lon
    gc.collect() # Clears deleted variables from memory 

    # Make predictions?
    if args.climatology_plot | args.time_series | args.case_studies | args.threat_score:
        # Make a dataset to make predictions with
        print('Loading data')
        data = load_ml_data(args.input_data_fname, path = '%s/%s'%(args.dataset, args.ra_model))
        evap = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = dataset_dir)

        ind = np.where( (evap['month'] >= 4) & (evap['month'] <= 10) )[0]
        dates = evap['ymd'][ind]
        
        I, J = evap['lat'].shape

        NVar, T, IJ, NFold = data.shape
        
        rotations = np.arange(NFold)
        
        # Make the predictions and plot the results for each method
        for method in methods:

            pred = np.ones((T * NFold, I, J)) * np.nan
            pred_var = np.ones((T * NFold, I, J)) * np.nan

            print('Making predictions for the %s method...'%method)
            # Make predictions based on one of the ML models
            for n in range(NFold):
                print(n)
                pred[n*T:(n+1)*T,:,:], pred_var[n*T:(n+1)*T,:,:] = make_predictions(data[:,:,:,n], 
                                                                                    evap['lat'], evap['lon'], 
                                                                                    probabilities = False, threshold = 0.5, keras = False, 
                                                                                    ml_model = args.ml_model, ra_model = args.ra_model, 
                                                                                    method = method, rotations = rotations, label = args.label, 
                                                                                    path = dataset_dir)
                
            # Load in the true labels?
            if args.time_series | args.case_studies | args.threat_score:
                print('Loading true labels for the %s method...'%method)
                true_fd = load_nc('fd', '%s.%s.pentad.nc'%(method, args.ra_model), path = dataset_dir)
                fd = true_fd['fd'][ind,:,:]


            # Plot the threat scores?
            if args.confusion_matrix_plots:
                plot('Plotting confusion matrix skill scores for the %s method...'%method)
                mask = load_mask(model = args.ra_model)

                display_threat_score(fd, pred, ch_fd['lat'], ch_fd['lon'], dates, mask, 
                                     model = args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)
                
                display_far_score(fd, pred, ch_fd['lat'], ch_fd['lon'], dates, 
                                  model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)
                
                display_pod_score(fd, pred, ch_fd['lat'], ch_fd['lon'], dates, 
                                  model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)


            # Plot the predicted climatology map?
            if args.climatology_plot:
                print('Plotting the predicted climatology for the the %s method...'%method)
                display_fd_climatology(pred, evap['lat'], evap['lon'], dates, 'Predicted FD for %s'%method, 
                                       model = '%s_%s'%(method, args.ra_model), path = dataset_dir, grow_season = True)


            # Plot the predicted time series (with true labels)?
            if args.time_series:
                print('Calculating areal coverage for the %s method...'%method)
                # Examine predicted time series
                T, I, J = pred.shape

                # Determine the areal coverage for the time series
                tmp_pred = np.nansum(pred.reshape(T, I*J), axis = -1)*32*32

                pred_area = []
                pred_area_var = []


                # Determine the true areal coverage
                tmp_true = np.nansum(fd.reshape(T, I*J), axis = -1)*32*32

                true_area = []
                true_area_var = []

                years = np.array([date.year for date in dates])

                # Determine the annual mean/std areal coverage of FD
                for year in np.unique(years):
                    ind = np.where(year == years)[0]

                    pred_area.append(np.nanmean(tmp_pred[ind]))
                    pred_area_var.append(np.nanstd(tmp_pred[ind]))

                    true_area.append(np.nanmean(tmp_true[ind]))
                    true_area_var.append(np.nanstd(tmp_true[ind]))

                pred_area = np.array(pred_area)
                pred_area_var = np.array(pred_area_var)

                true_area = np.array(true_area)
                true_area_var = np.array(true_area_var)


                # Display the time series
                print('Plotting the areal coverage of FD time series for the %s method...'%method)
                display_time_series(true_area, pred_area, true_area_var, pred_area_var, dates[::43], 
                                    r'Areal Coverage (km^2)', args.ra_model, '%s_%s'%(args.label, method), path = dataset_dir)

            # Plot a set of case studies?
            if args.case_studies:
                print('Plotting case studies for the %s method...'%method)

                # Plot the case studies for the predicted labels
                display_case_study_maps(pred, evap['lon'], evap['lat'], dates, args.case_study_years, 
                                        method = method, label = args.label, dataset = args.ra_model, 
                                        globe = False, path = dataset_dir, grow_season = True)

                # Plot the case studies for the true labels
                display_case_study_maps(fd, evap['lon'], evap['lat'], dates, args.case_study_years, 
                                        method = method, label = args.label, dataset = args.ra_model, 
                                        globe = False, path = dataset_dir, grow_season = True)
                
                
                
            
    print('Done')        
                    

