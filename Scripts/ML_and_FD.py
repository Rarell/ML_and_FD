#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 2 17:52:45 2021

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
#   1.0.0 - 12/24/2021 - Initial reformatting of the script to use a 'version' setting (note the version number is not in the script name, 
#                        so this is not a full version version control)
#   1.1.0 - 12/25/2021 - Implemented code to split the dataset into regions in the DetermineParameters and CreateSLModel functions
#   1.2.0 - 12/30/2021 - Modified the main functions to write outputs to .txt file instead of output to the terminal (easier to save everything)
#   1.3.0 - 1/13/2022 - Modified the ModelPredictions function to encorporate multiple regions.
#   2.0.1 - 9/05/2022 - Major modifications to the code structure and experiment design. Each growing season is now a fold, regions are based on 5 degree x 5 
#                       degree sections. sklearn section has been tested and is working.
#   2.0.2 - 10/22/2022 - Added more confusion table skill metrics. Finished adding standard ML approaches (includes RF, SVMs, and Ada boosted trees). Other
#                        minor corrections. Added predictions with test datasets only.
#   3.0.0 - 11/16/2022 - Removed splitting of data into regions to converse memory. Restructured execute_exp (now main experiment function) to train 
#                        1 rotation and 1 method at a time. execute_exp is now structured to work with the OU schooner supercomputer.
#
# Inputs:
#   - Data files for FD indices and identified FD (.nc format)
#
# Outputs:
#   - A number of figure showing the results of the SL algorithms
#   - Several outputs (in the terminal) showing performance metrics for the ML algorithms
#
# To Do:
#   - Main function
#       - Add learning curve calculations and figures
#   - May look into residual curves (training and validation metric performance over different number of folds; 
#                                    may be too computationally and temporally expensive)
#   - May look into Ceteris-Paribus effect
#   - Add argparse arguments for other ML models
#   - Might try a more effective approach to parallel processing for increased computation speed
#   - Keras models have not been tested
#
# Bugs:
#   - 
#
# Notes:
#   - See tf_environment.yml for a list of all packages and versions. netcdf4 and cartopy must be downloaded seperately.
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
import re
import argparse
import pickle
import numpy as np
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
    parser.add_argument('--method', type=str, default='christian', help='The FD identification method the ML model is being trained to recognize')
    
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
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the ML model after having run it for all rotations and FD identification methods')
    
    
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

def generate_model_fname(model, ml_model, method, rotation):
    '''
    Generate a filename for a ML model to be saved to. Models are differentiated by rotation, ML model trained, reanalysis model trained on, 
    and FD method trained to identify.
    
    Inputs:
    :param model: Reanalyses model the ML model is trained on
    :param ml_model: The ML model being saved
    :param method: The FD identification method used for labels
    :param rotation: Current rotation in the k-fold validation

    
    Outputs:
    :param fname: The filename the ML model will be saved to
    '''
    
    # Create the filename
    fname = '%s_%s_%s_rot_%s'%(model, 
                               ml_model, 
                               method, 
                               rotation)
    
    return fname
    
def generate_results_fname(model, ml_model, method, keras):
    '''
    Generate a filename the results (merged over all rotations) of a ML model will be saved to. 
    Results are differentiated by ML model trained, reanalysis model trained on, and FD method trained to identify.
    
    Inputs:
    :param model: Reanalyses model the ML model is trained on
    :param ml_model: The ML model being saved
    :param method: The FD identification method used for labels
    :param keras: Boolean indicating whether a file name for a keras model is being saved
    
    Outputs:
    :param fname: The filename the ML results will be saved to
    '''
    
    # Create the filename
    fname = '%s_%s_%s_merged_results'%(model, ml_model, method)
    
    if keras:
        return fname
    else:
        return '%s.pkl'%fname
    
    
    
#%%
##############################################

# Functions to conduct a single experiment
def execute_sklearn_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, data_in, data_out, rotation, 
                        model_fname, evaluate_each_grid = False):
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
    :param data_in: Entire input dataset. Results predict for the entire dataset instead of the training set (to simplify the merging step)
    :param data_out: Enitre output dataset. Results predict for the entire dataset instead of the training set (to simplify the merging step)
    :param rotation: Current rotation in the k-fold validation
    :param model_fname: The filename to save the model to
    :param evaluate_each_grid: Boolean indicating whether evaulation metrics should be examined for each grid point in the dataset
    
    Outputs:
    :param results: Dictionary results from the ML model, including predictions, performance metrics, and learning curves
    '''
    
    # Construct the base file name for the result
    results_fbase = 'results_%s_%s_%s_rot_%s'%(args.ra_model, 
                                               args.label, 
                                               args.method, 
                                               rotation)
    
    dataset_dir = '%s/%s'%(args.dataset, args.ra_model)
    results_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, args.method, results_fbase)
    
    # If the results exist, skip this experiment
    if os.path.exists('%s.pkl'%results_fname):
        print('The model has already been trained and results gathered.')
        return
    
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


    if os.path.exists('%s.pkl'%model_fname):
        # If the modeel exists, there is no need to make and train it
        try:
            with open('%s.pkl'%model_fname, 'rb') as fn:
                model = pickle.load(fn)
        except pickle.UnpicklingError:
            print('Could not load file: %s. Erasing and retraining it.'%model_fname)
            os.remove('%s.pkl'%model_fname)
            
            # Build the model
            model = build_sklearn_model(args)

            # Train the model
            model.fit(train_in.T, train_out)
            
            # Save the model
            with open('%s.pkl'%model_fname, 'wb') as fn:
                pickle.dump(model, fn)
                
    else:
        # Build the model
        model = build_sklearn_model(args)

        # Train the model
        model.fit(train_in.T, train_out)
        
        # Save the model
        with open('%s.pkl'%model_fname, 'wb') as fn:
            pickle.dump(model, fn)
    
    # Evaluate the model
    NV, T, IJ = data_in.shape
    data_in = data_in.reshape(NV, T*IJ, order = 'F')
    data_out = data_out.reshape(T*IJ, order = 'F')
    
    results = sklearn_evaluate_model(model, args, data_in, valid_in, test_in, data_out, valid_out, test_out, IJ, T, Tt)
    
    # Evaluate the model for each grid point?
    if evaluate_each_grid:
        
        # Initialize the gridded metrics
        eval_train_map = np.zeros((IJ, len(args.metrics))) * np.nan
        eval_valid_map = np.zeros((IJ, len(args.metrics))) * np.nan
        eval_test_map = np.zeros((IJ, len(args.metrics))) * np.nan
        
        data_in = data_in.reshape(NV, T, IJ, order = 'F')
        valid_in = valid_in.reshape(NV, Tt, IJ, order = 'F')
        test_in = test_in.reshape(NV, Tt, IJ, order = 'F')

        data_out = data_out.reshape(T, IJ, order = 'F')
        valid_out = valid_out.reshape(Tt, IJ, order = 'F')
        test_out = test_out.reshape(Tt, IJ, order = 'F')
        
        # Perform the evaluation for each grid point
        for ij in range(IJ):
            if np.mod(ij/IJ*100, 10) == 0:
                print('Currently %d through the spatial evaluation'%(int(ij/IJ*100)))
            
            if np.nansum(data_out[:,ij]) == 0:
                continue
            
            r_tmp = sklearn_evaluate_model(model, args, data_in[:,:,ij], valid_in[:,:,ij], test_in[:,:,ij], 
                                           data_out[:,ij], valid_out[:,ij], test_out[:,ij], None, T, Tt)
            
            # See if this clears any erased memory from running the function
            gc.collect() # Clears deleted variables from memory 
            
            # Try reducing the size of map data by half to reduce the RAM memory
            eval_train_map[ij,:] = np.float32(r_tmp['train_eval'])
            eval_valid_map[ij,:] = np.float32(r_tmp['valid_eval'])
            eval_test_map[ij,:] = np.float32(r_tmp['test_eval'])
            
            
        results['eval_train_map'] = eval_train_map
        results['eval_valid_map'] = eval_valid_map
        results['eval_test_map'] = eval_test_map
        
    # Save the results
    with open('%s.pkl'%results_fname, 'wb') as fn:
        pickle.dump(results, fn)
        
    return results
    
def sklearn_evaluate_model(model, args, train_in, valid_in, test_in, train_out, valid_out, test_out, IJ, T, Tt):
    '''
    Evaluate an sklearn machine learning (ML) model. Assumes the validation and test datasets have the same temporal dimension.
    
    Inputs:
    :param model: ML model being evaluated
    :param args: Argparse arguments
    :param train_in: Input training dataset used to train the model. Maybe the entire input dataset instead.
    :param valid_in: Input validation dataset to help validate the model
    :param test_in: Input testing dataset to test the model
    :param train_out: Output training dataset used to train the model. Maybe the entire output dataset instead.
    :param valid_out: Output validation dataset to help validate the model
    :param test_out: Output testing dataset to test the model
    :param IJ: Size of the total spatial dimension of the dataset
    :param T, Tt: Size of the total time dimension for train and valid/test datasets respectively
    
    Outputs:
    :param results: Dictionary results of evaluations for the ML model
    '''
    
    # Collect model results
    results = {}
        
    # Model performance

    eval_train = []
    eval_valid = []
    eval_test = []

    # Predictions
    train_pred = model.predict(train_in.T)
    valid_pred = model.predict(valid_in.T)
    test_pred = model.predict(test_in.T)
    
    only_zeros = (np.nansum(train_pred) == 0) | (np.nansum(train_out) == 0)
    
    # Perform the evaluations for each metric
    for metric in args.metrics:

        if metric == 'accuracy':
            e_train = metrics.accuracy_score(train_out, train_pred)
            e_valid = metrics.accuracy_score(valid_out, valid_pred)
            e_test = metrics.accuracy_score(test_out, test_pred)

        elif metric == 'auc':
            if (args.ml_model.lower() == 'rf') | (args.ml_model.lower() == 'random_forest'):
                train_pred_auc = model.predict_proba(train_in.T)
                valid_pred_auc = model.predict_proba(valid_in.T)
                test_pred_auc = model.predict_proba(test_in.T)
                
                # Test if the model predicts FD
                if train_pred_auc.shape[1] > 1:
                    train_pred_auc = train_pred_auc[:,1]
                else:
                    train_pred_auc = 0
                    
                if valid_pred_auc.shape[1] > 1:
                    valid_pred_auc = valid_pred_auc[:,1]
                else:
                    valid_pred_auc = 0
                    
                if test_pred_auc.shape[1] > 1:
                    test_pred_auc = test_pred_auc[:,1]
                else:
                    test_pred_auc = 0

            else:
                train_pred_auc = model.decision_function(train_in.T)
                valid_pred_auc = model.decision_function(valid_in.T)
                test_pred_auc = model.decision_function(test_in.T)
            
            e_train = np.nan if only_zeros else metrics.roc_auc_score(train_out, train_pred_auc)
            e_valid = np.nan if (np.nansum(valid_out) == 0) | (np.nansum(valid_out) == valid_out.size) else metrics.roc_auc_score(valid_out, valid_pred_auc)
            e_test = np.nan if (np.nansum(test_out) == 0) | (np.nansum(test_out) == test_out.size) else metrics.roc_auc_score(test_out, test_pred_auc)

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
        
    # Store the evaluation metrics    
    results['train_eval'] = eval_train
    results['valid_eval'] = eval_valid
    results['test_eval'] = eval_test
        
    # IJ is None, a grid point by grid point evaluation is being done, and only the evaluation metrics are desired
    if IJ == None:
        return results
    
    # Model predictions
    results['train_predict'] = train_pred.reshape(T, IJ, order = 'F')
    results['valid_predict'] = valid_pred.reshape(Tt, IJ, order = 'F')
    results['test_predict'] = test_pred.reshape(Tt, IJ, order = 'F')
    
        
    # Collect information for the ROC curve
    if args.roc_curve:
        # Test if the model predicts FD
        if (args.ml_model.lower() == 'rf') | (args.ml_model.lower() == 'random_forest'):
            train_pred_roc = model.predict_proba(train_in.T)
            valid_pred_roc = model.predict_proba(valid_in.T)
            test_pred_roc = model.predict_proba(test_in.T)

            if train_pred_roc.shape[1] > 1:
                train_pred_roc = train_pred_roc[:,1]
            else:
                train_pred_roc = 0

            if valid_pred_roc.shape[1] > 1:
                valid_pred_roc = valid_pred_roc[:,1]
            else:
                valid_pred_roc = 0

            if test_pred_roc.shape[1] > 1:
                test_pred_roc = test_pred_roc[:,1]
            else:
                test_pred_roc = 0
        
        else:
            train_pred_roc = model.decision_function(train_in.T)
            valid_pred_roc = model.decision_function(valid_in.T)
            test_pred_roc = model.decision_function(test_in.T)
        
        thresh = np.arange(0, 2, 1e-4)
        thresh = np.round(thresh, 4)
        results['fpr_train'] = np.ones((thresh.size)) * np.nan
        results['tpr_train'] = np.ones((thresh.size)) * np.nan
        results['fpr_valid'] = np.ones((thresh.size)) * np.nan
        results['tpr_valid'] = np.ones((thresh.size)) * np.nan
        results['fpr_test'] = np.ones((thresh.size)) * np.nan
        results['tpr_test'] = np.ones((thresh.size)) * np.nan
        
        fpr_train, tpr_train, thresh_train = metrics.roc_curve(train_out, train_pred_roc)
        
        fpr_valid, tpr_valid, thresh_valid = metrics.roc_curve(valid_out, valid_pred_roc)
        
        fpr_test, tpr_test, thresh_test = metrics.roc_curve(test_out, test_pred_roc)
        
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

    # Collect the feature importance
    if args.feature_importance:
        results['feature_importance'] = np.array(model.feature_importances_)
        
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
    
    
    
    
def execute_single_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, data_in, data_out, rotation, model_fname, evaluate_each_grid = False):
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
    :param data_in: Entire input dataset. Results predict for the entire dataset instead of the training set (to simplify the merging step)
    :param data_out: Enitre output dataset. Results predict for the entire dataset instead of the training set (to simplify the merging step)
    :param rotation: Current rotation in the k-fold validation
    :param model_fname: The filename to save the model to
    :param evaluate_each_grid: Boolean indicating whether evaulation metrics should be examined for each grid point in the dataset
    
    Outputs:
    :param results: Dictionary results from the ML model, including predictions, performance metrics, and learning curves
    '''
    
    # Execute the experiment based on whether it is an sklearn model or NN
    if args.keras:
        results = execute_keras_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, data_in, data_out, rotation,
                                    model_fname)
        
    else:
        results = execute_sklearn_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, data_in, data_out, rotation,
                                      model_fname, evaluate_each_grid)
    
    return results


#%%
##############################################

# Function to conduct an experiment for 1 rotation and 1 ML model and 1 FD method (for all regions)
def execute_exp(args, test = False):
    '''
    Run the ML experiment
    
    Inputs:
    :param args: Argparse arguments
    :param test: Boolean indicating if this is a test run; multiple rotations are runned and merged for testing and tuning hyperparameters
    
    Outputs:
    :param results: The results of the experiment
    '''
    
    # List of FD identification methods
    methods = np.asarray(['christian', 'nogeura', 'pendergrass', 'liu', 'otkin'])
    
    method_ind = np.where(methods == args.method)[0]
    
    # Save the original class weight
    weight = args.class_weight
    
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
    Nvar, T, IJ, Nfold  = data_in.shape
    Nmethods = data_out.shape[0]
    
    # Create a version of the entire dataset without being split
    data_in_whole = np.concatenate([data_in[:,:,:,fold] for fold in range(Nfold)], axis = 1)
    data_out_whole = np.concatenate([data_out[:,:,:,fold] for fold in range(Nfold)], axis = 1)
    
    # Load example data with subsetted lat/lon data
    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
    lat = et['lat']; lon = et['lon']
    
    # Collect the spatial size of the data
    I, J = lat.shape
    
    # Correct the longitude?
    if args.correct_lon:
        print('Correcting longitude...')
        for n in range(len(lon[:,0])):
            ind = np.where(lon[n,:] > 0)[0]
            lon[n,ind] = -1*lon[n,ind]
            
    

    # If this is a test run, train for multiple rotations
    if test:
        for rot in args.rotation:

            # Split the data into training, validation, and test sets
            train_in, valid_in, test_in = split_data(data_in, args.ntrain_folds, rot, normalize = args.normalize)
            train_out, valid_out, test_out = split_data(data_out, args.ntrain_folds, rot, normalize = False) # Note the label data is already binary

            # Generate the model filename
            model_fbase = generate_model_fname(args.ra_model, args.label, args.method, rot)
            model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, args.method, model_fbase)
            # print(model_fname)

            # If the training data has too few FD, bagging can easly find some subsets that have no FD, resulting in a class weight error.
            # Ignore the class weights for this scenario
            if (100*np.where(train_out == 1)[0].size/train_out.size) < 0.05:
                args.class_weight = None
            else:
                args.class_weight = weight

            # Perform the experiment
            # The ML model is saved in this step
            execute_single_exp(args, train_in, valid_in, test_in, 
                               train_out[method_ind,:,:], valid_out[method_ind,:,:], test_out[method_ind,:,:], 
                               data_in_whole, data_out_whole[method_ind,:,:], rot,
                               model_fname, evaluate_each_grid = True)
    
    # Otherwise, train for 1 rotation at a time
    else:
        # Split the data into training, validation, and test sets
        train_in, valid_in, test_in = split_data(data_in, args.ntrain_folds, args.rotation, normalize = args.normalize)
        train_out, valid_out, test_out = split_data(data_out, args.ntrain_folds, args.rotation, normalize = False) # Note the label data is already binary

        # Generate the model filename
        model_fbase = generate_model_fname(args.ra_model, args.label, args.method, args.rotation[0])
        model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, args.method, model_fbase)
        # print(model_fname)
        
        # Leave if the experiment has already been completed
        if os.path.exists(model_fname):
            print('File already exists/experiment has already been performed')
            return

        # If the training data has too few FD, bagging can easly find some subsets that have no FD, resulting in a class weight error.
        # Ignore the class weights for this scenario
        if (100*np.where(train_out == 1)[0].size/train_out.size) < 0.05:
            args.class_weight = None
        else:
            args.class_weight = weight

        # Perform the experiment
        # The ML model is saved in this step
        results = execute_single_exp(args, train_in, valid_in, test_in, 
                                     train_out[method_ind,:,:], valid_out[method_ind,:,:], test_out[method_ind,:,:], 
                                     data_in_whole, data_out_whole[method_ind,:,:], args.rotation[0],
                                     model_fname, evaluate_each_grid = True)

    # If this is a test run, merge the results over test rotations (otherwise, this is done separately in the main function, after all rotations are trained)
    if test:
        print('Merging the results of %s for the %s method...'%(args.ml_model, args.method))
        results = merge_results(args, args.method, lat, lon, Nfold, Nvar, T, I, J)
        
    print('Done.')
    return 


def merge_results(args, method, lat, lon, NFolds, NVar, T, I, J):
    '''
    Merge the results and predictions of a ML model across all rotations
    
    Inputs:
    :param args: Argparse arguments
    :param method: The FD identification method the ML model was trained to recognize
    :param lat: Gridded latitude values corresponding to the full dataset
    :param lon: Gridded longitude values corresponding to the full dataset
    :param NFolds: Total number of folds in the full dataset
    :param NVar: Number of variables used to train the ML model
    :param T, I, J: Size of the time (for 1 fold), horizontal, and width dimensions respectively
    '''
    
    # Construct the base file name for each result
    model_fbase = 'results_%s_%s_%s_rot_'%(args.ra_model, 
                                           args.label, 
                                           method)
    
    dataset_dir = '%s/%s/%s/%s'%(args.dataset, args.ra_model, args.ml_model, method)
    dataset_dir_hub = '%s/%s'%(args.dataset, args.ra_model)
    model_fname = '%s/%s'%(dataset_dir, model_fbase)
    
    # Collect the files for all rotations
    files = ['%s/%s'%(dataset_dir,f) for f in os.listdir(dataset_dir) if re.match(r'%s.+.pkl'%(model_fbase), f)]
    files.sort()
    
    print(files)
    
    Nrot = len(files)
    
    # Initialize results
    pred_train = np.ones((NFolds*T, I*J)) * np.nan
    pred_valid = np.ones((NFolds*T, I*J)) * np.nan
    pred_test = np.ones((NFolds*T, I*J)) * np.nan
    
    eval_train = np.ones((len(args.metrics))) * np.nan
    eval_valid = np.ones((len(args.metrics))) * np.nan
    eval_test = np.ones((len(args.metrics))) * np.nan
    
    eval_train_var = np.ones((len(args.metrics))) * np.nan
    eval_valid_var = np.ones((len(args.metrics))) * np.nan
    eval_test_var = np.ones((len(args.metrics))) * np.nan
    
    eval_train_map = np.ones((I,J,len(args.metrics))) * np.nan
    eval_valid_map = np.ones((I,J,len(args.metrics))) * np.nan
    eval_test_map = np.ones((I,J,len(args.metrics))) * np.nan
    
    eval_train_var_map = np.ones((I,J,len(args.metrics))) * np.nan
    eval_valid_var_map = np.ones((I,J,len(args.metrics))) * np.nan
    eval_test_var_map = np.ones((I,J,len(args.metrics))) * np.nan
    
    if args.feature_importance:
        feature_import = np.ones((NVar)) * np.nan
        feature_import_var = np.ones((NVar)) * np.nan
    
    if args.keras:
        learn_curves = np.ones((T, I*J)) * np.nan
        learn_curves_var = np.ones((T, I*J)) * np.nan
    
    # Initialize some lists
    ptrain = []
    
    etrain = []
    evalid = []
    etest = []
    
    etrain_map = []
    evalid_map = []
    etest_map = []

    lc = []
    fi = []

    fpr_train_tmp = []
    fpr_valid_tmp = []
    fpr_test_tmp = []

    tpr_train_tmp = []
    tpr_valid_tmp = []
    tpr_test_tmp = []
    
    
    # Collect the results for each rotation
    for rot, f in enumerate(files):
        with open(f, 'rb') as fn:
            result = pickle.load(fn)
        
        val_folds = int((np.array([args.ntrain_folds]) + rot) % NFolds)
        test_folds = int((np.array([args.ntrain_folds]) + 1 + rot) % NFolds)

        
        # "train" set is for the entire dataset; this gets averaged together in the merged results
        # Valid and test predictions get "stacked" together in temporal order (each rotation should only predict 1 fold for validation and test each)
        ptrain.append(result['train_predict'])
        pred_valid[val_folds*T:(val_folds+1)*T,:] = result['valid_predict']
        pred_test[test_folds*T:(test_folds+1)*T,:] = result['test_predict']
        
        etrain.append(result['train_eval'])
        evalid.append(result['valid_eval'])
        etest.append(result['test_eval'])
        
        etrain_map.append(result['eval_train_map'])
        evalid_map.append(result['eval_valid_map'])
        etest_map.append(result['eval_test_map'])

        if args.roc_curve:
            fpr_train_tmp.append(result['fpr_train'])
            fpr_valid_tmp.append(result['fpr_valid'])
            fpr_test_tmp.append(result['fpr_test'])

            tpr_train_tmp.append(result['tpr_train'])
            tpr_valid_tmp.append(result['tpr_valid'])
            tpr_test_tmp.append(result['tpr_test'])

        if args.feature_importance:
            fi.append(result['feature_importance'])

        if args.keras:
            lc.append(result['history'])
            
    # Merge the results
    pred_train = np.round(np.nanmean(np.stack(ptrain, axis = -1), axis = -1), 0) # The round restores the average back to binary 1 or 0; 
    pred_valid = np.round(pred_valid, 0)                                         # average < 0.5 means majority of rotations did not identify FD
    pred_test = np.round(pred_test, 0)

    eval_train = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
    eval_valid = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
    eval_test = np.nanmean(np.stack(etest, axis = -1), axis = -1)

    eval_train_var = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
    eval_valid_var = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
    eval_test_var = np.nanstd(np.stack(etest, axis = -1), axis = -1)
    
    eval_train_map = np.nanmean(np.stack(etrain_map, axis = -1), axis = -1)
    eval_valid_map = np.nanmean(np.stack(evalid_map, axis = -1), axis = -1)
    eval_test_map = np.nanmean(np.stack(etest_map, axis = -1), axis = -1)

    eval_train_var_map = np.nanstd(np.stack(etrain_map, axis = -1), axis = -1)
    eval_valid_var_map = np.nanstd(np.stack(evalid_map, axis = -1), axis = -1)
    eval_test_var_map = np.nanstd(np.stack(etest_map, axis = -1), axis = -1)    

    if args.roc_curve:
        fpr_train = np.nanmean(np.stack(fpr_train_tmp, axis = -1), axis = -1)
        fpr_valid = np.nanmean(np.stack(fpr_valid_tmp, axis = -1), axis = -1)
        fpr_test = np.nanmean(np.stack(fpr_test_tmp, axis = -1), axis = -1)

        fpr_train_var = np.nanstd(np.stack(fpr_train_tmp, axis = -1), axis = -1)
        fpr_valid_var = np.nanstd(np.stack(fpr_valid_tmp, axis = -1), axis = -1)
        fpr_test_var = np.nanstd(np.stack(fpr_test_tmp, axis = -1), axis = -1)

        tpr_train = np.nanmean(np.stack(tpr_train_tmp, axis = -1), axis = -1)
        tpr_valid = np.nanmean(np.stack(tpr_valid_tmp, axis = -1), axis = -1)
        tpr_test = np.nanmean(np.stack(tpr_test_tmp, axis = -1), axis = -1)

        tpr_train_var = np.nanstd(np.stack(tpr_train_tmp, axis = -1), axis = -1)
        tpr_valid_var = np.nanstd(np.stack(tpr_valid_tmp, axis = -1), axis = -1)
        tpr_test_var = np.nanstd(np.stack(tpr_test_tmp, axis = -1), axis = -1)


    if args.feature_importance:
        feature_import = np.nanmean(np.stack(fi, axis = -1), axis = -1)
        feature_import_var = np.nanstd(np.stack(fi, axis = -1), axis = -1)

    if args.keras:
        learn_curves[:,ind] = np.nanmean(np.stack(lc, axis = -1), axis = -1)
        learn_curves_var[:,ind] = np.nanstd(np.stack(lc, axis = -1), axis = -1)
        
        
    # Create a small plot of model performance across rotations for each metric
    rotations = np.arange(Nrot)
    for m, metric in enumerate(args.metrics):
        met_train = [e[m] for e in etrain]
        met_val = [e[m] for e in evalid]
        met_test = [e[m] for e in etest]
        
        fig, ax = plt.subplots(figsize = [12, 8])
        
        # Set the title
        ax.set_title('%s %s for each rotation for the %s'%(args.ml_model, metric, args.ra_model), fontsize = 16)

        # Make the plots
        ax.plot(rotations, met_train, color = 'r', linestyle = '-', linewidth = 1.5, marker = 'o', label = 'Training set')
        ax.plot(rotations, met_val, color = 'darkgreen', linestyle = '-', linewidth = 1.5, marker = 'o', label = 'Validation set')
        ax.plot(rotations, met_test, color = 'b', linestyle = '-', linewidth = 1.5, marker = 'o', label = 'Test set')

        # Make a legend
        ax.legend(loc = 'upper right', fontsize = 16)

        # Set the labels
        ax.set_ylabel(metric, fontsize = 16)
        ax.set_xlabel('Rotation', fontsize = 16)


        # Set the tick sizes
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(16)

        # Save the figure
        filename = '%s_%s_metric_performance_across_rotations.png'%(args.label, metric)
        plt.savefig('%s/%s'%(dataset_dir_hub, filename), bbox_inches = 'tight')
        plt.show(block = False)
        
            
    # Generate the name of the overall results file        
    results_fbase = generate_results_fname(args.ra_model, args.label, args.method, args.keras)
    results_fname = '%s/%s'%(dataset_dir_hub, results_fbase)
    print(results_fname)
    
    results = {}
    # Model coordinates
    results['lat'] = lat; results['lon'] = lon
    
    # Model predictions
    results['all_predict'] = pred_train.reshape(NFolds*T, I, J, order = 'F')
    results['valid_predict'] = pred_valid.reshape(NFolds*T, I, J, order = 'F')
    results['test_predict'] = pred_test.reshape(NFolds*T, I, J, order = 'F')
    
    # Overall model performance
    results['all_eval'] = eval_train
    results['valid_eval'] = eval_valid
    results['test_eval'] = eval_test
    
    results['all_eval_var'] = eval_train_var
    results['valid_eval_var'] = eval_valid_var
    results['test_eval_var'] = eval_test_var
    
    # Model performance over each individual grid point
    results['all_eval_map'] = eval_train_map.reshape(I, J, len(args.metrics), order = 'F')
    results['valid_eval_map'] = eval_valid_map.reshape(I, J, len(args.metrics), order = 'F')
    results['test_eval_map'] = eval_test_map.reshape(I, J, len(args.metrics), order = 'F')
    
    results['all_eval_var_map'] = eval_train_var_map.reshape(I, J, len(args.metrics), order = 'F')
    results['valid_eval_var_map'] = eval_valid_var_map.reshape(I, J, len(args.metrics), order = 'F')
    results['test_eval_var_map'] = eval_test_var_map.reshape(I, J, len(args.metrics), order = 'F')
    
    # ROC curve
    if args.roc_curve:
        results['fpr_all'] = fpr_train
        results['fpr_valid'] = fpr_valid
        results['fpr_test'] = fpr_test
        
        results['fpr_all_var'] = fpr_train_var
        results['fpr_valid_var'] = fpr_valid_var
        results['fpr_test_var'] = fpr_test_var
        
        results['tpr_all'] = tpr_train
        results['tpr_valid'] = tpr_valid
        results['tpr_test'] = tpr_test
        
        results['tpr_all_var'] = tpr_train_var
        results['tpr_valid_var'] = tpr_valid_var
        results['tpr_test_var'] = tpr_test_var
    
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
    
    return results



#%
##############################################
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_ml_parser()
    args = parser.parse_args()
    
    # Execute the experiments?
    if np.invert(args.nogo):
        print('Performing experiment...')
        execute_exp(args)
    
    # Perform model evaluations instead? (This is done after all rotations for all methods are run)
    if args.evaluate:
        print('Initializing some variables...')
        methods = ['christian', 'nogeura', 'pendergrass', 'liu', 'otkin']
        
        # Get the directory of the dataset
        dataset_dir = '%s/%s'%(args.dataset, args.ra_model)

        # Load the data
        # Data is Nfeatures/Nmethods x time x space x fold
        data_in = load_ml_data(args.input_data_fname, path = dataset_dir)

        # Make the rotations
        Nvar, T, IJ, Nfolds = data_in.shape
        
        # Load example data with subsetted lat/lon data
        et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
        lat = et['lat']; lon = et['lon']

        # Collect the spatial size of the data
        I, J = lat.shape
        
        print('Merging results results...')
        results = []

        for method in methods:
            result_method = merge_results(args, lat, lon, Nfolds, Nvar, T, I, J)
            
            results.append(result_method)

        # Note here that results[0] = christian; results[1] = nogeura; results[2] = pendergrass; rsults[3] = liue; results[4] = otkin

        # Obtain the latitude and longitude for metrics
        lat = results[0]['lat']; lon = results[0]['lon']
        
        # remove a large variable to clear space
        del result_method
        gc.collect() # Clears deleted variables from memory 


        # Plot the results of the metrics
        print('Plotting results...')
        for met, metric in enumerate(args.metrics):
            # Collect the metrics
            metrics_all = [results[m]['all_eval_map'][:,:,met] for m in range(len(methods))]
            metrics_valid = [results[m]['valid_eval_map'][:,:,met] for m in range(len(methods))]
            metrics_test = [results[m]['test_eval_map'][:,:,met] for m in range(len(methods))]

            if (metric == 'mse') | (metric == 'mae'):
                cmin = 0; cmax = 0.5; cint = 0.05
            else:
                cmin = 0; cmax = 1; cint = 0.1

            # Plot the metric
            display_metric_map(metrics_all, lat, lon, methods, 
                               metric, cmin, cmax, cint, args.ra_model, 
                               args.label, dataset = 'all', reverse = False, globe = args.globe, path = dataset_dir)

            display_metric_map(metrics_valid, lat, lon, methods, 
                               metric, cmin, cmax, cint, args.ra_model, 
                               args.label, dataset = 'valid', reverse = False, globe = args.globe, path = dataset_dir)

            display_metric_map(metrics_test, lat, lon, methods, 
                               metric, cmin, cmax, cint, args.ra_model, 
                               args.label, dataset = 'test', reverse = False, globe = args.globe, path = dataset_dir)

            # Remove variables at the end to clear space
            del metric_all, metric_valid, metric_test
            gc.collect() # Clears deleted variables from memory 


        # Plot the ROC curves?
        if args.roc_curve:
            # Collect the ROC curve informations
            print('Making ROC curves...')
            tpr_all = [results[m]['tpr_all'] for m in range(len(methods))]
            tpr_valid = [results[m]['tpr_valid'] for m in range(len(methods))]
            tpr_test = [results[m]['tpr_test'] for m in range(len(methods))]

            tpr_var_all = [results[m]['tpr_all_var'] for m in range(len(methods))]
            tpr_var_valid = [results[m]['tpr_valid_var'] for m in range(len(methods))]
            tpr_var_test = [results[m]['tpr_test_var'] for m in range(len(methods))]

            fpr_all = [results[m]['fpr_all'] for m in range(len(methods))]
            fpr_valid = [results[m]['fpr_valid'] for m in range(len(methods))]
            fpr_test = [results[m]['fpr_test'] for m in range(len(methods))]

            fpr_var_all = [results[m]['fpr_all_var'] for m in range(len(methods))]
            fpr_var_valid = [results[m]['fpr_valid_var'] for m in range(len(methods))]
            fpr_var_test = [results[m]['fpr_test_var'] for m in range(len(methods))]

            # Plot the ROC curves
            display_roc_curves(tpr_all, fpr_all, tpr_var_all, fpr_var_all, 
                               methods, args.ra_model, args.label, dataset = 'all', path = dataset_dir)

            display_roc_curves(tpr_valid, fpr_valid, tpr_var_valid, fpr_var_valid, 
                               methods, args.ra_model, args.label, dataset = 'valid', path = dataset_dir)

            display_roc_curves(tpr_test, fpr_test, tpr_var_test, fpr_var_test, 
                               methods, args.ra_model, args.label, dataset = 'test', path = dataset_dir)

            # Remove variables at the end to clear space
            del tpr_all, tpr_valid, tpr_test, tpr_all_var, tpr_valid_var, tpr_test_var
            del fpr_all, fpr_valid, fpr_test, fpr_all_var, fpr_valid_var, fpr_test_var
            gc.collect() # Clears deleted variables from memory 



        # Plot the feature importance?
        if args.feature_importance:
            features = [r'T', r'ET', r'$\Delta$ET', r'PET', r'$\Delta$PET', r'P', r'SM', r'$\Delta$SM']
            NFeature = results[0]['feature_importance'].shape[0]

            fi = [results[m]['feature_importance'] for m in range(len(methods))]
            fi_var = [results[m]['feature_importance_var'] for m in range(len(methods))]

            # Create a barplot of the overall feature variation
            print('Making overall feature importance barplot...')
            display_feature_importance(fi, fi_var, features, methods, args.ra_model, args.label, path = dataset_dir)

            # Remove variables at the end to clear space
            del fi, fi_var
            gc.collect() # Clears deleted variables from memory 


        # Plot the learning curve?


        # Make predictions?
        if (args.climatology_plot | args.time_series | args.case_studies | args.confusion_matrix_plots):

            # Make the predictions and plot the results for each method
            for m, method in enumerate(methods):
                
                pred_all = results[m]['all_predict']
                pred_valid = results[m]['valid_predict']
                pred_test = results[m]['test_predict']


                # Load in the true labels
                print('Loading true labels for the %s method...'%method)
                true_fd = load_nc('fd', '%s.%s.pentad.nc'%(method, args.ra_model), path = dataset_dir)

                ind = np.where( (true_fd['month'] >= 4) & (true_fd['month'] <= 10) )[0]
                fd = true_fd['fd'][ind,:,:]

                dates = true_fd['ymd'][ind]


                # Plot the threat scores?
                if args.confusion_matrix_plots:
                    plot('Plotting confusion matrix skill scores for the %s method...'%method)
                    mask = load_mask(model = args.ra_model)

                    # Plot the threat scores with the full predictions
                    display_threat_score(fd, pred, true_fd['lat'], true_fd['lon'], dates, mask, 
                                         model = args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far_score(fd, pred, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod_score(fd, pred, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)


                    
                    # Plot the threat scores with only the validation predictions (to see how the model generalizes to data it has not seen)
                    display_threat_score(fd, pred_valid, true_fd['lat'], true_fd['lon'], dates, mask, 
                                         model = args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far_score(fd, pred_valid, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod_score(fd, pred_valid, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)
                    
                    
                    
                    # Plot the threat scores with only the test predictions (to see how the model generalizes to data it has not seen)
                    display_threat_score(fd, pred_test, true_fd['lat'], true_fd['lon'], dates, mask, 
                                         model = args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far_score(fd, pred_test, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod_score(fd, pred_test, true_fd['lat'], true_fd['lon'], dates, 
                                      model =  args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)


                # Plot the predicted climatology map?
                if args.climatology_plot:
                    print('Plotting the predicted climatology for the the %s method...'%method)
                    display_fd_climatology(pred, true_fd['lat'], true_fd['lon'], dates, 'Predicted FD for %s'%method, 
                                           model = '%s_%s'%(method, args.ra_model), path = dataset_dir, grow_season = True)

                    
                    # Plot the climatology map with only the validation predictions (to see how the model generalizes to data it has not seen)
                    display_fd_climatology(pred_valid, true_fd['lat'], true_fd['lon'], dates, 'Predicted FD for %s'%method, 
                                           model = '%s_%s_valid_set'%(method, args.ra_model), path = dataset_dir, grow_season = True)
                    
                    
                    # Plot the climatology map with only the test predictions (to see how the model generalizes to data it has not seen)
                    display_fd_climatology(pred_test, true_fd['lat'], true_fd['lon'], dates, 'Predicted FD for %s'%method, 
                                           model = '%s_%s_test_set'%(method, args.ra_model), path = dataset_dir, grow_season = True)


                # Plot the predicted time series (with true labels)?
                if args.time_series:
                    print('Calculating areal coverage for the %s method...'%method)
                    # Examine predicted time series
                    T, I, J = pred.shape

                    # Determine the areal coverage for the time series
                    tmp_pred = np.nansum(pred.reshape(T, I*J), axis = -1)*32*32
                    tmp_pred_valid = np.nansum(pred_valid.reshape(T, I*J), axis = -1)*32*32
                    tmp_pred_test = np.nansum(pred_test.reshape(T, I*J), axis = -1)*32*32

                    pred_area = []
                    pred_area_var = []

                    pred_area_valid = []
                    pred_area_var_valid = []
                    
                    pred_area_test = []
                    pred_area_var_test = []


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

                        pred_area_valid.append(np.nanmean(tmp_pred_valid[ind]))
                        pred_area_var_valid.append(np.nanstd(tmp_pred_valid[ind]))
                        
                        pred_area_test.append(np.nanmean(tmp_pred_test[ind]))
                        pred_area_var_test.append(np.nanstd(tmp_pred_test[ind]))

                        true_area.append(np.nanmean(tmp_true[ind]))
                        true_area_var.append(np.nanstd(tmp_true[ind]))

                    pred_area = np.array(pred_area)
                    pred_area_var = np.array(pred_area_var)

                    pred_area_valid = np.array(pred_area_valid)
                    pred_area_var_valid = np.array(pred_area_var_valid)
                    
                    pred_area_test = np.array(pred_area_test)
                    pred_area_var_test = np.array(pred_area_var_test)

                    true_area = np.array(true_area)
                    true_area_var = np.array(true_area_var)


                    # Display the time series
                    print('Plotting the areal coverage of FD time series for the %s method...'%method)
                    display_time_series(true_area, pred_area, true_area_var, pred_area_var, dates[::43], 
                                        r'Areal Coverage (km^2)', args.ra_model, '%s_%s'%(args.label, method), path = dataset_dir)

                    # Display the time series with only the validation predictions (to see how the model generalizes to data it has not seen)
                    display_time_series(true_area, pred_area_valid, true_area_var, pred_area_var_valid, dates[::43], 
                                        r'Areal Coverage (km^2)', args.ra_model, '%s_%s_valid_set'%(args.label, method), path = dataset_dir)
                    
                    # Display the time series with only the test predictions (to see how the model generalizes to data it has not seen)
                    display_time_series(true_area, pred_area_test, true_area_var, pred_area_var_test, dates[::43], 
                                        r'Areal Coverage (km^2)', args.ra_model, '%s_%s_test_set'%(args.label, method), path = dataset_dir)

                # Plot a set of case studies?
                if args.case_studies:
                    print('Plotting case studies for the %s method...'%method)

                    # Plot the case studies for the predicted labels
                    display_case_study_maps(pred, true_fd['lon'], true_fd['lat'], dates, args.case_study_years, 
                                            method = method, label = args.label, dataset = args.ra_model, 
                                            globe = False, path = dataset_dir, grow_season = True)

                    # Plot the case studies for the true labels
                    display_case_study_maps(fd, true_fd['lon'], true_fd['lat'], dates, args.case_study_years, 
                                            method = method, label = args.label, dataset = args.ra_model, 
                                            globe = False, path = dataset_dir, grow_season = True)

                    
                    # Repeat the predicted case studes with predicted labels using validation sets (to see how the model generalizes to data it has not seen)
                    display_case_study_maps(pred_valid, true_fd['lon'], true_fd['lat'], dates, args.case_study_years, 
                                            method = method, label = '%s_test_set'%args.label, dataset = args.ra_model, 
                                            globe = False, path = dataset_dir, grow_season = True)
                    
                    # Repeat the predicted case studes with predicted labels using test sets only (to see how the model generalizes to data it has not seen)
                    display_case_study_maps(pred_test, true_fd['lon'], true_fd['lat'], dates, args.case_study_years, 
                                            method = method, label = '%s_test_set'%args.label, dataset = args.ra_model, 
                                            globe = False, path = dataset_dir, grow_season = True)




    print('Done')        
                    

