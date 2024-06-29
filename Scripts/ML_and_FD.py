#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 2 17:52:45 2021

##############################################################
# File: ML_and_FD.py
# Version: 3.5.0
# Author: Stuart Edris (sgedris@ou.edu)
# Description:
#     This is the main script for the employment of machine learning to identify flash drought study.
#     This script takes in indces calculated from the Raw_Data_Processing script (training data) and the 
#     identified flash drought in the Calculate_FD script (label data) and identifies flash drought
#     using those variables that indicates flash drought (T, P, ET, PET, and SM). Several models are employed
#     (ada boosted trees, random forests, SVMs, and ANNs/DNNs, CNNs, RNNs, and transformer networks).
#     Models are run for each flash drought identification method. Output results consist of performance statistics, ROC curves,
#     predicted climatologies, predicted case studies, feature contributions, etc.
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
#   3.1.0 - 12/28/2022 - Added Keras models and evaluation to the the code, and ANN, CNN, and RNNs to Construct_ML_Models.py. Other improvements and fixes.
#   3.2.0 - 1/14/2023 - Updated TensorFlow to 2.8 and added data augmentation layers to the CNNs
#   3.2.1 - 5/15/2023 - Changed TensorFlow models to use Datasets instead of numpy arrays. Other similar changes to reduce RAM uses of TensorFlow models. 
#                       Other bug fixes.
#   3.3.0 - 6/14/2023 - Added capability of utilizing GPUs with TF models used a mirrored strategy approach.
#   3.4.0 - 6/21/2023 - Added self attention networks, based on the transformer configuration.
#   3.5.0 - 7/10/2023 - Added CNN-RNN network code.
#   3.6.0 - 1/20/2024 - Added compatibility with global datasets.
#   3.7.0 - 6/29/2024 - Fixed U-Nets to run properly.
#
# Inputs:
#   - Data files for FD indicator features, and FD labels (.pkl format)
#   - Input arguments indicating where to find the data files, ML model parameters, and others (see create_ml_parser() function).
#
# Outputs:
#   - File for ML model created (.pkl for sklearn model, folders for TensorFlow model)
#   - File for model results for rotation created 
#     (contains overall statistical performance, maps of statistics, ROC curves, and learning curves (TF models); .pkl file)
#   - File containing the averaged results over all rotations (includes XAI variables; .pkl file)
#   - Figures displaying model results (coming from the file that contains the averaged results overall rotations)
#
# To Do:
#   - May look into residual curves (training and validation metric performance over different number of folds; 
#                                    may be too computationally and temporally expensive)
#   - May look into Ceteris-Paribus effect
#   - Add XAI for keras models
#   - See multiple hashtag comments in merge_results() function for list of tasks (4 total comments)
#   - See multiple hashtag comments in if __name__ == ... for list of tasks (3 total comments)
#
# Bugs:
#   - TensorFlow models are known to perform more poorly if the computer has not been power cycled (turned off and on) for a while. 
#     True even if the kernel is reset
#   - Current errors for the transformer models have been removed, however the local computer nearly crashed when a test run was done (during the training step;
#     this is presumed to be due to the transformers computation requirements)
#   - Notes on things to fix on lines 2251 and 2351
#       - FIXES IMPLEMENTED, NEED TO BE TESTED (True also for the new feature importances)
#
# Notes:
#   - See tf_environment.yml for a list of all packages and versions. netcdf4 and cartopy must be downloaded seperately.
#   - This script assumes it is being running in the 'ML_and_FD' directory
#   - Several of the comments for this program are based on the Spyder IDL, which can separate the code into different blocks of code, or 'cells' using '#%%'
#
###############################################################

"""

#####
            
    
#%%
##############################################

# Library imports
import os, sys, warnings
import gc
import re
import argparse
import pickle
import shap
import skexplain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colorbar as mcolorbar
import tensorflow as tf
from tensorflow import keras
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy.special import softmax
from scipy.special import gamma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics

# Tensorflow 2.x way of doing things
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Reshape, Masking, Flatten, RepeatVector
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, SpatialDropout2D, Concatenate
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Sequential, Model

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

    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ML Calculations', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--label', type=str, default='rf', help='Experiment label')
    
    parser.add_argument('--dataset', type=str, default='./Data', help='Data set directory')
    
    parser.add_argument('--input_data_fname', type=str, default='fd_input_features.pkl', help='Filename of the input data')
    parser.add_argument('--output_data_fname', type=str, default='fd_output_labels.pkl', help='Filename of the target output data')
        
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use GPU processors (only for TensorFlow models)')

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
    parser.add_argument('--case_study_years', type=int, nargs='+', default=[1988, 2000, 2003, 2011, 2012, 2017, 2019], help='List of years to make case studies for')
    parser.add_argument('--globe', action='store_true', help='Plot global dataset (otherwise plot CONUS)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the ML model after having run it for all rotations and FD identification methods')
    parser.add_argument('--interpret', action='store_true', help='Obtain the SHAP values and add plots for explainability of the ML models')
    
    
    # General Neural Network parameters
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--loss', type=str, default='binary_crossentropy', help = 'Loss being minimized by model')
    parser.add_argument('--focal_parameters', type=float, nargs=2, default=[2, 4], help = 'Parameters for the focal loss function (gamma is first value, alpha is the second)')
    parser.add_argument('--batch', type=int, default=43, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=2, help="How many batches to prefetch during training (keras models only)")
    parser.add_argument('--activation', type=str, nargs='+', default=['sigmoid', 'tanh'], help='Activation function(s) for each layer in the NNs')
    parser.add_argument('--output_activation', type=str, default = 'sigmoid', help='Activation function for the NN output layer')
    parser.add_argument('--multiprocess', action='store_true', help='Use multiple processing in training the deep model')
    
    
    # Nueral Network Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help='L1 regularization factor')
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help='L2 regularization factor (only active if no L2)')
    
    
    # Data augmentation parameter (currently only applied to CNNs
    parser.add_argument('--data_augmentation', action='store_true', help = 'Use data augmentation')
    parser.add_argument('--crop_height', type=int, default=None, help = 'Crop the map along the height axis. This must have the same primes used in the max pooling layers')
    parser.add_argument('--crop_width', type=int, default=None, help = 'Crop the map along the width axis. This must have the same primes used in the max pooling layers')
    parser.add_argument('--flip', type=str, default='None', help = 'Flip the map. Entries are "horizontal", "vertical", "horizontal_and_vertical", or "vertical_and_horizontal"')
    parser.add_argument('--data_aug_rotation', type=float, default=None, help = 'Rotate the map. Inputs are percentage of rotation (e.g., 0.2 is 20% of a 360 degree rotation)')
    parser.add_argument('--translation_height', type=float, default=None, help = 'Slide the map along the height axis. Inputs are percentage of slide (e.g., 0.2 is up to +/- 20% of the full image sliding')
    parser.add_argument('--translation_width', type=float, default=None, help = 'Slide the map along the width axis. Inputs are percentage of slide (e.g., 0.2 is up to +/- 20% of the full image sliding')
    parser.add_argument('--zoom_height', type=float, default=None, help = 'Zoom the map along the height axis. Inputs are percentage of zoom (e.g., 0.2 is up to a 20% zoom)')
    parser.add_argument('--zoom_width', type=float, default=None, help = 'Zoom the map along the width axis. Inputs are percentage of zoom (e.g., 0.2 is up to a 20% zoom)')

    
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
    
    # Artificial Neural Network
    parser.add_argument('--units', nargs='+', type=int, default=[200, 100, 50, 25, 10, 5], help='Number of hidden units per layer (sequence of ints)')
    
    # Convolutional U-Net 
    parser.add_argument('--sequential', action='store_true', help = 'Build a sequential U-net (has no skip connections)')
    parser.add_argument('--variational', action='store_true', help = 'Build a variational autoencoder/U-net')
    parser.add_argument('--nfilters', nargs='+', type=int, default = [20,20,20], help = 'Number of filters in each convolutional per layer (sequence of ints) for the encoder (reverse order for the decoder)')
    parser.add_argument('--kernel_size', nargs='+', type=int, default = [3,3,3], help = 'CNN kernel size per layer (sequence of ints) for the encoder (reverse order for the decoder)')
    parser.add_argument('--strides', nargs='+', type=int, default = [1,1,1], help = 'CNN strides per layer (sequence of ints) for the encoder (reverse order for the decoder)')
    parser.add_argument('--pool_size_horizontal', nargs='+', type=int, default=[1,1,2], help='Horizontal max pooling size per CNN layer (1=None) for the encoder (reverse order for the UpSample decoder)')
    parser.add_argument('--pool_size_vertical', nargs='+', type=int, default=[1,1,2], help='vertical max pooling size per CNN layer (1=None) for the encoder (reverse order for the UpSample decoder)')
    
    # Recurrent Neural Network
    parser.add_argument('--rnn_units', nargs='+', type=int, default = [10, 10], help = 'Number of hidden units in the RNN layers, per layer (sequence of ints)')
    parser.add_argument('--rnn_model', nargs='+', type=str, default = ['GRU', 'GRU'], help = 'Type of recurrent layer it use RNN per layer (sequence of strings)')
    parser.add_argument('--rnn_activation', nargs='+', type=str, default = ['tanh', 'tanh'], help = 'Activation function to use in the RNN per layer (sequence of strings)')
    
    # Attention parameters; note setting ml_model = transformer will override encoder_decoder and create a transformer network
    parser.add_argument('--attention_heads', type=int, default=3, help='Number of heads in multi-headed attention')
    parser.add_argument('--inner_unit', type=int, default=5, help='Units of the output dense layer in the transformer encoder/decoder')
    parser.add_argument('--inner_activation', type=str, default='elu', help='Activation function for the transformer encoder/decoder inner dense layer')
    parser.add_argument('--encoder_decoder', type=str, default='encoder', help='Create a transformer encoder and/or a decoder block (options are encoder/decoder/both')
    
    
    
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
def split_data(data, ntrain_folds, rotation):
    '''
    Split a data set into a training, validation, and test datasets based on folds
    
    Inputs:
    :param data: Input data to be split. Must be in a Nfeature/Nmethod x time x space x fold format
    :param ntrain_folds: Number of folds to include in the training set
    :param rotation: The rotation of k-fold to use
    
    Outputs:
    :param train: Training dataset
    :param validation: Validation dataset
    :param test: Testing dataset
    '''
    
    # Initialize some values
    N, T, IJ, Nfolds = data.shape
    
    data_norm = data
            
    # Determine the training, validation, and test folds
    train_folds = (np.arange(ntrain_folds) + rotation) % Nfolds
    validation_folds = int((np.array([ntrain_folds]) + rotation) % Nfolds)
    test_folds = int((np.array([ntrain_folds]) + 1 + rotation) % Nfolds)
    
    # Collect the training, validation, and test data    
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
    
def generate_results_fname(model, ml_model, method):
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
    NV, T, IJ = train_in.shape
    NV, Tt, IJ = valid_in.shape
    
    
    ###### May need to change transpose variables
    # Normalize data?
    if args.normalize:
        for ij in range(IJ):
            if np.mod(ij, 1000) == 0: 
                print('%4.2f percent through normalization...'%(ij/IJ*100))
                    
            scaler_train = StandardScaler()
            scaler_valid = StandardScaler()
            scaler_test = StandardScaler()
            scaler_whole = StandardScaler()
            
            tmp_train = scaler_train.fit_transform(train_in[:,:,ij].T)
            tmp_val   = scaler_valid.fit_transform(valid_in[:,:,ij].T)
            tmp_test  = scaler_test.fit_transform(test_in[:,:,ij].T)
            train_in[:,:,ij] = tmp_train.T
            valid_in[:,:,ij] = tmp_val.T
            test_in[:,:,ij]  = tmp_test.T

            tmp_whole = scaler_whole.fit_transform(data_in[:,:,ij].T)
            data_in[:,:,ij] = tmp_whole.T

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
        except pickle.UnpicklingError | (EOFError):
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
    
    # Revmove multiclass labels (they are not needed outside of training)
    data_out[data_out == 2] = 0
    
    train_out[train_out == 2] = 0
    valid_out[valid_out == 2] = 0
    test_out[test_out == 2] = 0
    
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

        # Collect the accuracy metric?
        if metric == 'accuracy':
            e_train = metrics.accuracy_score(train_out, train_pred)
            e_valid = metrics.accuracy_score(valid_out, valid_pred)
            e_test = metrics.accuracy_score(test_out, test_pred)

        # Collect the AUC metric?
        elif metric == 'auc':
            # RFs don't have the decision_function() subfunction, so predict_proba() is used instead
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
                # decision_function() from SVMs only return 1D array without breaking it into 0s and 1s columns
                if args.ml_model == 'svm':
                    train_pred_auc = model.decision_function(train_in.T)
                    valid_pred_auc = model.decision_function(valid_in.T)
                    test_pred_auc = model.decision_function(test_in.T)
                else:
                    train_pred_auc = model.decision_function(train_in.T)[:,1]
                    valid_pred_auc = model.decision_function(valid_in.T)[:,1]
                    test_pred_auc = model.decision_function(test_in.T)[:,1]
            
            e_train = np.nan if only_zeros else metrics.roc_auc_score(train_out, train_pred_auc)
            e_valid = np.nan if (np.nansum(valid_out) == 0) | (np.nansum(valid_out) == valid_out.size) else metrics.roc_auc_score(valid_out, valid_pred_auc)
            e_test = np.nan if (np.nansum(test_out) == 0) | (np.nansum(test_out) == test_out.size) else metrics.roc_auc_score(test_out, test_pred_auc)

        # Collect precision?
        elif metric == 'precision':
            e_train = metrics.precision_score(train_out, train_pred)
            e_valid = metrics.precision_score(valid_out, valid_pred)
            e_test = metrics.precision_score(test_out, test_pred)

        # Collect recall?
        elif metric == 'recall':
            e_train = metrics.recall_score(train_out, train_pred)
            e_valid = metrics.recall_score(valid_out, valid_pred)
            e_test = metrics.recall_score(test_out, test_pred)

        # Collect the F1-Score?
        elif metric == 'f1_score':
            e_train = metrics.f1_score(train_out, train_pred)
            e_valid = metrics.f1_score(valid_out, valid_pred)
            e_test = metrics.f1_score(test_out, test_pred)
            
        # Collect the MSE?
        elif metric == 'mse':
            e_train = metrics.mean_squared_error(train_out, train_pred)
            e_valid = metrics.mean_squared_error(valid_out, valid_pred)
            e_test = metrics.mean_squared_error(test_out, test_pred)
            
        # Collect the MAE?
        elif metric == 'mae': # Might also add Cross-entropy
            e_train = metrics.mean_absolute_error(train_out, train_pred)
            e_valid = metrics.mean_absolute_error(valid_out, valid_pred)
            e_test = metrics.mean_absolute_error(test_out, test_pred)
            
        # Collect the TSS (True Skill Score)?
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
            if args.ml_model == 'svm':
                train_pred_roc = model.decision_function(train_in.T)
                valid_pred_roc = model.decision_function(valid_in.T)
                test_pred_roc = model.decision_function(test_in.T)
            else:
                train_pred_roc = model.decision_function(train_in.T)[:,1]
                valid_pred_roc = model.decision_function(valid_in.T)[:,1]
                test_pred_roc = model.decision_function(test_in.T)[:,1]
        
        # Group the FPR and TPR predictions into specific bins, each 0.0001 apart
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
  

def execute_keras_exp(args, train_in, valid_in, test_in, train_out, valid_out, test_out, data_in, data_out, rotation, 
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
    
    # Limit how much CPU TF takes
    #os.environ['OMP_NUM_THREADS'] = "10"
    #os.environ['TF_NUM_INTEROP_THREADS'] = 1
    #os.environ['TF_NUM_INTRAOP_THREADS'] = 1
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    
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
    
    # Reshape data
    T, I, J, NV = train_in.shape
    Tt, I, J, NV = valid_in.shape
    Ttot, I, J, NV = data_in.shape

     if args.ml_model.lower() == 'cnn':
         reshape_method = 'C'
     else:
         reshape_method = 'F'
    
    # Normalize data?
    if args.normalize:
        for i in range(I):
            if np.mod(i, 10) == 0: 
                print('%4.2f percent through normalization...'%(i/I*100))
            for j in range(J):
                    
                scaler_train = StandardScaler()
                scaler_valid = StandardScaler()
                scaler_test = StandardScaler()
                scaler_whole = StandardScaler()
                    
                train_in[:,i,j,:] = scaler_train.fit_transform(train_in[:,i,j,:])
                valid_in[:,i,j,:] = scaler_valid.fit_transform(valid_in[:,i,j,:])
                test_in[:,i,j,:] = scaler_test.fit_transform(test_in[:,i,j,:])

                data_in[:,i,j,:] = scaler_whole.fit_transform(data_in[:,i,j,:])
    
    
    # Remove the scalers once their use is done
    del scaler_train, scaler_valid, scaler_test, scaler_whole
    gc.collect()
    
    # output data is read in shapes of (1, T, I, J). Remove the 1.
    train_out = np.squeeze(train_out)
    valid_out = np.squeeze(valid_out)
    test_out = np.squeeze(test_out)
    data_out = np.squeeze(data_out)
    
    
    # Reshape output data for training
    train_out = train_out.reshape(T, I*J, order = reshape_method)
    valid_out = valid_out.reshape(Tt, I*J, order = reshape_method)
    test_out = test_out.reshape(Tt, I*J, order = reshape_method)
    
    # Record the loss
    loss = args.loss
        
    # Rearrange data for ANNs so that all time steps and grid points are examples
    if (args.ml_model.lower() == 'ann') | (args.ml_model.lower() == 'artificial_neural_network'):
        train_in = train_in.reshape(T*I*J, NV, order = reshape_method)
        valid_in = valid_in.reshape(Tt*I*J, NV, order = reshape_method)
        test_in = test_in.reshape(Tt*I*J, NV, order = reshape_method)

        train_out = train_out.reshape(T*I*J, order = reshape_method)
        valid_out = valid_out.reshape(Tt*I*J, order = reshape_method)
        test_out = test_out.reshape(Tt*I*J, order = reshape_method)
        
        data_in = data_in.reshape(Ttot*I*J, NV, order = reshape_method)
        data_out = data_out.reshape(Ttot*I*J, order = reshape_method)
        
        print(data_in.shape)
        print(valid_in.shape)
        print(test_in.shape)


    # Rearrange data for CNNs and U-nets so that all pentads are examples, and the convolution is along space axis
    #elif (args.ml_model.lower() == 'cnn') | (args.ml_model.lower() == 'convolutional_neural_network') | (args.ml_model.lower() == 'u-network') | (args.ml_model.lower() == 'autoencoder'):
    #    train_in = train_in.reshape(T, I*J, NV, order = 'F')
    #    valid_in = valid_in.reshape(Tt, I*J, NV, order = 'F')
    #    test_in = test_in.reshape(Tt, I*J, NV, order = 'F')
    #    
    #    data_in = data_in.reshape(Ttot, I*J, NV, order = 'F')
    #    data_out = data_out.reshape(Ttot, I*J, order = 'F')
        
    # Rearrange data for RNNs and transformers so that all grid points are examples, and they are recurrsive along the time axis
    elif (args.ml_model.lower() == 'rnn') | (args.ml_model.lower() == 'recurrent_neural_network') | (args.ml_model.lower() == 'cnn-rnn'): #| (args.ml_model.lower() == 'attention') | (args.ml_model.lower() == 'transformer'):
        train_in = train_in.reshape(T, I*J, NV, order = reshape_method)
        valid_in = valid_in.reshape(Tt, I*J, NV, order = reshape_method)
        test_in = test_in.reshape(Tt, I*J, NV, order = reshape_method)

        # For RNNs, move the time axis (first) to the second, so that it is trained along the time axis, and each grid is an example
        train_in = np.moveaxis(train_in, 0, 1)
        valid_in = np.moveaxis(valid_in, 0, 1)
        test_in = np.moveaxis(test_in, 0, 1)

        train_out = np.moveaxis(train_out, 0, 1)
        valid_out = np.moveaxis(valid_out, 0, 1)
        test_out = np.moveaxis(test_out, 0, 1)
        
        data_in = data_in.reshape(Ttot, I*J, NV, order = reshape_method)
        data_out = data_out.reshape(Ttot, I*J, order = reshape_method)
        
        data_in = np.moveaxis(data_in, 0, 1)
        data_out = np.moveaxis(data_out, 0, 1)
        
    # Collect the input shape
    input_shape = train_in.shape

    # Set up sample weights for the neural network
    weights = np.ones((train_out.shape[:]))
    
    # Set locations with sea values (data_out == 2) to have weights of 0 (no impact)
    weights = np.where(train_out == 2, 0, weights)

    
    # Determine class weights
    if np.invert(args.class_weight == None):
        
        # Set weights where there is flash drought
        weights = np.where(train_out == 1, args.class_weight, weights)
        
        # Randomly set 90% of the datapoints where there is no flash drought to 0 (evens the scale of FD to no FD events
        #weights = weights.flatten()
        #no_fd_ind = np.where(train_out.flatten() == 0)[0]
        #rand_choice = np.random.choice(no_fd_ind, size = int(np.round(no_fd_ind.size*0.90)), replace = False)
        #weights[rand_choice] = 0
        #weights = weights.reshape(T, I*J)
        
        class_weights = {0:1, 1:args.class_weight, 2:0}

    # Reshape gives weights the same shape as the model output
    weight_shape = []
    for ws in weights.shape:
        weight_shape.append(ws)
        
    weight_shape.append(1)
    
    weights = weights.reshape(weight_shape, order = reshape_method)

    
    # Define the labels in the case of binary or categorical cross-entropy losses
    if loss == 'binary_crossentropy':
        # Invert the binary labels for binary cross entropy; should force the model to balance between normalizing the inputs and putting 0s in the weights
        tmp_train_out = 1 - train_out
        tmp_valid_out = 1 - valid_out
        
    elif (loss == 'categorical_crossentropy') | (loss == 'focal'):
        # For the focal and categorical_crossentropy losses, there needs to be a dimension of 2 at the end (data is one-hot encoded along this axis).
        # Else only FD probability = 1 is predicted
        train_shape = []
        valid_shape = []
        for (ts, vs) in zip(train_out.shape, valid_out.shape):
            train_shape.append(ts)
            valid_shape.append(vs)
            
        train_shape.append(3)
        valid_shape.append(3)
            
        tmp_train_out = np.zeros((train_shape))
        tmp_valid_out = np.zeros((valid_shape))
        
        tmp_train_out[train_out == 0, 0] = 1
        tmp_train_out[train_out == 1, 1] = 1
        
        tmp_valid_out[valid_out == 0,0] = 1
        tmp_valid_out[valid_out == 1,1] = 1
    else:
        tmp_train_out = train_out
        tmp_valid_out = valid_out
    
    # Turn the data into a dataset
    if (args.ml_model.lower() == 'attention') | (args.ml_model.lower() == 'transformer'):
        # The attention networks in TF_models does not accept a third part to the dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_in, tmp_train_out)) 
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_in, tmp_train_out, weights))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_in, tmp_valid_out))
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_in))
    full_dataset  = tf.data.Dataset.from_tensor_slices((data_in))
    
    # Batch and prefetch the data
    train_dataset = train_dataset.batch(args.batch)
    valid_dataset = valid_dataset.batch(args.batch)
    test_dataset  = test_dataset.batch(args.batch)
    full_dataset  = full_dataset.batch(args.batch)

    train_dataset = train_dataset.prefetch(args.prefetch)
    valid_dataset = valid_dataset.prefetch(args.prefetch)
    test_dataset  = test_dataset.prefetch(args.prefetch)
    full_dataset  = full_dataset.prefetch(args.prefetch)
    
    # May see if this breaks the code
    del train_in, valid_in, test_in, data_in
    gc.collect()
    
    # Define the loss function for focal loss functions or variational autoencoders
    if args.variational:
        args.loss = variational_loss(loss = loss, gamma = args.focal_parameters[0], alpha = args.focal_parameters[1])

    elif loss == 'focal':
        args.loss = focal_loss(gamma = args.focal_parameters[0], alpha = args.focal_parameters[1])

    # Define custom objects if a combined loss function (variational autocoder) or focal loss was used (custom object so they can be loaded)
    if os.path.exists('%s/'%model_fname):
        # Make any custom objects?
        if args.variational:
            custom_objects = {'combine_loss': args.loss}
        elif loss == 'focal':
            custom_objects = {'focal_loss_fixed': args.loss}
        else:
            custom_objects = None
        
        # If the modeel exists, there is no need to make and train it
        model = keras.models.load_model(model_fname, custom_objects = custom_objects)
                
    else:
        
        if args.gpu: # Set up the model to run on GPU
            
            ### Code comes from slides made and presented by Dr. Andrew Fagg, at https://drive.google.com/file/d/1YH_zSKT7TblLMC4eofmUruOdKXmt28yx/view
            ### (accessed from the AI2ES website, ai2es.org/products/education/#shortcourses
            
            # Determine the available devices
            physical_devices = tf.config.get_visible_devices('GPU')
            n_physical_devices = len(physical_devices)
            
            # Set the memory growth look for all available RAM (set to False; also ensures all devices have the same memory growth flag)
            for physical_device in physical_devices:
                tf.config.experimental.set_memory_growth(physical_device, False)
                
            # Output to ensure the correct number of GPUs are being used compared to expectations
            print('There are %d GPUs\n'%n_physical_devices)
            
            # TensorFlow object that does some magic (Use documentation to expand on this
            strategy = tf.distribute.MirroredStrategy()
            
            # Build the mode within the strategy scope to run in the GPU
            with strategy.scope():
                 # Build the model
                model = build_keras_model(args, shape = input_shape)
                
                # Models built within this scope will mirror computation accross devices.
                # This is akin to running multiple input batches across multiple devices.
                # Note the splitting does not have to be uniform.

        else: # Else just build the model normally
            # Build the model
            model = build_keras_model(args, shape = input_shape)

            
        # Callbacks
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                          restore_best_weights=True,
                                                          min_delta=args.min_delta)
        
        
        #print(np.nansum(train_out == 1)/train_out.size, np.log(np.nansum(train_out == 1)/train_out.size))
        # Train the model
        history = model.fit(train_dataset,
                            shuffle = False, 
                            epochs = args.epochs, verbose = args.verbose>=2,
                            validation_data = (valid_dataset), 
                            callbacks = [early_stopping_cb],
                            use_multiprocessing = args.multiprocess)#,
                            #sample_weight = weights)
        
        
        # Report if verbosity is turned on
        if args.verbose >= 1:
            print(model.summary())
            keras.utils.plot_model(model, 
                                   to_file = '%s_plot.png'%model_fname, 
                                   show_shapes = True, show_layer_names = True)
        
        
        # Save the model
        model.save('%s'%(model_fname))
    
        # Remove some of the unused variables to to reduce RAM usage
        del tmp_train_out, tmp_valid_out, weights
        gc.collect()
    
    
    # Remove the 2 values to prevent errors
    train_out = np.where(train_out == 2, 0, train_out)
    valid_out = np.where(valid_out == 2, 0, valid_out)
    test_out = np.where(test_out == 2, 0, test_out)

    data_out = np.where(data_out == 2, 0, data_out)
    
    
    # Perform initial evaluations
    results = keras_evaluate_model(model, args, full_dataset, valid_dataset, test_dataset, data_out, valid_out, test_out, loss)
    #results = keras_evaluate_model(model, args, data_in, valid_in, test_in, data_out, valid_out, test_out, loss)
    
    # Note this can cause an error if the model is loaded instead of trained, since history would not exist
    # (Currently not known how to retain the training history from the loaded model stage)
    results['history'] = history.history
    
    # Reshape the 2D data back into 3D
    if (args.ml_model.lower() == 'rnn') | (args.ml_model.lower() == 'recurrent_neural_network'):
        results['train_predict'] = results['train_predict'].reshape(I, J, Ttot, order = reshape_method)
        results['valid_predict'] = results['valid_predict'].reshape(I, J, Tt, order = reshape_method)
        results['test_predict'] = results['test_predict'].reshape(I, J, Tt, order = reshape_method)
        
        # For RNNs, the time axis needs to be moved back to the first axis for consistency with other models
        results['train_predict'] = np.moveaxis(results['train_predict'], 2, 0)
        results['valid_predict'] = np.moveaxis(results['valid_predict'], 2, 0)
        results['test_predict'] = np.moveaxis(results['test_predict'], 2, 0)
    else:
        results['train_predict'] = results['train_predict'].reshape(Ttot, I ,J, order = reshape_method)
        results['valid_predict'] = results['valid_predict'].reshape(Tt, I ,J, order = reshape_method)
        results['test_predict'] = results['test_predict'].reshape(Tt, I ,J, order = reshape_method)
    
    # Evaluate the model for each grid point?
    if evaluate_each_grid:
        
        # Reshape output data back into 3D for mapping evaluation
        if (args.ml_model.lower() == 'rnn') | (args.ml_model.lower() == 'recurrent_neural_network'):
            train_out = np.squeeze(train_out.reshape(I, J, T, order = reshape_method))
            valid_out = np.squeeze(valid_out.reshape(I, J, Tt,  order = reshape_method))
            test_out = np.squeeze(test_out.reshape(I, J, Tt, order = reshape_method))
            
            data_out = data_out.reshape(I, J, Ttot, order = reshape_method)
            
            # For RNNs, the time axis needs to be moved back to the first axis for consistency with other models
            train_out = np.moveaxis(train_out, 2, 0)
            valid_out = np.moveaxis(valid_out, 2, 0)
            test_out = np.moveaxis(test_out, 2, 0)
            
            data_out = np.moveaxis(data_out, 2, 0)
        else:
            train_out = np.squeeze(train_out.reshape(T, I, J, order = reshape_method))
            valid_out = np.squeeze(valid_out.reshape(Tt, I, J,  order = reshape_method))
            test_out = np.squeeze(test_out.reshape(Tt, I, J, order = reshape_method))
            
            data_out = data_out.reshape(Ttot, I, J, order = reshape_method)
        
        # Initialize the gridded metrics
        eval_train_map = np.zeros((I, J, len(args.metrics))) * np.nan
        eval_valid_map = np.zeros((I, J, len(args.metrics))) * np.nan
        eval_test_map = np.zeros((I, J, len(args.metrics))) * np.nan
        
        # Perform the evaluation for each grid point
        for i in range(I):
            if np.mod(i/I*100, 10) == 0:
                    print('Currently %d through the spatial evaluation'%(int(i/I*100)))
            for j in range(J):
                if np.nansum(data_out[:,i,j]) == 0:
                    continue

                r_tmp = keras_evaluate_model(model, args, results['train_predict'][:,i,j], results['valid_predict'][:,i,j], 
                                             results['test_predict'][:,i,j], 
                                             data_out[:,i,j], valid_out[:,i,j], test_out[:,i,j], loss, pred = True)

                # See if this clears any erased memory from running the function
                gc.collect() # Clears deleted variables from memory 

                # Try reducing the size of map data by half to reduce the RAM memory
                eval_train_map[i,j,:] = np.float32(r_tmp['train_eval'])
                eval_valid_map[i,j,:] = np.float32(r_tmp['valid_eval'])
                eval_test_map[i,j,:] = np.float32(r_tmp['test_eval'])
            
            
        results['eval_train_map'] = eval_train_map
        results['eval_valid_map'] = eval_valid_map
        results['eval_test_map'] = eval_test_map

        
    # Save the results
    with open('%s.pkl'%results_fname, 'wb') as fn:
        pickle.dump(results, fn)
        
    return results


def keras_evaluate_model(model, args, train_in, valid_in, test_in, train_out, valid_out, test_out, loss, pred = False):
    '''
    Evaluate an keras deep learning (ML) model. Assumes the validation and test datasets have the same temporal dimension.
    
    Inputs:
    :param model: ML model being evaluated
    :param args: Argparse arguments
    :param train_in: Input training dataset used to train the model. Maybe the entire input dataset instead.
    :param valid_in: Input validation dataset to help validate the model
    :param test_in: Input testing dataset to test the model
    :param train_out: Output training dataset used to train the model. Maybe the entire output dataset instead.
    :param valid_out: Output validation dataset to help validate the model
    :param test_out: Output testing dataset to test the model
    :Param loss: String indicating the loss function used
    :param pred: Boolean indicating whether the input data are predictions
    
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
    if np.invert(pred):
        
        # Define the model predictions based on  loss function
        if loss == 'binary_crossentropy':
            train_pred = 1 - np.squeeze(model.predict(train_in))
            valid_pred = 1 - np.squeeze(model.predict(valid_in))
            test_pred = 1 - np.squeeze(model.predict(test_in))
        elif (loss == 'categorical_crossentropy') | (loss == 'focal'):
            # The size of the model outputs for ANNs will be different from other NNs
            if (args.ml_model.lower() == 'ann') | (args.ml_model.lower() == 'artificial_neural_network'):
                train_pred = np.squeeze(model.predict(train_in))[:,1]
                valid_pred = np.squeeze(model.predict(valid_in))[:,1]
                test_pred = np.squeeze(model.predict(test_in))[:,1]
            else:
                train_pred = np.squeeze(model.predict(train_in))[:,:,1]
                valid_pred = np.squeeze(model.predict(valid_in))[:,:,1]
                test_pred = np.squeeze(model.predict(test_in))[:,:,1]
        
        # Output some predictions to see how the model performed (max values < 0.5 means no FD was predicted)
        print(np.nanmin(train_pred), np.nanmin(valid_pred), np.nanmin(test_pred))
        print(np.nanmax(train_pred), np.nanmax(valid_pred), np.nanmax(test_pred))
        print(train_pred.shape)
        print(train_out.shape)
    else:
        train_pred = train_in
        valid_pred = valid_in
        test_pred = test_in
    
    only_zeros = (np.nansum(train_pred < 0.5) == 0) | (np.nansum(train_out) == 0)

    
    if np.invert(pred):
        # Collect information for the ROC curve
        if args.roc_curve:
            # Group TPR and FPR values into bins, each 0.0001 apart
            thresh = np.arange(0, 2, 1e-4)
            thresh = np.round(thresh, 4)
            results['fpr_train'] = np.ones((thresh.size)) * np.nan
            results['tpr_train'] = np.ones((thresh.size)) * np.nan
            results['fpr_valid'] = np.ones((thresh.size)) * np.nan
            results['tpr_valid'] = np.ones((thresh.size)) * np.nan
            results['fpr_test'] = np.ones((thresh.size)) * np.nan
            results['tpr_test'] = np.ones((thresh.size)) * np.nan

            # Calcualte TPR and FPR
            fpr_train, tpr_train, thresh_train = metrics.roc_curve(train_out.flatten(), train_pred.flatten())

            fpr_valid, tpr_valid, thresh_valid = metrics.roc_curve(valid_out.flatten(), valid_pred.flatten())

            fpr_test, tpr_test, thresh_test = metrics.roc_curve(test_out.flatten(), test_pred.flatten())

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



        # Turn predictions into binary values
        train_pred = np.where(train_pred >= 0.5, 1, 0)
        valid_pred = np.where(valid_pred >= 0.5, 1, 0)
        test_pred = np.where(test_pred >= 0.5, 1, 0)

        # Model predictions   
        results['train_predict'] = train_pred
        results['valid_predict'] = valid_pred
        results['test_predict'] = test_pred
    
        
    # Perform the evaluations for each metric
    for metric in args.metrics:

        # Collect accuracy?
        if metric == 'accuracy':
            e_train = metrics.accuracy_score(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.accuracy_score(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.accuracy_score(test_out.flatten(), test_pred.flatten())

        # Collect AUC?? Note some checks to ensure AUC is calculated without error if no FD is found
        elif metric == 'auc':
            e_train = np.nan if only_zeros else metrics.roc_auc_score(train_out.flatten(), train_pred.flatten())
            e_valid = np.nan if (np.nansum(valid_out < 0.5) == 0) | (np.nansum(valid_out < 0.5) == valid_out.size) else metrics.roc_auc_score(valid_out.flatten(), valid_pred.flatten())
            e_test = np.nan if (np.nansum(test_out < 0.5) == 0) | (np.nansum(test_out < 0.5) == test_out.size) else metrics.roc_auc_score(test_out.flatten(), test_pred.flatten())

        # Collect precision?
        elif metric == 'precision':
            e_train = metrics.precision_score(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.precision_score(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.precision_score(test_out.flatten(), test_pred.flatten())

        # Collect recall?
        elif metric == 'recall':
            e_train = metrics.recall_score(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.recall_score(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.recall_score(test_out.flatten(), test_pred.flatten())

        # Collect the F1-Score?
        elif metric == 'f1_score':
            e_train = metrics.f1_score(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.f1_score(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.f1_score(test_out.flatten(), test_pred.flatten())
            
        # Collect the MSE?
        elif metric == 'mse':
            e_train = metrics.mean_squared_error(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.mean_squared_error(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.mean_squared_error(test_out.flatten(), test_pred.flatten())
            
        # Collect the MAE?
        elif metric == 'mae': # Might also add Cross-entropy
            e_train = metrics.mean_absolute_error(train_out.flatten(), train_pred.flatten())
            e_valid = metrics.mean_absolute_error(valid_out.flatten(), valid_pred.flatten())
            e_test = metrics.mean_absolute_error(test_out.flatten(), test_pred.flatten())
            
        # Collect the TSS (True Skill Score)?
        elif metric == 'tss':
            # To get the true skill score, determine the true positives, true negatives, and false positives (summed over time and the region)
            tp_train = np.nansum(np.where((train_pred.flatten() >= 0.5) & (train_out.flatten() == 1), 1, 0))
            tp_valid = np.nansum(np.where((valid_pred.flatten() >= 0.5) & (valid_out.flatten() == 1), 1, 0))
            tp_test = np.nansum(np.where((test_pred.flatten() >= 0.5) & (test_out.flatten() == 1), 1, 0))
            
            tn_train = np.nansum(np.where((train_pred.flatten() < 0.5) & (train_out.flatten() == 0), 1, 0))
            tn_valid = np.nansum(np.where((valid_pred.flatten() < 0.5) & (valid_out.flatten() == 0), 1, 0))
            tn_test = np.nansum(np.where((test_pred.flatten() < 0.5) & (test_out.flatten() == 0), 1, 0))
            
            fp_train = np.nansum(np.where((train_pred.flatten() >= 0.5) & (train_out.flatten() == 0), 1, 0))
            fp_valid = np.nansum(np.where((valid_pred.flatten() >= 0.5) & (valid_out.flatten() == 0), 1, 0))
            fp_test = np.nansum(np.where((test_pred.flatten() >= 0.5) & (test_out.flatten() == 0), 1, 0))
            
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
                                    model_fname, evaluate_each_grid)
        
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

    # There is a strange bug in the christian method where most points (an unrealistically high number) in the first year was labeled with FD; turn them to 0 for now
    if args.globe & (args.method == 'christian'):
        data_out[:,:,:,0] = 0

    
    # Determine the data size
    Nvar, T, IJ, Nfold  = data_in.shape
    Nmethods = data_out.shape[0]
    
    # Remove NaNs?
    if args.remove_nans:
        #if args.keras:
        if args.ml_model == 'svm':
            for var in range(Nvar):
                data_in[var,np.isnan(data_in[var,:,:,:])] = 0
            
            data_out[np.isnan(data_out)] = 0
            
        else:
            for var in range(Nvar):
                data_in[var,np.isnan(data_in[var,:,:,:])] = np.nanmean(data_in[var,:,:,:])

            data_out[np.isnan(data_out)] = 2
        #else:
        #    data_in[np.isnan(data_in)] = -995
        #    data_out[np.isnan(data_out)] = 0
    
    print('Input size (NVariables x time x space x NFolds):', data_in.shape)
    print('Output size (NMethods x time x space x NFolds):', data_out.shape)
    
    
    # Load example data with subsetted lat/lon data
    if args.globe:
        # Lat/lon data in evap is off in the global dataset on account of sea grids being removed.
        # Load the lat and lon from the pickle file instead
        with open("%s/%s"%(dataset_dir, args.input_data_fname), "rb") as fp:
            _ = pickle.load(fp)
            lat = pickle.load(fp)
            lon = pickle.load(fp)
            
        # The grid size with the sea values removed is ambiguous since lat/lon are vectors, so a specific number for I or J is needed
        # Here 201 was found to be one of the primes for I.size
        I = 201
        J = int(lat.size/I)

        ### Test
        # if test:
        #     ind = np.where( ((lat > 23) & (lat < 49)) & ((lon > 230) & (lon < 295)) )[0]
        #     data_in = data_in[:,:,ind,:]
        #     data_out = data_out[:,:,ind,:]
        #     I = 359
        #     J = int(data_in.shape[2]/I)

        #     print(data_in.shape)

        #     lat = lat[ind].reshape(I, J, order = 'F')
        #     lon = lon[ind].reshape(I, J, order = 'F')
        # else:
        #     # Reshape lat/lon into 2D arrays
        #     lat = lat.reshape(I, J, order = 'F')
        #     lon = lon.reshape(I, J, order = 'F')

        lat = lat.reshape(I, J, order = 'F')
        lon = lon.reshape(I, J, order = 'F')

    
    else:
        et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
        lat = et['lat']; lon = et['lon']
    
        # Remove the unused data
        del et
        gc.collect()
    
        # Collect the spatial size of the data
        I, J = lat.shape


    # Create a version of the entire dataset without being split
    data_in_whole = np.concatenate([data_in[:,:,:,fold] for fold in range(Nfold)], axis = 1)
    data_out_whole = np.concatenate([data_out[method_ind,:,:,fold] for fold in range(Nfold)], axis = 1)
    
    # Keras data needs to be rearranged (NVariables/NFilters needs to be at the end, and each time step is an example)
    if args.keras:
        # Move the first axis (NVar) to the last axis
        data_in_whole = np.moveaxis(data_in_whole, 0, -1)
        
        # Reorder the main data into 3D (4D with NVar/NMethods) data
        data_in_whole = data_in_whole.reshape(Nfold*T, I, J, Nvar, order = 'F')
        data_out_whole = data_out_whole.reshape(Nfold*T, I, J, order = 'F')
    
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
            train_in, valid_in, test_in = split_data(data_in, args.ntrain_folds, rot)
            train_out, valid_out, test_out = split_data(data_out, args.ntrain_folds, rot) # Note the label data is already binary
            
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
                
            
            # Keras data needs to be rearranged (NVariables/NFilters needs to be at the end, and each time step is an example)
            if args.keras:
                # Move the first axis (NVar) to the last axis
                train_in = np.moveaxis(train_in, 0, -1)
                valid_in = np.moveaxis(valid_in, 0, -1)
                test_in = np.moveaxis(test_in, 0, -1)
                
                
                # Reorder the main data into 3D (4D with NVar/NMethods) data
                train_in = train_in.reshape(train_in.shape[0], I, J, Nvar, order = 'F')
                valid_in = valid_in.reshape(valid_in.shape[0], I, J, Nvar, order = 'F')
                test_in = test_in.reshape(test_in.shape[0], I, J, Nvar, order = 'F')
                
                train_out = train_out.reshape(Nmethods, train_out.shape[1], I, J, order = 'F')
                valid_out = valid_out.reshape(Nmethods, valid_out.shape[1], I, J, order = 'F')
                test_out = test_out.reshape(Nmethods, test_out.shape[1], I, J, order = 'F')
                
                
                # Perform the experiment
                # The ML model is saved in this step
                execute_single_exp(args, train_in, valid_in, test_in, 
                                   train_out[method_ind,:,:,:], valid_out[method_ind,:,:,:], test_out[method_ind,:,:,:], 
                                   data_in_whole, data_out_whole[:,:,:], rot,
                                   model_fname, evaluate_each_grid = True)
                
            else:

                # Perform the experiment
                # The ML model is saved in this step
                execute_single_exp(args, train_in, valid_in, test_in, 
                                   train_out[method_ind,:,:], valid_out[method_ind,:,:], test_out[method_ind,:,:], 
                                   data_in_whole, data_out_whole[:,:], rot,
                                   model_fname, evaluate_each_grid = True)
    
    # Otherwise, train for 1 rotation at a time
    else:
        # Generate the model filename
        model_fbase = generate_model_fname(args.ra_model, args.label, args.method, args.rotation[0])
        model_fname = '%s/%s/%s/%s'%(dataset_dir, args.ml_model, args.method, model_fbase)
        # print(model_fname)
        
        # Leave if the experiment has already been completed
        if os.path.exists(model_fname):
            print('File already exists/experiment has already been performed')
            #return
        
        # Split the data into training, validation, and test sets
        print('Splitting data training/validation/test sets...')
        train_in, valid_in, test_in = split_data(data_in, args.ntrain_folds, args.rotation)
        train_out, valid_out, test_out = split_data(data_out, args.ntrain_folds, args.rotation) # Note the label data is already binary

        # Remove very large files from the RAM
        del data_in, data_out
        gc.collect()
        

        # If the training data has too few FD, bagging can easly find some subsets that have no FD, resulting in a class weight error.
        # Ignore the class weights for this scenario
        if (100*np.where(train_out == 1)[0].size/train_out.size) < 0.05:
            args.class_weight = None
        else:
            args.class_weight = weight

        # Perform the experiment
        # The ML model is saved in this step
        if args.keras:
            # Move the first axis (NVar) to the last axis
            train_in = np.moveaxis(train_in, 0, -1)
            valid_in = np.moveaxis(valid_in, 0, -1)
            test_in = np.moveaxis(test_in, 0, -1)


            # Reorder the main data into 3D (4D with NVar/NMethods) data
            train_in = train_in.reshape(train_in.shape[0], I, J, Nvar, order = 'F')
            valid_in = valid_in.reshape(valid_in.shape[0], I, J, Nvar, order = 'F')
            test_in = test_in.reshape(test_in.shape[0], I, J, Nvar, order = 'F')

            train_out = train_out.reshape(Nmethods, train_out.shape[1], I, J, order = 'F')
            valid_out = valid_out.reshape(Nmethods, valid_out.shape[1], I, J, order = 'F')
            test_out = test_out.reshape(Nmethods, test_out.shape[1], I, J, order = 'F')

            # Perform the experiment
            # The ML model is saved in this step
            execute_single_exp(args, train_in, valid_in, test_in, 
                               train_out[method_ind,:,:,:], valid_out[method_ind,:,:,:], test_out[method_ind,:,:,:], 
                               data_in_whole, data_out_whole[:,:,:], args.rotation[0],
                               model_fname, evaluate_each_grid = True)
                
        else:
            # Perform the experiment
            # The ML model is saved in this step
            execute_single_exp(args, train_in, valid_in, test_in, 
                               train_out[method_ind,:,:], valid_out[method_ind,:,:], test_out[method_ind,:,:], 
                               data_in_whole, data_out_whole[:,:], args.rotation[0],
                               model_fname, evaluate_each_grid = True)

    # If this is a test run, merge the results over test rotations (otherwise, this is done separately in the main function, after all rotations are trained)
    if test:
        print('Merging the results of %s for the %s method...'%(args.ml_model, args.method))
        results = merge_results(args, args.method, lat, lon, Nfold, Nvar, T, I, J)
        
    print('Done.')
    return 


def merge_results(args, method, lat, lon, NFolds, NVar, T, I, J, data_in = None, data_out = None):
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
    :param data_in: The input dataset for the ML models. Most be entered if collecting feature attribution/importance
    
    Outputs:
    .pkl file containing merged results
    '''
    
    if data_in is not None:
        data_in = data_in.astype(np.float32)
    
    if data_out is not None:
        data_out = data_out.astype(np.float32)
    
    # Construct the base file name for each result
    model_fbase = 'results_%s_%s_%s_rot_'%(args.ra_model, 
                                           args.label, 
                                           method)
    
    dataset_dir = '%s/%s/%s/%s'%(args.dataset, args.ra_model, args.ml_model, method)
    dataset_dir_hub = '%s/%s'%(args.dataset, args.ra_model)
    model_fname = '%s/%s'%(dataset_dir, model_fbase)
    
    # Generate the name of the overall results file        
    results_fbase = generate_results_fname(args.ra_model, args.label, method)
    results_fname = '%s/%s'%(dataset_dir_hub, results_fbase)
    print(results_fname)
    
    # If the file already exists, simply load the file
    if os.path.exists('%s'%results_fname):
        print('The model results have already been merged. Loading the results...')
        
        with open(results_fname, 'rb') as fn:
            results = pickle.load(fn)
        
        return results
    
    # Collect the files for all rotations
    # NOTE: filenames with single digit numbers need leading zeros to sort properly (e.g., 01, 02, 03, ...)
    files = ['%s/%s'%(dataset_dir,f) for f in os.listdir(dataset_dir) if re.match(r'%s.+.pkl'%(model_fbase), f)]
    files.sort()
    
    print(files)
    
    Nrot = len(files)
    
    # Initialize results
    if args.keras: # Keras models are already in a 3D shape
        pred_train = np.ones((Nrot, NFolds*T, I, J), dtype = np.float32) * np.nan
        pred_valid = np.ones((NFolds*T, I, J), dtype = np.float32) * np.nan
        pred_test = np.ones((NFolds*T, I, J), dtype = np.float32) * np.nan
    else:
        pred_train = np.ones((Nrot, NFolds*T, I*J), dtype = np.float32) * np.nan
        pred_valid = np.ones((NFolds*T, I*J), dtype = np.float32) * np.nan
        pred_test = np.ones((NFolds*T, I*J), dtype = np.float32) * np.nan
    
    eval_train = np.ones((len(args.metrics))) * np.nan
    eval_valid = np.ones((len(args.metrics))) * np.nan
    eval_test = np.ones((len(args.metrics))) * np.nan
    
    eval_train_var = np.ones((len(args.metrics))) * np.nan
    eval_valid_var = np.ones((len(args.metrics))) * np.nan
    eval_test_var = np.ones((len(args.metrics))) * np.nan
    
    eval_train_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan
    eval_valid_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan
    eval_test_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan

    eval_train_var_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan
    eval_valid_var_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan
    eval_test_var_map = np.ones((I,J,len(args.metrics)), dtype = np.float32) * np.nan
    
    if args.interpret:
        feature_import = np.ones((NVar)) * np.nan
        feature_import_var = np.ones((NVar)) * np.nan
        attributions = np.ones((NFolds*T, NVar)) * np.nan
        attributions_cs = np.ones((len(args.case_study_years), T, NVar)) * np.nan
    
    if args.keras:
        learn_curves = np.ones((args.epochs)) * np.nan
        learn_curves_var = np.ones((args.epochs)) * np.nan
    
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
    fi_var = []
    fi_gini = []
    fi_pi = []
    fi_pi_var = []

    fpr_train_tmp = []
    fpr_valid_tmp = []
    fpr_test_tmp = []

    tpr_train_tmp = []
    tpr_valid_tmp = []
    tpr_test_tmp = []

    if args.interpret:
        if data_in is None:
            assert "data_in must be entered to interpret models!"
        else:
            # The data loaded by the pkl file is loaded as a masked array; must be turned into a np array for shap
            if np.ma.isMaskedArray(data_in):
            	data_in = data_in.filled(fill_value = 0)
            years = np.array([1979 + rot for rot in range(Nrot)])
    
    # Collect the results for each rotation
    for rot, f in enumerate(files):
        print(f)
        with open(f, 'rb') as fn:
            result = pickle.load(fn)

        #rotation = int(f[68:70]) # Only true for C23 method, remove after!

        train_folds = (np.arange(args.ntrain_folds) + rot) % NFolds
        train_folds = np.sort(train_folds)
        val_folds = int((np.array([args.ntrain_folds]) + rot) % NFolds)
        test_folds = int((np.array([args.ntrain_folds]) + 1 + rot) % NFolds)
        print(test_folds)
        
        # For the "training" results loaded, recall it is predictions for all 43 folds
        train_fold_tmp = (np.arange(NFolds) + rot) % NFolds
        train_fold_tmp = np.sort(train_fold_tmp)
        train_fold_ind = np.concatenate([np.arange(T*fold, T*(fold+1)) for fold in train_fold_tmp])
        print(train_fold_ind)

        
        # "train" set is for the entire dataset; this gets averaged together in the merged results
        # Valid and test predictions get "stacked" together in temporal order (each rotation should only predict 1 fold for validation and test each)
        #ptrain.append(result['train_predict'])
        pred_train[rot, train_fold_ind, :] = result['train_predict']
        pred_valid[val_folds*T:(val_folds+1)*T,:] = result['valid_predict']
        pred_test[test_folds*T:(test_folds+1)*T,:] = result['test_predict']
        
        etrain.append(result['train_eval'])
        evalid.append(result['valid_eval'])
        etest.append(result['test_eval'])

        etrain_map.append(result['eval_train_map'].astype(np.float32))
        evalid_map.append(result['eval_valid_map'].astype(np.float32))
        etest_map.append(result['eval_test_map'].astype(np.float32))

        if args.roc_curve:
            fpr_train_tmp.append(result['fpr_train'])
            fpr_valid_tmp.append(result['fpr_valid'])
            fpr_test_tmp.append(result['fpr_test'])

            tpr_train_tmp.append(result['tpr_train'])
            tpr_valid_tmp.append(result['tpr_valid'])
            tpr_test_tmp.append(result['tpr_test'])

        if args.interpret:
            # Obtain the background data
            # Make a time series of the input data
            train = np.concatenate([data_in[:,:,:,fold] for fold in train_folds], axis = 1)

            data_in_ts = data_in[:,:,:,test_folds]
            # Standardize the data
            for ij in range(IJ):
                scaler_train = StandardScaler()
                scaler_test = StandardScaler()
            
                tmp_train = scaler_train.fit_transform(train[:,:,ij].T)
                tmp_test  = scaler_test.fit_transform(data_in_ts[:,:,ij].T)
                train[:,:,ij]  = tmp_train.T
                data_in_ts[:,:,ij] = tmp_test.T
        
            train_ts = np.nanmean(train, axis = -1)
            data_in_ts = np.nanmean(data_in_ts, axis = -1)
            
            # Create the model name
            model_fbase = '%s_%s_%s_rot_%02d'%(args.ra_model, 
                                               args.label, 
                                               method,
                                               rot)
            feature_names = ['T', 'ET', r'$\Delta$ET', 'PET', r'$\Delta$PET', 'P', 'SM', r'$\Delta$SM']

            # Load the model
            model = load_single_model('%s/%s/%s/%s/%s'%(args.dataset, args.ra_model, args.ml_model, method, model_fbase), args.keras)

            # Determine which prediction method to use for the explanation
            if (args.ml_model.lower() == 'ann') | (args.ml_model.lower() == 'artificial_neural_network'):
                explain_method = model.predict
            elif (args.ml_model.lower() == 'svm') | (args.ml_model.lower() == 'support_vector_machine'):
                explain_method = model.decision_function
            else:
                explain_method = model.predict_proba

            # Explain the model
            explainer = shap.KernelExplainer(explain_method, data = train_ts.T, feature_names = feature_names)

            # Collect the shapley values
            shap_values = explainer.shap_values(data_in_ts.T)
            
            # The decision_function works differently and produces shap values differently (list entry per example, and each list is an entry to each feature)
            # Reshape into a standard list (list entry for each class, each list entry an array of Nexamples x Nfeatures)
            if (args.ml_model.lower() == 'svm') | (args.ml_model.lower() == 'support_vector_machine'):
                n_examples = len(shap_values)
                n_features = shap_values[0].shape[0]
                tmp_values = np.ones((n_examples, n_features)) * np.nan
                tmp = []
                tmp.append(0) # First list entry is unimportant (class 0); it isn't used in the final results
                for n, value in enumerate(shap_values):
                    tmp_values[n,:] = value # An entry here for each features; repeat for each example
                
                tmp.append(tmp_values)
                shap_values = tmp
                

            # Calculate the feature importance
            importances = []
            importances_var = []
            for feature in range(shap_values[1].shape[1]):
                importances.append(np.nanmean(np.abs(shap_values[1][:,feature])))
                importances_var.append(np.nanstd(np.abs(shap_values[1][:,feature])))
            
            #importances = softmax(importances)
            importances = np.array(importances)

            # Store the values
            attributions[test_folds*T:(test_folds+1)*T,:] = shap_values[1]
            fi.append(importances)
            fi_var.append(importances_var)

            # For the special case of case studies, where attribution is examined overa specific domain
            year = years[test_folds]
            if year in args.case_study_years:
               ind = np.where(year == args.case_study_years)[0]
                
               # Subset the data to the specific domain
               train = np.array([get_domain(train[var,:,:], lat, lon, year, globe = args.globe) for var in range(NVar)])
               data_in_subset = np.array([get_domain(data_in[var,:,:,test_folds], lat, lon, year, globe = args.globe, sea_points = False) for var in range(NVar)])

               train_ts = np.nanmean(train, axis = -1)
               data_in_ts = np.nanmean(data_in_subset, axis = -1)

               #  Explain the model
               explainer = shap.KernelExplainer(explain_method, data = train_ts.T, feature_names = feature_names)
    
               # Collect the shapley values
               shap_values = explainer.shap_values(data_in_ts.T)

               attributions_cs[ind[0],:,:] = shap_values[1]

            if (args.ml_model == 'rf') | (args.ml_model == 'ada'):
                # Get the input data for permutation importance for test folds (test folds because size reduction is needed to not crash the computer)
                ins = np.concatenate([data_in[:,:,:,test_folds]], axis = 1)
                Nvar, IJ_train, T_train = ins.shape
                
                # Standardize the input data
                for ij in range(IJ):
                    scaler_test = StandardScaler()
            
                    tmp_test  = scaler_test.fit_transform(ins[:,:,ij].T)
                    ins[:,:,ij] = tmp_test.T
                
                ins = ins.reshape(Nvar, IJ_train*T_train, order = 'F')
                
                # Remove NaNS    
                for var in range(Nvar):
                    ins[var,np.isnan(ins[var,:])] = np.nanmean(ins[var,:]) 

                # Get the output labels for the permutation importance
                outs = np.concatenate([data_out[:,:,test_folds]], axis = 1)
                outs = outs.reshape(IJ_train*T_train, 1, order = 'F')
                
                # Remove NaNs
                print(ins.shape)
                outs[np.isnan(outs)] = 0

                # Feature names for permutation importance
                feature_names = ['T', 'ET', r'$\Delta$ET', 'PET', r'$\Delta$PET', 'P', 'SM', r'$\Delta$SM']

                # Create the explainer
                explainer = skexplain.ExplainToolkit(estimators = (args.ml_model, model), X = ins.T, y = outs, feature_names = feature_names)

                # Permutation importance; rpss (ranked probability skill score) used for multiclassification (important features have a minimized rpss)
                importances = explainer.permutation_importance(n_vars = len(feature_names), evaluation_fn = 'rpss', 
                                                               scoring_strategy = 'minimize', n_permute = 50, 
                                                               subsample = 0.1, n_jobs = 2, verbose = True, direction = 'backward')

                # Collect the multipass results
                pi_key = '%s_%s'%('backward_multipass_scores_', args.ml_model)
                rankings_key = '%s_%s'%('backward_multipass_rankings_', args.ml_model)
                
                pi = importances[pi_key].values
                rankings = importances[rankings_key].values
                pi_fi = []
                pi_fi_var = []
                
                # Take the average and standard deviation ins PI scores across all permutations (put them in the same order as in feature_names)
                for feature in feature_names:
                    ind = np.where(feature == rankings)[0]
                    pi_fi.append(np.nanmean(np.array(pi[ind,:])))
                    pi_fi_var.append(np.nanstd(np.array(pi[ind,:])))
                    
                pi_fi = np.array(pi_fi)
                pi_fi_var = np.array(pi_fi_var)
                  
                print(pi_fi.shape)
                print(pi_fi_var.shape)  
                # Collect importances averaged over all permutations
                fi_pi.append(pi_fi)
                fi_pi_var.append(pi_fi_var)

            # Collect the GINI importance for tree based models
            if args.feature_importance:
                importances = model.feature_importances_
                fi_gini.append(importances)
                
                

        if args.keras:
           lc.append(result['history'])
    
    results = {}
    # Model coordinates
    results['lat'] = lat; results['lon'] = lon
            
    # Merge the results
    pred_train = np.round(np.nanmean(pred_train, axis = 0), 0)
    #pred_train = np.round(np.nanmean(np.stack(ptrain, axis = -1), axis = -1), 0) # The round restores the average back to binary 1 or 0; 
    pred_valid = np.round(pred_valid, 0)                                         # average < 0.5 means majority of rotations did not identify FD
    pred_test = np.round(pred_test, 0)
    
    # Model predictions
    if args.keras:
        results['all_predict'] = pred_train
        results['valid_predict'] = pred_valid
        results['test_predict'] = pred_test
    else:
        results['all_predict'] = pred_train.reshape(NFolds*T, I, J, order = 'F')
        results['valid_predict'] = pred_valid.reshape(NFolds*T, I, J, order = 'F')
        results['test_predict'] = pred_test.reshape(NFolds*T, I, J, order = 'F')
        
    del pred_train, pred_valid, pred_test, ptrain
    gc.collect()
    
    # Overall model performance
    results['all_eval'] = np.nanmean(np.stack(etrain, axis = -1), axis = -1)
    results['valid_eval'] = np.nanmean(np.stack(evalid, axis = -1), axis = -1)
    results['test_eval'] = np.nanmean(np.stack(etest, axis = -1), axis = -1)
    
    results['all_eval_var'] = np.nanstd(np.stack(etrain, axis = -1), axis = -1)
    results['valid_eval_var'] = np.nanstd(np.stack(evalid, axis = -1), axis = -1)
    results['test_eval_var'] = np.nanstd(np.stack(etest, axis = -1), axis = -1)

    eval_train_map = np.nanmean(np.stack(etrain_map, axis = -1), axis = -1)
    eval_valid_map = np.nanmean(np.stack(evalid_map, axis = -1), axis = -1)
    eval_test_map = np.nanmean(np.stack(etest_map, axis = -1), axis = -1)

    eval_train_var_map = np.nanstd(np.stack(etrain_map, axis = -1), axis = -1)
    eval_valid_var_map = np.nanstd(np.stack(evalid_map, axis = -1), axis = -1)
    eval_test_var_map = np.nanstd(np.stack(etest_map, axis = -1), axis = -1)   
    
    # Model performance over each individual grid point
    if args.keras:
        results['all_eval_map'] = eval_train_map
        results['valid_eval_map'] = eval_valid_map
        results['test_eval_map'] = eval_test_map

        results['all_eval_var_map'] = eval_train_var_map
        results['valid_eval_var_map'] = eval_valid_var_map
        results['test_eval_var_map'] = eval_test_var_map
    else:
        results['all_eval_map'] = eval_train_map.reshape(I, J, len(args.metrics), order = 'F')
        results['valid_eval_map'] = eval_valid_map.reshape(I, J, len(args.metrics), order = 'F')
        results['test_eval_map'] = eval_test_map.reshape(I, J, len(args.metrics), order = 'F')

        results['all_eval_var_map'] = eval_train_var_map.reshape(I, J, len(args.metrics), order = 'F')
        results['valid_eval_var_map'] = eval_valid_var_map.reshape(I, J, len(args.metrics), order = 'F')
        results['test_eval_var_map'] = eval_test_var_map.reshape(I, J, len(args.metrics), order = 'F')
        
    del eval_train_map, eval_valid_map, eval_test_map, eval_train_var_map, eval_valid_var_map, eval_test_var_map
    gc.collect()

    # Merge ROC curves
    if args.roc_curve:
        results['fpr_all'] = np.nanmean(np.stack(fpr_train_tmp, axis = -1), axis = -1)
        results['fpr_valid'] = np.nanmean(np.stack(fpr_valid_tmp, axis = -1), axis = -1)
        results['fpr_test'] = np.nanmean(np.stack(fpr_test_tmp, axis = -1), axis = -1)

        results['fpr_all_var'] = np.nanstd(np.stack(fpr_train_tmp, axis = -1), axis = -1)
        results['fpr_valid_var'] = np.nanstd(np.stack(fpr_valid_tmp, axis = -1), axis = -1)
        results['fpr_test_var'] = np.nanstd(np.stack(fpr_test_tmp, axis = -1), axis = -1)

        results['tpr_all'] = np.nanmean(np.stack(tpr_train_tmp, axis = -1), axis = -1)
        results['tpr_valid'] = np.nanmean(np.stack(tpr_valid_tmp, axis = -1), axis = -1)
        results['tpr_test'] = np.nanmean(np.stack(tpr_test_tmp, axis = -1), axis = -1)

        results['tpr_all_var'] = np.nanstd(np.stack(tpr_train_tmp, axis = -1), axis = -1)
        results['tpr_valid_var'] = np.nanstd(np.stack(tpr_valid_tmp, axis = -1), axis = -1)
        results['tpr_test_var'] = np.nanstd(np.stack(tpr_test_tmp, axis = -1), axis = -1)


    if args.interpret:
        results['attributions'] = attributions
        results['attributions_cs'] = attributions_cs
        results['feature_import'] = np.nanmean(np.stack(fi, axis = -1), axis = -1)
        results['feature_import_var'] = np.nanmean(np.stack(fi_var, axis = -1), axis = -1)

        if (args.ml_model == 'rf') | (args.ml_model == 'ada'):
            results['feature_import_pi'] = np.nanmean(np.stack(fi_pi, axis = -1), axis = -1)
            results['feature_import_pi_var'] = np.nanmean(np.stack(fi_pi_var, axis = -1), axis = -1)

        if args.feature_importance:
            results['feature_import_gini'] = np.nanmean(np.stack(fi_gini, axis = -1), axis = -1)
            results['feature_import_gini_var'] = np.nanstd(np.stack(fi_gini, axis = -1), axis = -1)

    # Merge learning curves
    ### ADD SPIGGATI LEARNING CURVES
    if args.keras:
        learn_curves = {}
        learn_curves_var = {}
        for key in lc[0].keys():
            tmp = np.zeros((NFolds, args.epochs)) * np.nan
            for n, curve in enumerate(lc):
                n_epochs = len(curve[key])

                tmp[n,:n_epochs] = curve[key]
               
            learn_curves[key] = np.nanmean(tmp, axis = 0)
            learn_curves_var[key] = np.nanstd(tmp, axis = 0)
           
        results['history'] = learn_curves
        results['history_var'] = learn_curves_var
        
        
    # Potentially halve some of the datasize
    for key in results.keys():
        results[key] = results[key].astype(np.float32)
        
    # Save the results
    with open("%s"%(results_fname), "wb") as fp:
        pickle.dump(results, fp)
        
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
        filename = '%s_%s_%s_metric_performance_across_rotations.png'%(args.label, method, metric)
        plt.savefig('%s/%s'%(dataset_dir_hub, filename), bbox_inches = 'tight')
        plt.show(block = False)

    return results


# Function to obtain the domain for a specific region
def get_domain(data, lat, lon, year, globe = False, sea_points = False):
    '''
    Subset a dataset to a specific domain

    Inputs:
    :param data: Data to be subsetted. Needs to be in a time x lat*lon format
    :param lat: 2D Latitude data for data
    :param lon: 2D Longitude data for data
    :param year: Int. Year the data is being subsetted for
    :param globe: Boolean indicating whether the data is global or not
    :param sea_points: Boolean when global datasets are used - indicates if sea grid points are in the dataset
    '''

    # Determine domain of focus for the specific year
    if globe:
        if year == 2001:
            # India
            lon_min = 62
            lon_max = 100
            lat_min = 0
            lat_max = 40
            
        elif year == 2010:
            # Russia
            lon_min = 28
            lon_max = 52
            lat_min = 44
            lat_max = 59
            
        elif year == 2015:
            # Southern Africa
            # lon_min = 10
            # lon_max = 40
            # lat_min = -35
            # lat_max = -10

            # Amazon
            # Remember ERA5 lon specifically goes from 0 to 360
            lon_min = -76+360
            lon_max = -39+360
            lat_min = -24
            lat_max = 0
            
        elif year == 2016:
            # Eastern Africa
            lon_min = 33
            lon_max = 55
            lat_min = 0
            lat_max = 20
                
        elif year == 2018:
            # Southeast Australia
            lon_min = 142
            lon_max = 154
            lat_min = -42
            lat_max = -28

    else:
        if year == 1988:
            lon_min = -115
            lon_max = -80
            lat_min = 30
            lat_max = 50
            
        elif year == 2000:
            lon_min = -105
            lon_max = -79
            lat_min = 28
            lat_max = 45
    
        elif year == 2003:
            lon_min = -107
            lon_max = -83
            lat_min = 30
            lat_max = 50
            
        elif year == 2011:
            lon_min = -110
            lon_max = -78
            lat_min = 25
            lat_max = 42
            
        elif year == 2012:
            lon_min = -105
            lon_max = -82
            lat_min = 30
            lat_max = 50
            
        elif year == 2017:
            lon_min = -118
            lon_max = -95
            lat_min = 40
            lat_max = 50
            
        elif year == 2019:
            lon_min = -107
            lon_max = -74
            lat_min = 25
            lat_max = 40

    # Reshape the data into 3D arrays for the subset function
    if globe: ############################### Update this on how to use it properly (block is needed for getting attributions, but errors in the case studies)
        T, IJ = data.shape
        if np.invert(sea_points):
            print("Sea points were not restored.")
            I = 201
            J = int(IJ/I)
        else:
            print("Sea points were restored.")
            I, J = lat.shape
    else:
        I, J = lat.shape
        T = data.shape[0]
        
    data_sub = data.reshape(T, I, J, order = 'F')
    
    # Subset the data
    data_sub, _, _ = subset_data(data_sub, lat, lon, 
                                 LatMin = lat_min, LatMax = lat_max, 
                                 LonMin = lon_min, LonMax = lon_max)

    # Reshape the data back into a 2D array
    T, I, J = data_sub.shape
    data_sub = data_sub.reshape(T, I*J, order = 'F')

    return data_sub


#%
##############################################
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_ml_parser()
    args = parser.parse_args()
    
    # Silence warnings
    warnings.simplefilter('ignore')
    
    # Execute the experiments?
    if np.invert(args.nogo):
        print('Performing experiment...')
        execute_exp(args)
    
    # Perform model evaluations instead? (This is done after all rotations for all methods are run)
    if args.evaluate:
        print('Initializing some variables...')
        methods = ['christian', 'nogeura', 'pendergrass', 'liu', 'otkin']
        #methods = ['otkin']
        
        # Get the directory of the dataset
        dataset_dir = '%s/%s'%(args.dataset, args.ra_model)

        # Load the data
        # Data is Nfeatures/Nmethods x time x space x fold
        data_in = load_ml_data(args.input_data_fname, path = dataset_dir)
        data_out = load_ml_data(args.output_data_fname, path = dataset_dir)

        # Make the rotations
        Nvar, T, IJ, Nfolds = data_in.shape

        # Load the mask; might be worth moving this into the if block and loading in the AI mask for the global data
        mask = load_mask(model = args.ra_model, path = args.dataset)
        
        et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.ra_model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
        lat = et['lat']; lon = et['lon']

        # Remove the unused data
        del et
        gc.collect()
    
        # Collect the spatial size of the data
        I, J = lat.shape
            
        if args.globe:
            I_data = 201
            J_data = int(IJ/I_data)
        else:
            I_data = I
            J_data = J

        # Collect latitude and longitude values with the same dimensions as the dataset without the sea grid points (for data merging)
        if args.globe:
            with open("%s/%s"%(dataset_dir, args.input_data_fname), "rb") as fp:
                _ = pickle.load(fp)
                lat_sub = pickle.load(fp)
                lon_sub = pickle.load(fp)
                lat_sub = lat_sub.reshape(I_data, J_data, order = 'F')
                lon_sub = lon_sub.reshape(I_data, J_data, order = 'F')
        else:
            lat_sub = lat
            lon_sub = lon
        
        print('Merging results results...')
        results = []

        for m, method in enumerate(methods):
            result_method = merge_results(args, method, lat_sub, lon_sub, Nfolds, Nvar, T, I_data, J_data, data_in = data_in, data_out = data_out[m,:,:,:])
            
            results.append(result_method)
            gc.collect() # Clear out any large variables discarded from the result_method function

        del data_out
        gc.collect()
        # Note here that results[0] = christian; results[1] = nogeura; results[2] = pendergrass; results[3] = liu; results[4] = otkin

        # Obtain the latitude and longitude for metrics
        #lat = results[0]['lat']; lon = results[0]['lon']


        # Load the true labels
        print('Loading true labels...')

        true_fd = []
        # for method in methods:
        #     fd = load_nc('fd', '%s.%s.pentad.nc'%(method, args.ra_model), path = '%s/FD_Data/'%dataset_dir)
        #     fd = collect_grow_seasons(fd['fd'], fd['ymd'], fd['lat'][:,0])
        #     true_fd.append(fd)
        # del fd
        # gc.collect()

        ### This method is better (predictions and outputs have similiar shapes)
        with open("%s/%s"%(dataset_dir, args.output_data_fname), "rb") as fp:
           fd = pickle.load(fp)
           Nfold  = fd.shape[-1]
           fd = np.concatenate([fd[:,:,:,fold] for fold in range(Nfold)], axis = 1)
           fd[np.isnan(fd)] = 0
           for m, method in enumerate(methods):
               # Make output labels conform to the same grid
               true_fd.append(fd[m,:,:].reshape(T*Nfolds, I_data, J_data, order = 'F'))
                
                
        fd = load_nc('fd', '%s.%s.pentad.nc'%(args.method, args.ra_model), path = '%s/FD_Data/'%dataset_dir)
        # Collect the dates
        dates = fd['ymd']

        # Get the years and months
        years = np.array([date.year for date in dates])
        months = np.array([date.month for date in dates])

        # Subset the date information to the "growing seaon"
        if args.globe:
            ind = np.where((months >= 4) & (months <= 10) & (years < years[-1]))[0]
        else:
            ind = np.where((months >= 4) & (months <= 10))[0]
        dates_grow = dates[ind]
        years_grow = years[ind]
        months_grow = months[ind]

        # Get the unique years in the dataset
        years_unique = np.unique(years)

        # remove a large variable to clear space
        del result_method, fd
        gc.collect() # Clears deleted variables from memory 


        # Rename the methods to what they will be displayed as on figures
        methods = ['C23', 'N20', 'P20', 'L20', 'O21']

        # Plot the results of the metrics
        print('Plotting results...')
        for met, metric in enumerate(args.metrics):
            # Collect the metrics
            if args.globe:
                # To plot global data correctly, the sea grid points must be re-added

                # Reshape the mask
                mask2d = mask.reshape(I*J, order = 'F')

                # Initialize lists
                metrics_all = []
                metrics_valid = []
                metrics_test = []

                for m in range(len(methods)):
                    # Initialize the full map
                    full_map_all = np.ones((I*J)) * np.nan
                    full_map_valid = np.ones((I*J)) * np.nan
                    full_map_test = np.ones((I*J)) * np.nan

                    # Reshape the space shape into one axis
                    tmp_all = results[m]['all_eval_map'][:,:,met].reshape(IJ, order = 'F')
                    tmp_valid = results[m]['valid_eval_map'][:,:,met].reshape(IJ, order = 'F')
                    tmp_test = results[m]['test_eval_map'][:,:,met].reshape(IJ, order = 'F')
                    n = 0
                    for ij in range(I*J):
                        # Add an entry and increment n for the land points only (sea points in the full maps are left as NaN)
                        if mask2d[ij] == 1:
                            full_map_all[ij]   = tmp_all[n]
                            full_map_valid[ij] = tmp_valid[n]
                            full_map_test[ij]  = tmp_test[n]
                            n = n + 1

                    # Append the data to the lists
                    metrics_all.append(full_map_all.reshape(I, J, order = 'F'))
                    metrics_valid.append(full_map_valid.reshape(I, J, order = 'F'))
                    metrics_test.append(full_map_test.reshape(I, J, order = 'F'))

                # Remove the excessive, and potentially large, datasets
                del full_map_all, full_map_valid, full_map_test, tmp_all, tmp_valid, tmp_test
                gc.collect()
            else:
                metrics_all = [results[m]['all_eval_map'][:,:,met] for m in range(len(methods))]
                metrics_valid = [results[m]['valid_eval_map'][:,:,met] for m in range(len(methods))]
                metrics_test = [results[m]['test_eval_map'][:,:,met] for m in range(len(methods))]

            if (metric == 'mse') | (metric == 'mae'):
                cmin = 0; cmax = 0.5; cint = 0.005
            else:
                cmin = 0; cmax = 1; cint = 0.01

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
                               
            for m, method in enumerate(methods):
                print('Overall metric score, %s, for %s is %4.2f'%(metric, method, np.nanmean(metrics_test[m])))

            # Remove variables at the end to clear space
            del metrics_all, metrics_valid, metrics_test
            gc.collect() # Clears deleted variables from memory

        # Metrics to display on the compound figure
        metrics = ['accuracy', 'precision', 'recall', 'auc']

        # Place the metrics in a single variable
        data_metric = []
        for metric in metrics:
            ind = np.where(metric == np.array(args.metrics))[0]
            if args.globe:
                # Reshape the mask
                mask2d = mask.reshape(I*J, order = 'F')
                
                full_map_test = np.ones((I*J)) * np.nan

                tmp_test = []
                for m in range(len(methods)):
                    tmp = results[m]['test_eval_map'][:,:,ind[0]].reshape(IJ, order = 'F')
                    n = 0
                    for ij in range(I*J):
                        # Add an entry and increment n for the land points only (sea points in the full maps are left as NaN)
                        if mask2d[ij] == 1:
                            full_map_test[ij] = tmp[n]
                            n = n + 1

                    tmp_test.append(full_map_test.reshape(I, J, order = 'F'))

                data_metric.append(tmp_test)

                del full_map_test, tmp, tmp_test
                gc.collect()
                    
            else:
                data_metric.append([results[m]['test_eval_map'][:,:,ind[0]] for m in range(len(methods))])

        # Make the compound figure (only needed with test set)
        display_metric_map_new(data_metric, lat, lon, methods, metrics, 0, 1, 0.01, args.ml_model, 
                               args.label, dataset = 'test', globe = args.globe, path = dataset_dir)


        # Prepare a plot of the metrics variation in time (test set only); this is plotted for each year to be equivalent to variation in each rotation
        data_ts = np.ones((int(years_unique.size), len(methods), len(metrics)))

        for y, year in enumerate(years_unique):
            ind = np.where(year == years_grow)[0]
            print(ind)

            # Get the test predictions
            pred = [results[m]['test_predict'][ind,:,:] for m in range(len(methods))]
            for m in range(len(methods)):
                pred[m][np.isnan(pred[m])] = 0
            for n, metric in enumerate(metrics):
                # Calculate for each metric
                if metric == 'accuracy':
                    data_ts[y,:,n] = np.array([sklearn.metrics.accuracy_score(true_fd[m][ind,:,:].flatten(), pred[m].flatten()) for m in range(len(methods))])
                elif metric == 'precision':
                    data_ts[y,:,n] = np.array([sklearn.metrics.precision_score(true_fd[m][ind,:,:].flatten(), pred[m].flatten()) for m in range(len(methods))])
                elif metric == 'recall':
                    data_ts[y,:,n] = np.array([sklearn.metrics.recall_score(true_fd[m][ind,:,:].flatten(), pred[m].flatten()) for m in range(len(methods))])
                elif metric == 'auc':
                    # Test for FD predictions
                    valid_test = [(np.nansum(pred[m] < 0.5) == 0) | (np.nansum(pred[m] < 0.5) == pred[m].size) for m in range(len(methods))]
                    
                    data_ts[y,:,n] = np.array([np.nan if valid_test[m] else sklearn.metrics.roc_auc_score(true_fd[m][ind,:,:].flatten(), pred[m].flatten()) for m in range(len(methods))])
                    
        pred = [results[m]['test_predict'][:,:,:] for m in range(len(methods))]
        for m in range(len(methods)):
            pred[m][np.isnan(pred[m])] = 0
            valid_test = (np.nansum(pred[m] < 0.5) == 0) | (np.nansum(pred[m] < 0.5) == pred[m].size)
            score = np.nan if valid_test else sklearn.metrics.roc_auc_score(true_fd[m][:,:,:].flatten(), pred[m].flatten())
            print('Overall Test AUC for %s is: %4.2f'%(methods[m], score))

        # Create the time series plot
        display_metrics_in_time(data_ts, methods, years_unique, metrics, path = dataset_dir)
        
        for m, method in enumerate(methods):
            for n, metric in enumerate(metrics):
                print('Overall metric score, %s, for %s is %4.2f'%(metric, method, np.nanmean(data_ts[:,m,n])))

        # Remove variables at the end to clear space
        del data_metric, data_ts, pred, score 
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

            # Make a set of ROC Curves on only 1 plot (only needed for test data)
            display_roc_curves_new(tpr_test, fpr_test, tpr_var_test, fpr_var_test, 
                             methods, args.ra_model, args.label, dataset = 'test', path = dataset_dir)

            # Remove variables at the end to clear space
            del tpr_all, tpr_valid, tpr_test, tpr_var_all, tpr_var_valid, tpr_var_test
            del fpr_all, fpr_valid, fpr_test, fpr_var_all, fpr_var_valid, fpr_var_test
            gc.collect() # Clears deleted variables from memory 



        # Plot the feature importance?
        if args.interpret:
            features = [r'T', r'ET', r'$\Delta$ET', r'PET', r'$\Delta$PET', r'P', r'SM', r'$\Delta$SM']
            NFeature = results[0]['feature_import'].shape[0]

            print('Making overall feature importance barplot...')
            # Feature importance plots for permutation importance
            # Note there are 42 rotations/samples
            if (args.ml_model == 'rf') | (args.ml_model == 'ada'):
                fi = [results[m]['feature_import_pi'] for m in range(len(methods))]
                fi_var = [results[m]['feature_import_pi_var'] for m in range(len(methods))]
                
                display_feature_importance(fi, fi_var, 42, features, methods, args.ml_model, '%s_feature_importance_pi'%args.label, path = dataset_dir)

                # Feature importance for GINI importance
                if args.feature_importance:
                    fi = [results[m]['feature_import_gini'] for m in range(len(methods))]
                    fi_var = [results[m]['feature_import_gini_var'] for m in range(len(methods))]
                    
                    display_feature_importance(fi, fi_var, 42, features, methods, args.ml_model, '%s_feature_importance_gini'%args.label, path = dataset_dir)

            fi = [results[m]['feature_import'] for m in range(len(methods))]
            fi_var = [results[m]['feature_import_var'] for m in range(len(methods))]
            print(fi)
            print(fi_var)


            attribution = [np.nanmean(results[m]['attributions'], axis = 0) for m in range(len(methods))]
            print(attribution)

            # Create a barplot of the overall feature importance and attribution
            #display_feature_importance(fi, fi_var, features, methods, args.ra_model, '%s_feature_importance'%args.label, path = dataset_dir)
            display_feature_importance(fi, fi_var, 42, features, methods, args.ml_model, '%s_feature_importance'%args.label, path = dataset_dir)

            display_feature_importance(attribution, None, None, features, methods, args.ml_model, '%s_attribution'%args.label, path = dataset_dir)

            # Remove variables at the end to clear space
            del fi, fi_var, attribution
            gc.collect() # Clears deleted variables from memory 


        # Plot the learning curve?
        ##### ADD SPAGGATTI LEARNING CURVES
        if args.keras:
           for m, method in enumerate(methods):
               display_learning_curve(results[m]['history'], results[m]['history_var'], ['loss', 'categorical_accuracy'], 
                                      False, args.ra_model, method, path = dataset_dir)


        # Make predictions?
        if (args.climatology_plot | args.time_series | args.case_studies | args.confusion_matrix_plots):

            # Collect the predicted labels (full maps)
            if args.globe:
                # Reshape the mask
                mask2d = mask.reshape(I*J, order = 'F')
                
                pred_all = []
                pred_valid = []
                pred_test = []
                fd_label = []

                for m in range(len(methods)):
                    # Initialize the full map
                    full_map_all = np.ones((T*Nfolds, I*J), dtype = np.float32) * np.nan
                    full_map_valid = np.ones((T*Nfolds, I*J), dtype = np.float32) * np.nan
                    full_map_test = np.ones((T*Nfolds, I*J), dtype = np.float32) * np.nan
                    full_true_fd = np.ones((T*Nfolds, I*J), dtype = np.float32) * np.nan

                    # Reshape the space shape into one axis
                    tmp_all = results[m]['all_predict'][:,:,:].reshape(T*Nfolds, IJ, order = 'F')
                    tmp_valid = results[m]['valid_predict'][:,:,:].reshape(T*Nfolds, IJ, order = 'F')
                    tmp_test = results[m]['test_predict'][:,:,:].reshape(T*Nfolds, IJ, order = 'F')

                    tmp_true = true_fd[m][:,:,:].reshape(T*Nfolds, IJ, order = 'F')
                    n = 0
                    for ij in range(I*J):
                        # Add an entry and increment n for the land points only (sea points in the full maps are left as NaN)
                        if mask2d[ij] == 1:
                            full_map_all[:,ij]   = tmp_all[:,n]
                            full_map_valid[:,ij] = tmp_valid[:,n]
                            full_map_test[:,ij]  = tmp_test[:,n]

                            full_true_fd[:,ij] = tmp_true[:,n]
                            n = n + 1

                    # Append the data to the lists
                    pred_all.append(full_map_all.reshape(T*Nfolds, I, J, order = 'F'))
                    pred_valid.append(full_map_valid.reshape(T*Nfolds, I, J, order = 'F'))
                    pred_test.append(full_map_test.reshape(T*Nfolds, I, J, order = 'F'))

                    fd_label.append(full_true_fd.reshape(T*Nfolds, I, J, order = 'F'))

                true_fd = fd_label
                # Remove the excessive, and potentially large, datasets
                del full_map_all, full_map_valid, full_map_test, full_true_fd, tmp_all, tmp_valid, tmp_test, tmp_true, fd_label
                gc.collect()
                    
            else:
                pred_all = [results[m]['all_predict'] for m in range(len(methods))]
                pred_valid = [results[m]['valid_predict'] for m in range(len(methods))]
                pred_test = [results[m]['test_predict'] for m in range(len(methods))]

            # Compound plots (these are made with test set and true labels only

            # Make compound case study maps inlcuding feature attribution
            print('Making compound case study plots...')
            years = np.array([1979 + rot for rot in range(42)])
            variables = ['Temperature Anomalies', 'Evapotranspiration Anomalies', 'Change in Evapotranspiration Anomalies', 
                         'Potential Evaporation Anomalies', 'Change in Potential Evaporation Anomalies', 'Precipitaiton Anomalies', 
                         'Soil Moisture Anomalies', 'Change in Soil Moisture Anomalies']
            colors = ['r', 'g', 'g', 'orange', 'orange', 'b', 'k', 'k']
            # Make plots of raw variables
            for year in args.case_study_years:
                ind = np.where(year == years)[0]
                raw_variable = data_in[:,:,:,ind[0]]
                # Standardize the raw variable
                for ij in range(IJ):
                    scaler = StandardScaler()
            
                    tmp = scaler.fit_transform(raw_variable[:,:,ij].T)
                    raw_variable[:,:,ij]  = tmp.T
                    
                #raw_variable = raw_variable.reshape(Nvar, T*IJ, order = 'F')
                    
                for m, var in enumerate(range(Nvar)):
                    # Subset the variable
                    raw_var = raw_variable[m,:,:]
                    raw_var = get_domain(raw_var, lat_sub, lon_sub, year, globe = args.globe, sea_points = True)
                    raw_var = np.nanmean(raw_var, axis = -1)
                    
                    # Create the plot of the variable
                    fig, ax = plt.subplots(figsize = [12, 8])
                    
                    # Set the title
                    ax.set_title(variables[m], fontsize = 22)
                    
                    # Make the plots color = 'r', linestyle = '-', linewidth = 1, label = 'True values'
                    ax.plot(dates_grow[:43], raw_var, color = colors[m], linestyle = '-', linewidth = 1.5)
                    
                    # Set the labels
                    ax.set_xlabel('Time', fontsize = 22)
                    
                    # Set the ticks
                    ax.xaxis.set_major_formatter(DateFormatter('%b'))
                    
                    # Set the tick sizes
                    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                        i.set_size(22)
                        
                    # Save the figure
                    filename = '%s_%s.png'%(variables[m], year)
                    plt.savefig('%s/%s'%(dataset_dir, filename), bbox_inches = 'tight')
                    plt.show(block = False)

            
            if args.interpret:# & np.invert(args.globe):
                features = [r'T', r'ET', r'$\Delta$ET', r'PET', r'$\Delta$PET', r'P', r'SM', r'$\Delta$SM']
                
                #pred = [results[m]['test_predict'] for m in range(len(methods))]
                attributions = [results[m]['attributions_cs'] for m in range(len(methods))]
                
                display_case_study_maps_full(true_fd, pred_test, attributions, features, lon, lat, dates_grow, args.case_study_years, 
                                             methods = methods, label = args.label, dataset = args.ra_model, 
                                             globe = args.globe, path = dataset_dir, grow_season = True)

                gc.collect() # Clears deleted variables from memory

            else:
                #pred = [results[m]['test_predict'] for m in range(len(methods))]
                total_land = np.nansum(mask)

                #T, I, J = pred[m].shape

				# Instead of feature attribution, plot FD coverage for the case study
                land_coverage = []
                for m, method in enumerate(methods):
                    # Make variable that holds the FD Coverage for all years
                    ind = np.where(args.case_study_years[0] == years_grow)[0]
                    all_coverage = np.ones((len(args.case_study_years), ind.size, 2)) * np.nan
                    
                    # Determine the annual mean/std areal coverage of FD
                    for y, year in enumerate(args.case_study_years):
                        ind = np.where(year == years_grow)[0]
                        T_tmp = ind.size
                
                        # Subset the data to the specific domain
                        tmp_pred = pred_test[m][ind,:,:].reshape(T_tmp, I*J, order = 'F')
                        tmp_pred = get_domain(tmp_pred, lat, lon, year, globe = args.globe, sea_points = True)
                        tmp_pred = np.nansum(tmp_pred, axis = -1)

                        # Determine the true areal coverage
                        tmp_true = true_fd[m][ind,:,:].reshape(T_tmp, I*J, order = 'F')
                        tmp_true = get_domain(tmp_true, lat, lon, year, globe = args.globe, sea_points = True)
                        tmp_true = np.nansum(tmp_true, axis = -1)
                        
                        pred_area = np.array(tmp_pred/total_land*100)
                        true_area = np.array(tmp_true/total_land*100)

                        all_coverage[y,:,0] = true_area
                        all_coverage[y,:,1] = pred_area

                    land_coverage.append(all_coverage)


                display_case_study_maps_full(true_fd, pred_test, land_coverage, ['True Label', 'Predictions'], lon, lat, dates_grow, args.case_study_years, 
                                             methods = methods, label = args.label, dataset = args.ra_model, 
                                             globe = args.globe, path = dataset_dir, grow_season = True)
                
                #del pred
                #gc.collect() # Clears deleted variables from memory
            
            # Plot the confusion matrices?
            if args.confusion_matrix_plots & np.invert(args.globe):
                print('Making confusion matrix plot...')
                labels = [0, 1]
                label_names = ['No FD', 'FD']

                pred = [results[m]['test_predict'] for m in range(len(methods))]

                display_confusion_matrix(true_fd, pred, labels, label_names, methods, savename = '%s_confusion_matrix.png'%args.label)

                del pred
                gc.collect() # Clears deleted variables from memory
                
            # Plot the predicted climatology map?
            if args.climatology_plot:
            
                print('Plotting the predicted climatologies...')
                display_fd_climatology(pred_all, lat, lon, dates_grow, mask, methods, globe = args.globe,
                                        model = '%s_%s'%(args.label, args.ra_model), path = dataset_dir, grow_season = True)

                    
                # Plot the climatology map with only the validation predictions (to see how the model generalizes to data it has not seen)
                display_fd_climatology(pred_valid, lat, lon, dates_grow, mask, methods, globe = args.globe,
                                        model = '%s_%s_valid_set'%(args.label, args.ra_model), path = dataset_dir, grow_season = True)
                    
                    
                # Plot the climatology map with only the test predictions (to see how the model generalizes to data it has not seen)
                display_fd_climatology(pred_test, lat, lon, dates_grow, mask, methods, globe = args.globe,
                                        model = '%s_%s_test_set'%(args.label, args.ra_model), path = dataset_dir, grow_season = True)
                                        
                                           
            # Make the predictions and plot the results for each method individually
            for m, method in enumerate(methods):

                # Plot the threat scores?
                if args.confusion_matrix_plots:
                    print('Plotting confusion matrix skill scores for the %s method...'%method)
                    mask = load_mask(model = args.ra_model)

                    # Plot the threat scores with the full predictions
                    display_threat_score(true_fd[m], pred_all[m], lat, lon, dates_grow, mask, 
                                         model = args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far(true_fd[m], pred_all[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod(true_fd[m], pred_all[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s'%(args.label, method), globe = args.globe, path = dataset_dir)


                    
                    # Plot the threat scores with only the validation predictions (to see how the model generalizes to data it has not seen)
                    display_threat_score(true_fd[m], pred_valid[m], lat, lon, dates_grow, mask, 
                                         model = args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far(true_fd[m], pred_valid[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod(true_fd[m], pred_valid[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s_valid_set'%(args.label, method), globe = args.globe, path = dataset_dir)
                    
                    
                    
                    # Plot the threat scores with only the test predictions (to see how the model generalizes to data it has not seen)
                    display_threat_score(true_fd[m], pred_test[m], lat, lon, dates_grow, mask, 
                                         model = args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_far(true_fd[m], pred_test[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)

                    display_pod(true_fd[m], pred_test[m], lat, lon, dates_grow, 
                                model =  args.ra_model, label = '%s_%s_test_set'%(args.label, method), globe = args.globe, path = dataset_dir)


                    # Plot a map of confusion matrix values for the test predictions (agreement, false positives, and false negatives)
                    display_confusion_matrix_maps(true_fd[m], pred_test[m], lat, lon, method, globe = args.globe, 
                                                  path = dataset_dir, savename = '%s_%s_confusion_matrix_maps.png'%(args.label, methods[m]))

                    # Create a map of the composite difference between the true label and test predictions
                    display_difference_map(true_fd[m], pred_test[m], lat, lon, method, args.label, globe = args.globe, path = dataset_dir)


                # Plot the predicted time series (with true labels)?
                if args.time_series:
                    print('Calculating areal coverage for the %s method...'%method)
                    # Examine predicted time series
                    T, I, J = pred_all[m].shape

                    # Determine the areal coverage for the time series
                    tmp_pred = np.nansum(pred_all[m].reshape(T, I*J), axis = -1)
                    tmp_pred_valid = np.nansum(pred_valid[m].reshape(T, I*J), axis = -1)
                    tmp_pred_test = np.nansum(pred_test[m].reshape(T, I*J), axis = -1)

                    pred_area = []
                    pred_area_var = []

                    pred_area_valid = []
                    pred_area_var_valid = []
                    
                    pred_area_test = []
                    pred_area_var_test = []


                    # Determine the true areal coverage
                    tmp_true = np.nansum(true_fd[m].reshape(T, I*J), axis = -1)
                    total_land = np.nansum(mask)

                    true_area = []
                    true_area_var = []

                    # Determine the annual mean/std areal coverage of FD
                    for year in np.unique(years_grow):
                        ind = np.where(year == years_grow)[0]

                        pred_area.append(np.nanmean(tmp_pred[ind]/total_land))
                        pred_area_var.append(np.nanstd(tmp_pred[ind]/total_land))

                        pred_area_valid.append(np.nanmean(tmp_pred_valid[ind]/total_land))
                        pred_area_var_valid.append(np.nanstd(tmp_pred_valid[ind]/total_land))
                        
                        pred_area_test.append(np.nanmean(tmp_pred_test[ind]/total_land))
                        pred_area_var_test.append(np.nanstd(tmp_pred_test[ind]/total_land))

                        true_area.append(np.nanmean(tmp_true[ind]/total_land))
                        true_area_var.append(np.nanstd(tmp_true[ind]/total_land))

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
                    display_time_series(true_area*100, pred_area*100, true_area_var*100, pred_area_var*100, dates_grow[::43], 
                                        r'Areal Coverage (%)', args.ra_model, '%s_%s'%(args.label, method), path = dataset_dir)

                    # Display the time series with only the validation predictions (to see how the model generalizes to data it has not seen)
                    display_time_series(true_area*100, pred_area_valid*100, true_area_var*100, pred_area_var_valid*100, dates_grow[::43], 
                                        r'Areal Coverage (%)', args.ra_model, '%s_%s_valid_set'%(args.label, method), path = dataset_dir)
                    
                    # Display the time series with only the test predictions (to see how the model generalizes to data it has not seen)
                    display_time_series(true_area*100, pred_area_test*100, true_area_var*100, pred_area_var_test*100, dates_grow[::43], 
                                        r'Areal Coverage (%)', args.ra_model, '%s_%s_test_set'%(args.label, method), path = dataset_dir)

                
                # Plot a set of case studies?
                if args.case_studies:
                    print('Plotting case studies for the %s method...'%method)

                    # Plot the case studies for the predicted labels
                    display_case_study_maps(pred_all[m], lon, lat, dates_grow, args.case_study_years, 
                                            method = method, label = args.label, dataset = args.ra_model, 
                                            globe = args.globe, path = dataset_dir, grow_season = True, pred_type = 'pred')

                    # Plot the case studies for the true labels
                    display_case_study_maps(true_fd[m], lon, lat, dates_grow, args.case_study_years, 
                                            method = method, label = args.label, dataset = args.ra_model, 
                                            globe = args.globe, path = dataset_dir, grow_season = True, pred_type = 'true')

                    
                    # Repeat the predicted case studes with predicted labels using validation sets (to see how the model generalizes to data it has not seen)
                    display_case_study_maps(pred_valid[m], lon, lat, dates_grow, args.case_study_years, 
                                            method = method, label = '%s_valid_set'%args.label, dataset = args.ra_model, 
                                            globe = args.globe, path = dataset_dir, grow_season = True, pred_type = 'pred')
                    
                    # Repeat the predicted case studes with predicted labels using test sets only (to see how the model generalizes to data it has not seen)
                    display_case_study_maps(pred_test[m], lon, lat, dates_grow, args.case_study_years, 
                                            method = method, label = '%s_test_set'%args.label, dataset = args.ra_model, 
                                            globe = args.globe, path = dataset_dir, grow_season = True, pred_type = 'pred')




    print('Done')        
                    

