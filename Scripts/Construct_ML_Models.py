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
    Model load functions have not been tested
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


#%%
##############################################
# Functions to build the ML models

def build_sklearn_model(args):
    '''
    Build a ML model from the sklearn package
    
    Inputs:
    :param args: Argparse arguments
    '''
    
    if (args.ml_model.lower() == 'rf') | (args.ml_model.lower() == 'random_forest'):
        model = build_rf_model(args)
        
    elif (args.ml_model.lower() == 'svm') | (args.ml_model.lower() == 'support_vector_machine'):
        model = build_svm_model(args)
        
    elif (args.ml_model.lower() == 'ada') | (args.ml_model.lower() == 'boosting') | (args.ml_model.lower() == 'ada_boosting'):
        model = build_adaboost_model(args)
    
    return model


def build_keras_model():
    '''
    Build a ML model (a nueral network) from the keras package
    '''
    
    model = 10
    
    return model


#%%
##############################################
# Functions to generate file names

def generate_model_fname_build(model, ml_model, method, rotation, lat_labels, lon_labels):
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
    

    
#%%
##############################################
# Functions to load the ML models

def load_single_model(fname, keras):
    '''
    Load a single machine learning model
    
    Inputs:
    :param fname: The filename of the model being loaded
    :param kera: Boolean indicating whether a keras model is being loaded
    
    Outputs:
    :param model: The machine learning model that was loaded
    '''
    
    # If a keras model is being loaded, use the build-in load function. Else load a pickle file
    if keras:
        pass ##### Commands to load a Keras model go here
    else:
        with open('%s.pkl'%fname, 'rb') as fn:
            model = pickle.load(fn)
            
    return model


def load_all_models(keras, ml_model, model, method, rotation = 0, path = './Data/narr/christian_models'):
    '''
    Load all ML models associated with a single FD method and a single dataset
    
    Inputs:
    :param keras: A boolean indicating whether a keras ML model is being loaded
    :param ml_model: The name of the ML model being loaded
    :param ra_model: The name of the reanalysis model used to train the ML model
    :param method: The name of the FD identification method the ML model learned
    :param rotations: List of rotations over which to load the data
    :param path: Path to the ML models that will be loaded
    
    Outputs:
    :param models: A list of all machine learning models loaded for a single FD identification method and trained on a single reanalysis
    '''
    
    # Initialize the models
    models = []
    
    # Load lat/lon labels
    with open('%s/lat_lon_labels.pkl'%(path), 'rb') as fn:
        lat_labels = pickle.load(fn)
        lon_labels = pickle.load(fn)
    
    I = len(lat_labels)
    J = len(lon_labels)
    
    # Load all models for each combination of lat and lon labels
    for n in range(len(lat_labels)):
            
        # Generate the model filename
        model_fbase = generate_model_fname_build(model, ml_model, method, rotation, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
        model_fname = '%s/%s/%s/%s'%(path, ml_model, method, model_fbase)
        print(model_fname)
        
        model = load_single_model(model_fname, keras)
        
        models.append(model)
    
            
    return models

# Function to make predictions for all models in a rotation (and average them together/take the standard deviation)
def make_predictions(data, lat, lon, probabilities, threshold, keras, ml_model, ra_model, 
                     method, rotations, label, path = './Data/narr/christian_models'):
    '''
    Function designed make predictions of FD for a full dataset for a given ML model. Predictions are the average over all rotations and the standard deviation.
    
    Inputs:
    :param data: Data used to make FD predictions. Must be in a NVar x time x space format
    :param lat: The latitude coordinates corresponding to data
    :param lon: The longitude coordinates corresponding to data
    :param probabilities: Boolean indicating whether to use return the average probabilistic predictions (true), or average yes/no predictions (false)
    :param theshold: The probability threshold above which FD is said to occur
    :param keras: A boolean indicating whether a keras ML model is being used
    :param ml_model: The name of the ML model being used
    :param ra_model: The name of the reanalysis model used to train the ML model
    :param method: The name of the FD identification method the ML model learned
    :param rotations: List of rotations over which to load the data
    :param label: The experiment label of the ML models
    :param path: Path to the ML models that will be loaded
    
    Outputs:
    :param pred: The mean FD predictions of FD
    :param pred_var: The variation in FD probability predictions across all rotations
    '''
    
        
    # Reshape lat and lon into 1D arrays
    print('Initializing some values')
    I, J = lat.shape
    lat1d = lat.reshape(I*J, order = 'F')
    lon1d = lon.reshape(I*J, order = 'F')
    
    # Load in lat/lon labels
    lat_labels = np.arange(-90, 90+5, 5)
    lon_labels = np.arange(-180, 180+5, 5)
    
    I_lab = len(lat_labels)
    J_lab = len(lon_labels)
    
    # Remove NaNs?
    if np.invert(keras):
        data[np.isnan(data)] = -995
    
    # Initialize the prediction variables
    NVar, T, IJ = data.shape
    
    # data_reshaped = data.reshape(NVar, T, I*J, order = 'F')
    
    pred = np.ones((T, I*J)) * np.nan
    pred_var = np.ones((T, I*J)) * np.nan
    
    # Split the dataset into regions
    print('Splitting data into regions')
    data_split = []
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
            
            data_split.append(data[:,:,ind])
            
    print('There are %d regions.'%len(data_split))
            
            
    # Begin making predictions
    print('Loading models and making predictions')
    for n in range(len(data_split)):
        ind = np.where( ((lat1d >= lat_lab[n]) & (lat1d <= lat_lab[n]+5)) & ((lon1d >= lon_lab[n]) & (lon1d <= lon_lab[n]+5)) )[0]
        
        pred_tmp = []
        for rot in rotations:
            # Generate the model filename
            model_fbase = generate_model_fname_build(ra_model, label, method, rot, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
            model_fname = '%s/%s/%s/%s/%s'%(path, ra_model, ml_model, method, model_fbase)
            if keras:
                test_name = model_fname
            else:
                test_fname = '%s.pkl'%model_fname


            # Check if model exists (it will not if there are no land points)
            if np.invert(os.path.exists(test_fname)):
                continue
            
            model = load_single_model(model_fname, keras)
            
            # Code to make prediction depends on whether a keras model is used
            if keras:
                pred_tmp.append(model.predict(data_split[n]))
            else:
                NVar, T, IJ_tmp = data_split[n].shape
                
                tmp_data = data_split[n].reshape(NVar, T*IJ_tmp, order = 'F')
                if (ml_model.lower() == 'svm') | (ml_model.lower() == 'support_vector_machine'):
                    # Note SVMs do not have a predict_proba option
                    tmp = model.predict(tmp_data.T)
                    pred_tmp.append(tmp.reshape(T, IJ_tmp, order = 'F'))
                
                else:
                    tmp = model.predict_proba(tmp_data.T)

                    # Check if the model only predicts 0s
                    only_zeros = tmp.shape[1] <= 1
                    if only_zeros:
                        pred_tmp.append(np.zeros((T,len(ind))))
                    else:
                        pred_tmp.append(tmp[:,1].reshape(T, IJ_tmp, order = 'F'))
                    
        # For sea values, pred_tmp will be empty. Continue to the next region if this happens
        if len(pred_tmp) < 1:
            continue
                    
        # Take the average of the probabilistic predictions across all rotations
        pred[:,ind] = np.nanmean(np.stack(pred_tmp, axis = -1), axis = -1)

        # Take the standard deviation of the probabilistic predictions across all rotations
        pred_var[:,ind] = np.nanstd(np.stack(pred_tmp, axis = -1), axis = -1)
            
    # Turn the mean probabilistic predictions into true/false?
    if np.invert(probabilities):
        pred = np.where(pred >= threshold, 1, pred)
        pred = np.where(pred < threshold, 0, pred) # Performing this twice preserves NaN values as not available
        
    # Turn the predictions into 3D data
    pred = pred.reshape(T, I, J, order = 'F')
    pred_var = pred_var.reshape(T, I, J, order = 'F')
    
    return pred, pred_var
                    



#%%
##############################################

# Function to make a RF model
def build_rf_model(args):
    '''
    Construct a random forest model using sklearn
    
    Inputs:
    :param args: Argparse arguments
    '''
    
    # args.tree_max_features is a str. Turn it into a None object if None is specified
    if args.tree_max_features == 'None':
        args.tree_max_features = None 
        
    # Determine class weights
    if args.class_weight != None:
        weights = {0:1, 1:args.class_weight}
    else:
        weights = 'balanced'
    
    # Create the model
    model = ensemble.RandomForestClassifier(n_estimators = args.n_trees, 
                                            criterion = args.tree_criterion,
                                            max_depth = args.tree_depth,
                                            max_features = args.tree_max_features,
                                            bootstrap = args.feature_importance,
                                            oob_score = args.feature_importance,
                                            class_weight = weights,
                                            verbose = args.verbose >= 1)
                                            #n_jobs = )
    
    return model


#%%
##############################################

# Function to make a SVM model
def build_svm_model(args):
    '''
    Construct a linear support vector classifier using sklearn
    
    :Inputs:
    :param args: Argparse arguments
    '''
    
    # Determine class weights
    if args.class_weight != None:
        weights = {0:1, 1:args.class_weight}
    else:
        weights = 'balanced'
      
    # Build the model
    if args.svm_kernel == 'linear':
        model = svm.LinearSVC(penalty = args.svm_regularizer,
                              loss = args.svm_loss,
                              dual = False, # Perferred when n_samples > n_features. n_features = 8, and n_samples >= 43
                              tol = args.svm_stopping_err,
                              C = args.svm_regularizer_value,
                              fit_intercept = args.svm_intercept,
                              intercept_scaling = args.svm_intercept_scale,
                              class_weight = weights,
                              verbose = args.verbose >= 1,
                              max_iter = args.svm_max_iter)
        
    else:
        model = svm.SVC(C = args.svm_regularizer_value,
                        kernel = args.svm_kernel,
                        tol = args.svm_stopping_err,
                        class_weight = weights,
                        verbose = args.verbose >= 1,
                        max_iter = args.svm_max_iter)
    
    return model


#%%
##############################################

# Function to make an Ada Boosted tree model
def build_adaboost_model(args):
    '''
    Construct a Ada boosted tree model using sklearn
    
    :Inputs:
    :param args: Argparse arguments
    '''
    
    # Determine class weights
    if args.class_weight != None:
        weights = {0:1, 1:args.class_weight}
    else:
        weights = 'balanced'
      
    # Build the base for the model
    base = tree.DecisionTreeClassifier(criterion = args.tree_criterion,
                                       max_depth = args.tree_depth,
                                       max_features = args.tree_max_features,
                                       class_weight = weights)
    
    # Build the model
    model = ensemble.AdaBoostClassifier(base_estimator = base,
                                        n_estimators = args.ada_n_estimators,
                                        learning_rate = args.ada_learning_rate)
    
    return model