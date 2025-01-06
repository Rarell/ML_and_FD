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

TO DO:
- Fill out the build_model() function
- Test build_attention_model() function more; ensure they work and output what is desired, if a decoder is needed, etc.
- Add NNs:
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
import tensorflow as tf
from tensorflow import keras
#import tensorflow_models as tfm
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

import xgboost as xgb

# Tensorflow 2.x way of doing things
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Reshape, Masking, Flatten, RepeatVector
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, SpatialDropout2D, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, SpatialDropout1D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
#from tensorflow_models.nlp import layers
#from tensorflow_models.nlp.layers import TransformerEncoderBlock, TransformerDecoderBlock
from tensorflow.keras.models import Sequential, Model


#%%
##############################################
# Functions to build the ML models

def build_sklearn_model(args):
    '''
    Build a ML model from the sklearn package
    
    Inputs:
    :param args: Argparse arguments
    
    Outputs:
    :param model: sklearn model (still needs to use .fit())
    '''
    
    # Determine which model to build
    if (args.ml_model.lower() == 'rf') | (args.ml_model.lower() == 'random_forest'):
        model = build_rf_model(args)
        
    elif (args.ml_model.lower() == 'svm') | (args.ml_model.lower() == 'support_vector_machine'):
        model = build_svm_model(args)
        
    elif (args.ml_model.lower() == 'ada') | (args.ml_model.lower() == 'boosting') | (args.ml_model.lower() == 'ada_boosting'):
        model = build_adaboost_model(args)
        
    elif args.ml_model.lower() == 'xgboost':
        model = build_xgboost_model(args)
    
    return model


def build_keras_model(args, shape = None):
    '''
    Build a ML model (a nueral network) from the keras package
    
    Inputs:
    :param args: Argparse arguments
    :param shape: Shape of input data
    
    Outputs:
    :param model: TensorFlow model (still needs to use .fit())
    '''
    
    # Determine which model to build
    if (args.ml_model.lower() == 'ann') | (args.ml_model.lower() == 'artificial_neural_network'):
        model = build_ann_model(args, shape)
        
    elif (args.ml_model.lower() == 'cnn') | (args.ml_model.lower() == 'convolutional_neural_network') | (args.ml_model.lower() == 'u-network') | (args.ml_model.lower() == 'autoencoder'):
        model = build_cnn_model(args, shape)
        
    elif (args.ml_model.lower() == 'cnn-rnn'):
        model = build_cnn_rnn_model(args, shape)
        
    elif (args.ml_model.lower() == 'rnn') | (args.ml_model.lower() == 'recurrent_neural_network'):
        model = build_rnn_model(args, shape)
        
    elif (args.ml_model.lower() == 'attention') | (args.ml_model.lower() == 'transformer'):
        model = build_attention_model(args, shape)
    
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

def load_single_model(fname, use_keras, use_xgb = False):
    '''
    Load a single machine learning model
    
    Inputs:
    :param fname: The filename of the model being loaded
    :param use_keras: Boolean indicating whether a keras model is being loaded
    :param use_xgb: Boolean indicating whether a XGBoost model is being loaded
    
    Outputs:
    :param model: The machine learning model that was loaded
    '''
    
    # If a keras model is being loaded, use the build-in load function. Else load a pickle file
    if use_keras:
        model = keras.models.load_model(fname)
    else:
        if use_xgb:
            model = xgb.XGBClassifier()
            model.load_model('%s.json'%fname)
        else:
            with open('%s.pkl'%fname, 'rb') as fn:
                model = pickle.load(fn)
            
    return model


#%%
##############################################

# Function to make a RF model
def build_rf_model(args):
    '''
    Construct a random forest model using sklearn
    
    Inputs:
    :param args: Argparse arguments
    
    Outputs:
    :param model: Random Forest Classifier model
    '''
    
    # args.tree_max_features is a str. Turn it into a None object if None is specified
    if args.tree_max_features == 'None':
        args.tree_max_features = None 
        
    # Determine class weights
    if np.invert(args.class_weight == None):
        #weights = [{0:1, 1:1}, {0:1, 1:args.class_weight}, {0:1, 1:0}]
        weights = {0:1, 1:args.class_weight, 2:0}
    else:
        weights = None
    
    # Create the model
    model = ensemble.RandomForestClassifier(n_estimators = args.n_trees, 
                                            criterion = args.tree_criterion,
                                            max_depth = args.tree_depth,
                                            max_features = args.tree_max_features,
                                            bootstrap = args.feature_importance,
                                            oob_score = args.feature_importance,
                                            class_weight = weights,
                                            verbose = args.verbose >= 2)
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
    
    Outputs:
    :param model: Support Vector Machine Classifier model
    '''
    
    # Determine class weights
    if args.class_weight != None:
        #weights = [{0:1, 1:1}, {0:1, 1:args.class_weight}, {0:1, 1:0}]
        weights = {0:1, 1:args.class_weight}
    else:
        weights = 'balanced'
      
    # Build the model
    if args.svm_kernel == 'linear':
        # Linear SVMS have an omptized method in the LinearSVC class
        model = svm.LinearSVC(penalty = args.svm_regularizer,
                              loss = args.svm_loss,
                              dual = False, # False is perferred when n_samples > n_features. n_features = 8, and n_samples >= 43
                              tol = args.svm_stopping_err,
                              C = args.svm_regularizer_value,
                              fit_intercept = args.svm_intercept,
                              intercept_scaling = args.svm_intercept_scale,
                              class_weight = weights,
                              verbose = args.verbose >= 2,
                              max_iter = args.svm_max_iter)
        
    else:
        model = svm.SVC(C = args.svm_regularizer_value,
                        kernel = args.svm_kernel,
                        tol = args.svm_stopping_err,
                        class_weight = weights,
                        verbose = args.verbose >= 2,
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
    
    Outputs:
    :param model: Ada boosted Decision Tree Classifier model
    '''
    
    # Determine class weights
    if args.class_weight != None:
        #weights = [{0:1, 1:1}, {0:1, 1:args.class_weight}, {0:1, 1:0}]
        weights = {0:1, 1:args.class_weight, 2:0}
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



#%%
##############################################

# Function to make an Ada Boosted tree model
def build_xgboost_model(args):
    '''
    Construct a XG Boosting tree model
    
    :Inputs:
    :param args: Argparse arguments
    
    Outputs:
    :param model: XG Boosted Tree Classifier model
    '''
    
    # Create the early stopper
    early_stopping = xgb.callback.EarlyStopping(rounds = args.patience, metric_name = 'mlogloss')

    # Build the model
    model = xgb.XGBClassifier(n_estimators = args.ada_n_estimators, 
                              max_depth = args.tree_depth, 
                              learning_rate = args.ada_learning_rate, 
                              reg_lambda = args.L2_regularization, 
                              verbosity = 2, 
                              #n_jobs = 2, 
                              missing = np.nan, 
                              importance_type = 'gain', 
                              eval_metric = ['mlogloss', 'merror', 'auc'], 
                              early_stopping_rounds = args.patience, 
                              callbacks = [early_stopping])
    
    return model
    

#%%
##############################################

# Function to make a ANN model
def build_ann_model(args, shape):
    '''
    Construct an artificial neural network (ANN) model using keras
    
    :Inputs:
    :param args: Argparse arguments
    :param shape: The shape of the training data (size/n_examples x map shape x n_variables)
    
    Outputs:
    :param model: Artificial/Densely connected NN model
    '''
    
                    
    # Define any possible regularization
    if args.L1_regularization is not None:
        regularizer = keras.regularizers.l1(args.L1_regularization)
    elif args.L2_regularization is not None:
        regularizer = keras.regularizers.l2(args.L2_regularization)
    else:
        regularizer = None # Define the regularizar for the model, but set to 0 to not use it
                    
    # Create the model
    model = Sequential()
    
    # Add the input layer
    model.add(InputLayer(input_shape = (shape[1],)))
    
    # Add a dropout layer to the input?
    if np.invert(args.dropout == None):
        model.add(Dropout(rate = args.dropout, name = 'input_dropout'))
           
    # Add the hidden layers
    for n, unit in enumerate(args.units):
        model.add(Dense(unit, use_bias = True, name = 'hidden%02d'%(n+1), activation = args.activation[n],
                       kernel_regularizer = regularizer))
                    
        # Add dropout layers?
        if np.invert(args.dropout == None):
            model.add(Dropout(rate = args.dropout, name = 'hidden%02d_input'%(n+1)))
    
    # Add a reshape layer so sample weights can work
    #model.add(Reshape((shape[1]*shape[2], args.units[-1]), name = 'Output_reshape'))
    
    # Add the output layer
    model.add(Dense(units = 3, use_bias = True, name = 'Output', activation = args.output_activation))
    
    
    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = keras.optimizers.Adam(learning_rate = args.lrate, beta_1 = 0.9, beta_2 = 0.999,
                                epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)
    
    # Build the model and define the loss function
    mode = 'temporal' if np.invert(args.class_weight == None) else None
    
    model.compile(loss = args.loss, optimizer = opt, 
                  metrics = ['categorical_accuracy', tf.keras.metrics.Precision(name = 'precision'), 
                             tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.AUC(name = 'auc')], sample_weight_mode = mode)
    
    return model



#%%
##############################################

# Function to make a Convolutional U-net model
def build_cnn_model(args, shape):
    '''
    Build a convolutional U-net either sequentially (without skip connections) or non-sequentially (with skip connections)
    
    :Inputs:
    :param args: Argparse arguments
    :param shape: The shape of the training data (size/n_examples x map shape x n_variables)
    
    Outputs:
    :param model: Convolutional NN model
    '''
    
    # Create a dictionary of the arguments for convience
    arg_dict = {'sequential': args.sequential,
                'map_size': shape,
                'nfilters': args.nfilters,
                'kernel_size': args.kernel_size,
                'strides': args.strides,
                'pool_size_horizontal': args.pool_size_horizontal,
                'pool_size_vertical': args.pool_size_vertical,
                'activation': args.activation,
                'output_activation': args.output_activation,
                'loss': args.loss,
                'dropout': args.dropout,
                'L1_regularizer': args.L1_regularization,
                'L2_regularizer': args.L2_regularization,
                'lrate': args.lrate,
                'metrics': ['categorical_accuracy', tf.keras.metrics.AUC(name = 'auc')],#, 
               #             tf.keras.metrics.Precision(name = 'precision'), 
               #             tf.keras.metrics.Recall(name = 'recall')], # Recall and precision may be used, but could cause errors when using variational autoencoders
                'class_weight': args.class_weight,
                'variational': args.variational,
                # Data augmentation parameters
                'data_augmentation': args.data_augmentation,
                'crop_height': args.crop_height,
                'crop_width': args.crop_width,
                'flip': args.flip,
                'rotation': args.data_aug_rotation,
                'translate_height': args.translation_height,
                'translate_width': args.translation_width,
                'zoom_height': args.zoom_height,
                'zoom_width': args.zoom_width}
    
    return sequential_cnn(arg_dict) if args.sequential else model_cnn(arg_dict)

class Sampling(tf.keras.layers.Layer):
    '''Uses inputs (z_mean, z_log_var) to sample a tensor z from a vector. 
       Code taken from the keras autoencoder example at https://keras.io/examples/generative/vae/'''
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dimension = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dimension))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon


def sequential_cnn(args):
    '''
    Build a sequential convolutional U-style autoencoder (does not have skip connections)
    
    :Inputs:
    :param args: Dictionary of parameters for the model. Dictionary must contain:
                     map_size: Size of input data
                     nfilters: List of the number of filters in each CNN layer
                     kernel_size: List of kernel size of each CNN layer
                     strides: List of the number of strides of each CNN layer
                     pool_size: List of the stride for pooling/upscale size for each CNN layer (1 = no pooling/upscaling)
                     activation: Activation function/Nonlinearity for each CNN layers
                     output_activation: Activation function/Nonlinearity for the output layer
                     loss: Loss function being minimized
                     dropout: Dropout probability for dropout layers
                     lambda_l1: Lambda_1 parameter for L1 regularization
                     lambda_l2: Lambda_2 parameter for L2 regularization
                     lrate: Learning rate for the model
                     metrics: List of metrics to calculate for each epoch and for evaluations
                     class_weight: Float of the class weight that will be applied
                     variational: Boolean indicating whether to use a variational autoencoder
                     data_augmentation: Boolean indicating whether to use data augmentation (DA)
                     crop_height: Number of pixels/grid points to crop along the height axis (DA parameter)
                     crop_width: Number of pixels/grid points to crop along the width axis (DA parameter)
                     flip: String of either 'horizontal', 'vertical', 'both', or None for flipping the image (DA parameter)
                     rotation: Float of how much to rotation the image 
                               (value is a decimal percentage of rotation, e.g., 0.5 is half of 2*pi rotation; DA parameter)
                     translate_height: Float of how much to translate the image in the height axis (DA parameter)
                     translate_width: Float of how much to translate the image in the width axis (DA parameter)
                     zoom_height: Float of how much to translate the image in the height axis (DA parameter)
                     zoom_width: Float of how much to translate the image in the width axis (DA parameter)
    
    :Outputs:
    :param model: Sequential CNN model
    '''
    # Define the regularizer
    if args['L2_regularizer'] is not None:
        kernel_regularizer = tf.keras.regularizers.l2(args['L2_regularizer'])
    elif args['L1_regularizer'] is not None:
        kernel_regularizer = tf.keras.regularizers.l1(args['L1_regularizer'])
    else:
        kernel_regularizer = None
    
    # Create the model
    model = Sequential()
    
    #### Note, depending on how the data is set up, this may take some editing for 1D or 2D data.
    ### There are 9 locations to change 1D/2D shapes (InputLayer, Conv1D/Conv2D, SpatialDropout1D/2D, MaxPooling1D/2D, UpSampling1D/2D, Conv1D/2D,
    ### SpatialDropout1D/2D, Conv1D/2D, and commenting/uncommenting Reshape)
    
    # Add the input layer
    model.add(InputLayer(input_shape = (args['map_size'][1], args['map_size'][2], args['map_size'][3]), name = 'Input'))
    
    # Add data augmentation?
    if args['data_augmentation']:
        data_aug_list = []
        
        # Add cropping?
        if np.invert(args['crop_height'] == None) | np.invert(args['crop_width'] == None):
            if args['crop_height'] == None:
                args['crop_height'] = 0
            if args['crop_width'] == None:
                args['crop_width'] = 0
            
            data_aug_list.append(tf.keras.layers.RandomCrop(args['crop_height'], args['crop_width'], name = 'Crop'))
            
        # Flip the image?
        elif np.invert(args['flip'] == 'None'):
            data_aug_list.append(tf.keras.layers.RandomFlip(mode = args['flip'], name = 'Flip'))
        
        # Add rotation?
        elif np.invert(args['rotation'] == None):
            data_aug_list.append(tf.keras.layers.RandomRotation(args['rotation'], fill_mode = 'constant', name = 'Rotate'))
            
        # Add translation?
        elif np.invert(args['translate_height'] == None) | np.invert(args['translate_width'] == None):
            if args['translate_height'] == None:
                args['translate_height'] = 0
            if args['translate_width'] == None:
                args['translate_width'] = 0
            
            data_aug_list.append(tf.keras.layers.RandomTranslation(args['translate_height'], args['translate_width'], fill_mode = 'constant',
                                                                  name = 'Translate'))
        # Add zooming?
        elif np.invert(args['zoom_height'] == None) | np.invert(args['zoom_width'] == None):
            if args['zoom_height'] == None:
                args['zoom_height'] = 0
            if args['zoom_width'] == None:
                args['zoom_width'] = 0
                
            data_aug_list.append(tf.keras.layers.RandomZoom(args['zoom_height'], args['zoom_width'], fill_mode = 'constant', name = 'Zoom'))
            
        data_augmentation = Sequential(data_aug_list, name = 'Data_Augmentation')
        model.add(data_augmentation)
    
    # Build the encode side
    for n, (nf, k, s, psh, psv) in enumerate(zip(args['nfilters'], 
                                                 args['kernel_size'], 
                                                 args['strides'], 
                                                 args['pool_size_horizontal'], 
                                                 args['pool_size_vertical'])):
        # Add the convolutional layer(s)
        model.add(Conv2D(kernel_size = k,
                         filters = nf, 
                         strides = s,
                         activation = args['activation'][n],
                         padding = 'same',
                         use_bias = True,
                         kernel_initializer = 'random_uniform',
                         bias_initializer = 'zeros',
                         kernel_regularizer = kernel_regularizer,
                         name = 'CDown%d'%(n+1)))
        
        # Add dropout?
        if args['dropout'] is not None:
            model.add(SpatialDropout2D(rate = args['dropout'], name = 'Spatial_Dropout_Down%d'%(n+1)))
            
        # Downscale?
        if (psh > 1) | (psv > 1):
            #model.add(MaxPooling1D(pool_size = psh,
            #                       strides = psh,
            #                       name = 'MAX%d'%(n+1)))
            model.add(MaxPooling2D(pool_size = (psv, psh),
                                   strides = (psv, psh),
                                   name = 'MAX%d'%(n+1)))
            
    # Make a variational encoder? (Currently only works with non-sequential models)
    if args['variational']:
        pass
        #x, y, filters = tf.shape(tensor).numpy()
        #model.add(Flatten())
        #model.add(Dense(int(args['nfilters'][-1]*1.5), activate = 'elu', name = 'encoder_dense'))
        #tensor_mean = Dense(2, activation = 'elu', name = 'encoder_mean')(tensor)
        #tensor_log_var = Dense(2, activation = 'elu', name = 'encoder_log_var')(tensor)
        #model.add(Sampling(tensor_mean, tensor_log_var))
        #model.add(Dense(x*y*filters, activation = 'elu', name = 'decoder_dense'))
        #model.add(Reshape(x, y, filters)(tensor))
            
            
    # Build the decoder side
    for n, (nf, k, s, ush, usv) in enumerate(zip(reversed(args['nfilters']), 
                                                 reversed(args['kernel_size']), 
                                                 reversed(args['strides']), 
                                                 reversed(args['pool_size_horizontal']), 
                                                 reversed(args['pool_size_vertical']))):

        # Upsample?
        if (ush > 1) | (usv > 1):
            #model.add(UpSampling1D(size = ush, 
            #                       name = 'UpSample%d'%(n+1)))
            model.add(UpSampling2D(size = (usv, ush), 
                                   name = 'UpSample%d'%(n+1)))


        # Add convolutional layers
        model.add(Conv2D(kernel_size = k,
                         filters = nf, 
                         strides = s,
                         activation = args['activation'][n],
                         padding = 'same',
                         use_bias = True,
                         kernel_initializer = 'random_uniform',
                         bias_initializer = 'zeros',
                         kernel_regularizer = kernel_regularizer,
                         name = 'CUp%d'%(n+1)))

        # Add dropout?
        if args['dropout'] is not None:
            model.add(SpatialDropout2D(rate = args['dropout'], name = 'Spatial_Dropout_Up%d'%(n+1)))

    # Add the output layer
    model.add(Conv2D(kernel_size = 1,
                     filters = 3, 
                     strides = 1,
                     activation = args['activation'][-1],
                     use_bias = True,
                     kernel_initializer = 'random_uniform',
                     bias_initializer = 'zeros',
                     kernel_regularizer = kernel_regularizer,
                     name = 'Output_CNN'))

    
    # Add the output layer
    model.add(Dense(units = 3, use_bias = True, name = 'Output', activation = args['output_activation']))

    # This last reshape and dense layer allows the use of sample weights (data shape must be < 3D)
    model.add(Reshape((args['map_size'][1]*args['map_size'][2], 3), name = 'Output_reshape'))

    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = tf.keras.optimizers.Adam(learning_rate = args['lrate'], beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)

    # Build the model and define the loss function
    mode = 'temporal' if (args['class_weight'] != None) else None
    
    # Compile the model
    model.compile(loss = args['loss'], optimizer = opt, metrics = args['metrics'], sample_weight_mode = mode)
    
    return model
    
def model_cnn(args):
    '''
    Build a non-sequential convolutional U-net with skip connections right before each pool/right after each upsample
    
    :Inputs:
    :param args: Dictionary of parameters for the model. Dictionary must contain:
                     image_size: Size of input data
                     nclasses: Number of classes to be predicted
                     nfilters: List of the number of filters in each CNN layer
                     kernel_size: List of kernel size of each CNN layer
                     strides: List of the number of strides of each CNN layer
                     pool_size: List of the stride for pooling/upscale size for each CNN layer (1 = no pooling/upscaling)
                     activation: Activation function/Nonlinearity for each CNN layers
                     output_activation: Activation function/Nonlinearity for the output layer
                     loss: Loss function being minimized
                     dropout: Dropout probability for dropout layers
                     lambda_l1: Lambda_1 parameter for L1 regularization
                     lambda_l2: Lambda_2 parameter for L2 regularization
                     lrate: Learning rate for the model
                     metrics: List of metrics to calculate for each epoch and for evaluations
                     class_weight: Float of the class weight that will be applied
                     variational: Boolean indicating whether to use a variational autoencoder
                     data_augmentation: Boolean indicating whether to use data augmentation (DA)
                     crop_height: Number of pixels/grid points to crop along the height axis (DA parameter)
                     crop_width: Number of pixels/grid points to crop along the width axis (DA parameter)
                     flip: String of either 'horizontal', 'vertical', 'both', or None for flipping the image (DA parameter)
                     rotation: Float of how much to rotation the image 
                               (value is a decimal percentage of rotation, e.g., 0.5 is half of 2*pi rotation; DA parameter)
                     translate_height: Float of how much to translate the image in the height axis (DA parameter)
                     translate_width: Float of how much to translate the image in the width axis (DA parameter)
                     zoom_height: Float of how much to translate the image in the height axis (DA parameter)
                     zoom_width: Float of how much to translate the image in the width axis (DA parameter)
    
    :Outputs:
    :param model: Sequential CNN model
    '''
    # Define the regularizer
    if args['L2_regularizer'] is not None:
        kernel_regularizer = tf.keras.regularizers.l2(args['L2_regularizer'])
    elif args['L1_regularizer'] is not None:
        kernel_regularizer = tf.keras.regularizers.l1(args['L1_regularizer'])
    else:
        kernel_regularizer = None
    
    
    #### Note, depending on how the data is set up, this may take some editing for 1D or 2D data.
    ### There are 10 locations to change 1D/2D shapes (InputLayer, Conv1D/Conv2D, SpatialDropout1D/2D, MaxPooling1D/2D, 
    ### (x,y,filters reshape in variational component), UpSampling1D/2D, Conv1D/2D,
    ### SpatialDropout1D/2D, Conv1D/2D, and commenting/uncommenting Reshape
    
    # Define the input tensor
    input_tensor = Input(shape = (args['map_size'][1], args['map_size'][2], args['map_size'][3]), name = 'Input')
    tensor = input_tensor
    
    # Add data augmentation?
    if args['data_augmentation']:
        data_aug_list = []
        
        # Crop the image?
        if np.invert(args['crop_height'] == None) | np.invert(args['crop_width'] == None):
            if args['crop_height'] == None:
                args['crop_height'] = 0
            if args['crop_width'] == None:
                args['crop_width'] = 0
            
            data_aug_list.append(tf.keras.layers.RandomCrop(args['crop_height'], args['crop_width'], name = 'Crop'))
            
        # Flip the image?
        elif np.invert(args['flip'] == 'None'):
            data_aug_list.append(tf.keras.layers.RandomFlip(mode = args['flip'], name = 'Flip'))
        
        # Rotate the image?
        elif np.invert(args['rotation'] == None):
            data_aug_list.append(tf.keras.layers.RandomRotation(args['rotation'], fill_mode = 'constant', name = 'Rotate'))
            
        # Translate the image?
        elif np.invert(args['translate_height'] == None) | np.invert(args['translate_width'] == None):
            if args['translate_height'] == None:
                args['translate_height'] = 0
            if args['translate_width'] == None:
                args['translate_width'] = 0
            
            data_aug_list.append(tf.keras.layers.RandomTranslation(args['translate_height'], args['translate_width'], fill_mode = 'constant', 
                                                                  name = 'Translate'))
            
        # Zoom the image?
        elif np.invert(args['zoom_height'] == None) | np.invert(args['zoom_width'] == None):
            if args['zoom_height'] == None:
                args['zoom_height'] = 0
            if args['zoom_width'] == None:
                args['zoom_width'] = 0
                
            data_aug_list.append(tf.keras.layers.RandomZoom(args['zoom_height'], args['zoom_width'], fill_mode = 'constant', name = 'Zoom'))
            
        data_augmentation = Sequential(data_aug_list, name = 'Data_Augmentation')
        tensor = data_augmentation(tensor)
    
    # Define an empty list to be used for skip connections
    skip_connections = []
    
    # Build the encoder side
    for n, (nf, k, s, psh, psv) in enumerate(zip(args['nfilters'], 
                                                 args['kernel_size'], 
                                                 args['strides'], 
                                                 args['pool_size_horizontal'], 
                                                 args['pool_size_vertical'])):
        # Add the convolutional layer(s)
        tensor = Conv2D(kernel_size = k,
                        filters = nf, 
                        strides = s,
                        activation = args['activation'][n],
                        padding = 'same',
                        use_bias = True,
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'zeros',
                        kernel_regularizer = kernel_regularizer,
                        name = 'CDown%d'%(n+1))(tensor)
        
        # Add dropout?
        if args['dropout'] is not None:
            tensor = SpatialDropout2D(rate = args['dropout'], name = 'Spatial_Dropout_Down%d'%(n+1))(tensor)
            
        # Downscale?
        if (psh > 1) | (psv > 1):
            # Skip connections will be placed right before downscaling/right after upscaling
            skip_connections.append(tensor)
            
            #tensor = MaxPooling1D(pool_size = psh, 
            #                      strides = psh,
            #                      name = 'MAX%d'%(n+1))(tensor)
            tensor = MaxPooling2D(pool_size = (psv, psh), 
                                  strides = (psv, psh),
                                  name = 'MAX%d'%(n+1))(tensor)
    
    # Make the encoder a variational encoder?
    if args['variational']:
        x = tensor.shape[1]
        y = tensor.shape[2]
        #filters = tensor.shape[2]
        filters = tensor.shape[3]
        
        # Flattern the tensor
        tensor = Flatten()(tensor)
        tensor = Dense(int(args['nfilters'][-1]*1.5), activation = 'elu', name = 'encoder_dense')(tensor)
        
        # Determine the mean and standard deviation
        tensor_mean = Dense(2, activation = 'elu', name = 'encoder_mean')(tensor)
        tensor_log_var = Dense(2, activation = 'elu', name = 'encoder_log_var')(tensor)
        
        # Sample the mean and standard deviation
        tensor = Sampling()([tensor_mean, tensor_log_var])
        
        # Restore the sampled data back to the original latent layer size
        tensor = Dense(x*y*filters, activation = 'elu', name = 'decoder_dense')(tensor)
        tensor = Reshape((x, y, filters), name = 'decoder_reshape')(tensor)
            
            
    # Build the decoder side
    for n, (nf, k, s, ush, usv) in enumerate(zip(reversed(args['nfilters']), 
                                                 reversed(args['kernel_size']), 
                                                 reversed(args['strides']), 
                                                 reversed(args['pool_size_horizontal']), 
                                                 reversed(args['pool_size_vertical']))):
        # Upsample?
        if (ush > 1) | (usv > 1):
            #tensor = UpSampling1D(size = ush, 
            #                      name = 'UpSample%d'%(n+1))(tensor)
            tensor = UpSampling2D(size = (usv, ush), 
                                  name = 'UpSample%d'%(n+1))(tensor)

            # Attach skip connection
            tensor = Concatenate()([tensor, skip_connections.pop()])

        # Add the convolutional layer(s)
        tensor = Conv2D(kernel_size = k,
                        filters = nf, 
                        strides = s,
                        activation = args['activation'][n],
                        padding = 'same',
                        use_bias = True,
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'zeros',
                        kernel_regularizer = kernel_regularizer,
                        name = 'CUp%d'%(n+1))(tensor)

        # Add dropout?
        if args['dropout'] is not None:
            tensor = SpatialDropout2D(rate = args['dropout'], name = 'Spatial_Dropout_Up%d'%(n+1))(tensor)

    # Add the output layer
    tensor = Conv2D(kernel_size = 1,
                    filters = 3, 
                    strides = 1,
                    activation = args['activation'][-1],
                    use_bias = True,
                    kernel_initializer = 'random_uniform',
                    bias_initializer = 'zeros',
                    kernel_regularizer = kernel_regularizer,
                    name = 'Output_Convolution')(tensor)
    
    
    # Add the output layer
    output_tensor = Dense(units = 3, use_bias = True, #bias_initializer = keras.initializers.Constant(-5.0682), # Number comes from  np.log(# pos obs/# total obs)
                          name = 'Output', activation = args['output_activation'])(tensor)

    # This last reshape and dense layer allows the use of sample weights (data shape must be < 3D)
    output_tensor = Reshape((args['map_size'][1]*args['map_size'][2], 3), name = 'Output_reshape')(output_tensor)

    # Create the model
    model = Model(inputs = input_tensor, outputs = output_tensor)

    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = tf.keras.optimizers.Adam(learning_rate = args['lrate'], beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)

    # Build the model and define the loss function
    mode = 'temporal' if (args['class_weight'] != None) else None
    
    # Compile the model
    model.compile(loss = args['loss'], optimizer = opt, metrics = args['metrics'], sample_weight_mode = mode)
    
    return model


#%%
##############################################

# Function to make a Recurrent network model

def build_rnn_model(args, shape):
    '''
    Construct an recurrent neural network (RNN) model using keras
    
    :Inputs:
    :param args: Argparse arguments
    :param shape: The shape of the training data (time x map shape/N samples x n_variables)
    
    :Outputs:
    :param model: Recurrent/Recursive NN model
    '''
    # Define the regularizer
    if args.L1_regularization is not None:
        regularizer = keras.regularizers.l1(args.L1_regularization)
    elif args.L2_regularization is not None:
        regularizer = keras.regularizers.l2(args.L2_regularization)
    else:
        regularizer = None # Define the regularizar for the model, but set to 0 to not use it
        
    # Create the model
    model = Sequential()

    # Add the embedding layer layer
    model.add(InputLayer(input_shape = (None, shape[2]), name = 'Input'))
    
    #for n, (nf, k, s) in enumerate(zip(args.nfilters, 
        #                               args.kernel_size, 
        #                               args.strides)):
        #if k > 5:
        #    model.add(Conv1D(kernel_size = k,
        #                     filters = nf, 
        #                     strides = s,
        #                     activation = args.activation[n],
        #                     padding = 'same',
        #                     use_bias = True,
        #                     kernel_initializer = 'random_uniform',
        #                     bias_initializer = 'zeros',
        #                     kernel_regularizer = regularizer,
        #                     name = 'C%d'%(n+1)))
              
    # Add recurrent layers
    for n, (unit, activation, model_type) in enumerate(zip(args.rnn_units, args.rnn_activation, args.rnn_model)):
        # Add a GRU layer?
        if model_type == 'GRU': # The local bash run did seem to acknowledge the literal 'is' here
            model.add(GRU(unit,
                          activation = activation,
                          use_bias = True,
                          return_sequences = True,
                          kernel_initializer = 'random_uniform',
                          bias_initializer = 'random_uniform',
                          kernel_regularizer = regularizer,
                          dropout = args.dropout,
                          name = 'GRU_layer%d'%(n+1)))

        # Add an LSTM?
        elif model_type == 'LSTM':
            model.add(LSTM(unit,
                           activation = activation,
                           use_bias = True,
                           return_sequences = True,
                           kernel_initializer = 'random_uniform',
                           bias_initializer = 'random_uniform',
                           kernel_regularizer = regularizer,
                           dropout = args.dropout,
                           name = 'LSTM_layer%d'%(n+1)))

        # If another layer is specified, add a simple RNN layer instead
        else:
            model.add(SimpleRNN(unit,
                                activation = activation,
                                use_bias = True,
                                return_sequences = True,
                                kernel_initializer = 'random_uniform',
                                bias_initializer = 'random_uniform',
                                kernel_regularizer = regularizer,
                                dropout = args.dropout,
                                name = 'sRNN_layer%d'%(n+1)))
    
    # Add dense layers
    for n, unit in enumerate(args.units):
        model.add(Dense(unit, 
                        use_bias = True,
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'zeros',
                        activation = args.activation[n],
                        name = 'D%d'%(n+1), 
                        kernel_regularizer = regularizer))
        
        # Add dropout?
        if args.dropout is not None:
            model.add(Dropout(rate = args.dropout, name = 'D%d_dropout'%(n+1)))
            
    # Add the output layer
    # Activation for the output layer must be softmax
    model.add(Dense(3, 
                    use_bias = True, 
                    activation = args.output_activation, 
                    kernel_initializer = 'random_uniform',
                    bias_initializer = 'zeros',
                    name = 'Output'))
    
    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = tf.keras.optimizers.Adam(learning_rate = args.lrate, beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)
    
    # Build the model and define the loss function
    mode = 'temporal' if np.invert(args.class_weight == None) else None
    
    model.compile(loss = args.loss, optimizer = opt, 
                  metrics = ['categorical_accuracy', tf.keras.metrics.Precision(name = 'precision'), 
                             tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.AUC(name = 'auc')], sample_weight_mode = mode)
    
    return model


#%%
##############################################

# Function to make a CNN-RNN model
def build_cnn_rnn_model(args, shape):
    '''
    Construct an CNN-RNN network model using keras. The begins with a convolutional network, then goes to into a RNN.
    
    :Inputs:
    :param args: Argparse arguments
    :param shape: The shape of the training data (time x map shape/N samples x n_variables)
    
    :Outputs:
    :param model: Recurrent/Recursive NN model
    '''
    # Define the regularizer
    if args.L1_regularization is not None:
        regularizer = keras.regularizers.l1(args.L1_regularization)
    elif args.L2_regularization is not None:
        regularizer = keras.regularizers.l2(args.L2_regularization)
    else:
        regularizer = None # Define the regularizar for the model, but set to 0 to not use it
        
    # Create the model
    model = Sequential()

    # Add the embedding layer layer
    model.add(InputLayer(input_shape = (None, shape[2]), name = 'Input'))
    
    # Add the convolutional layer(s)
    for n, (nf, k, s) in enumerate(zip(args.nfilters, 
                                       args.kernel_size, 
                                       args.strides)):
        
        # Add the convolutional layer(s)
        #if k > 1:
        model.add(Conv1D(kernel_size = k,
                         filters = nf, 
                         strides = s,
                         activation = args.activation[n],
                         padding = 'same',
                         use_bias = True,
                         kernel_initializer = 'random_uniform',
                         bias_initializer = 'zeros',
                         kernel_regularizer = regularizer,
                         name = 'C%d'%(n+1)))

        # Add dropout?
        if args.dropout is not None:
            model.add(SpatialDropout1D(rate = args.dropout, name = 'Spatial_Dropout_Down%d'%(n+1)))
            
            
    # Add recurrent layers
    for n, (unit, activation, model_type) in enumerate(zip(args.rnn_units, args.rnn_activation, args.rnn_model)):
       # Add a GRU layer?
       if model_type == 'GRU': # The local bash run did seem to acknowledge the literal 'is' here
           model.add(GRU(unit,
                         activation = activation,
                         use_bias = True,
                         return_sequences = True,
                         kernel_initializer = 'random_uniform',
                         bias_initializer = 'random_uniform',
                         kernel_regularizer = regularizer,
                         dropout = args.dropout,
                         name = 'GRU_layer%d'%(n+1)))

       # Add an LSTM?
       elif model_type == 'LSTM':
           model.add(LSTM(unit,
                          activation = activation,
                          use_bias = True,
                          return_sequences = True,
                          kernel_initializer = 'random_uniform',
                          bias_initializer = 'random_uniform',
                          kernel_regularizer = regularizer,
                          dropout = args.dropout,
                          name = 'LSTM_layer%d'%(n+1)))

       # If another layer is specified, add a simple RNN layer instead
       else:
           model.add(SimpleRNN(unit,
                               activation = activation,
                               use_bias = True,
                               return_sequences = True,
                               kernel_initializer = 'random_uniform',
                               bias_initializer = 'random_uniform',
                               kernel_regularizer = regularizer,
                               dropout = args.dropout,
                               name = 'sRNN_layer%d'%(n+1)))
    
    # Add dense layers
    for n, unit in enumerate(args.units):
        model.add(Dense(unit, 
                        use_bias = True,
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'zeros',
                        activation = args.activation[n],
                        name = 'D%d'%(n+1), 
                        kernel_regularizer = regularizer))
        
        # Add dropout?
        if args.dropout is not None:
            model.add(Dropout(rate = args.dropout, name = 'D%d_dropout'%(n+1)))
            
    # Add the output layer
    # Activation for the output layer must be softmax
    model.add(Dense(3, 
                    use_bias = True, 
                    activation = args.output_activation, 
                    kernel_initializer = 'random_uniform',
                    bias_initializer = 'zeros',
                    name = 'Output'))
    
    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = tf.keras.optimizers.Adam(learning_rate = args.lrate, beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)
    
    # Build the model and define the loss function
    mode = 'temporal' if np.invert(args.class_weight == None) else None
    
    model.compile(loss = args.loss, optimizer = opt, 
                  metrics = ['categorical_accuracy', tf.keras.metrics.AUC(name = 'auc')], sample_weight_mode = mode)
                             #tf.keras.metrics.Precision(name = 'precision'), 
                             #tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.AUC(name = 'auc')], sample_weight_mode = mode)
    
    return model


#%%
##############################################

# Function to make an attention network model, based on a transformer model

def build_attention_model(args, shape):
    '''
    Construct an recurrent neural network (RNN) model using keras
    
    :Inputs:
    :param args: Argparse arguments
    :param shape: The shape of the training data (time x map shape/N samples x n_variables)
    
    :Outputs:
    :param model: Recurrent/Recursive NN model
    '''
    # Define the regularizer
    if args.L1_regularization is not None:
        regularizer = keras.regularizers.l1(args.L1_regularization)
    elif args.L2_regularization is not None:
        regularizer = keras.regularizers.l2(args.L2_regularization)
    else:
        regularizer = None # Define the regularizar for the model, but set to 0 to not use it
        
    if args.dropout is None:
        dropout = 0.0
    else:
        dropout = args.dropout
        
    
    # Define the input tensor
    input_tensor = Input(shape = (shape[1], shape[2], shape[3]), name = 'Input')
    tensor = input_tensor
         
    #### Do Convolution before hand?

    # Add a transformer encoder layer?
    if (args.encoder_decoder.lower() == 'encoder') | (args.encoder_decoder.lower() == 'both'):
        tensor = layers.TransformerEncoderBlock(num_attention_heads = args.attention_heads,
                                                inner_dim = args.inner_unit,
                                                inner_activation = args.inner_activation,
                                                activity_regularizer = regularizer,
                                                output_dropout = dropout,
                                                attention_dropout = dropout,
                                                inner_dropout = dropout)(tensor)

    # Add a transformer decoder layer?
    elif (args.encoder_decoder.lower() == 'decoder') | (args.encoder_decoder.lower() == 'both'):
        tensor = layers.TransformerDecoderBlock(num_attention_heads = args.attention_heads,
                                                intermediate_size = args.inner_unit,
                                                intermediate_activation = args.inner_activation,
                                                dropout_rate = dropout,
                                                attention_dropout_rate = dropout,
                                                multi_channel_cross_attention = False)
    
    
    # This last reshape and dense layer allows the use of sample weights (data shape must be < 3D)
    tensor = Reshape((shape[1]*shape[2], shape[3]), name = 'Output_reshape')(tensor)
    
    # Add the output layer
    # Activation for the output layer must be softmax
    output_tensor = Dense(3, 
                          use_bias = True, 
                          activation = args.output_activation, 
                          kernel_initializer = 'random_uniform',
                          bias_initializer = 'zeros',
                          name = 'Output')(tensor)
    
    # Create the model
    model = Model(inputs = input_tensor, outputs = output_tensor)
    
    # Define the optimizer
    # NOTE: In newer versions of TF, the decay parameter is weight_decay
    # Likewise, None is not a valid entry in newer versions; epsilon = 1e-7 (default) needed instead
    opt = tf.keras.optimizers.Adam(learning_rate = args.lrate, beta_1 = 0.9, beta_2 = 0.999,
                                   epsilon = 1e-7, weight_decay = 0.0, amsgrad = False)
    
    # Build the model and define the loss function
    mode = 'temporal' if np.invert(args.class_weight == None) else None
    
    model.compile(loss = args.loss, optimizer = opt, 
                  metrics = ['categorical_accuracy', tf.keras.metrics.Precision(name = 'precision'), 
                             tf.keras.metrics.Recall(name = 'recall'), tf.keras.metrics.AUC(name = 'auc')], sample_weight_mode = mode)
    
    return model

############
# A TF custom loss function, modifying the cross entropy loss to 1/N*sum(alpha*(1 - p)**gamma * y * log(p))
# Code was obtained from, and information on the focal loss can be found at, 
# https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
def focal_loss(gamma=2, alpha=4):
    '''
    Custom loss function (focal loss), that modifies the categorical cross-entropy loss for class imbalance
    
    :Inputs:
    :param gamma, alpha: Loss function parameters that controll how much emphasis is palced on class imbalance
    
    :Outputs:
    :param focal_loss_fixed: Focal loss
    '''

    # Turn the parameters into floats
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in the paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        # Calculate the focal loss
        model_out = tf.add(y_pred, epsilon)
        
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        
        weight = tf.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
        
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        
        reduced_fl = tf.reduce_max(fl, axis=1)
        
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def variational_loss(loss = 'categorical_crossentropy', gamma = 2.0, alpha = 4.0):
    '''
    Define the combined loss functions (loss + KL divergence loss) for variational autoencoders
    
    :Inputs:
    :param loss: String indicating main loss function (must be 'categorical_crossentropy' or 'focal')
    :param gamma, alpha: Parameters for the focal loss function controlling the emphasis on class imbalance
    
    :Outputs:
    :param combine_loss: The combined loss functions
    '''
    
    gamma = float(gamma)
    alpha = float(alpha)
    
    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in the paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        # Calculate the focal loss
        model_out = tf.add(y_pred, epsilon)
        
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        
        weight = tf.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
        
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        
        reduced_fl = tf.reduce_max(fl, axis=1)
        
        return tf.reduce_mean(reduced_fl)
    
    def combine_loss(y_true, y_pred):
        '''
        Combine the the KL Divergence and focal/categorical cross entropy loss functions for variational autoencoders
        
        :Inputs:
        :param y_true: True label/prediction
        :param y_pred: Predicted label/prediction
        
        :Outputs:
        :param combined_loss: The combined loss functions
        '''
        
        # Determine the main loss
        if loss == 'categorical_crossentropy':
            combined_loss = tf.add(tf.keras.losses.categorical_crossentropy(y_true, y_pred), tf.keras.losses.kl_divergence(y_true, y_pred))
        elif loss == 'focal':
            combined_loss = tf.add(focal_loss_fixed(y_true, y_pred), tf.keras.losses.kl_divergence(y_true, y_pred))
        #else:
        #    raise "Variational encoder loss is only compatible with categoricall crossentropy and focal losses (currently)"
            
        return combined_loss
    return combine_loss
        
def cce_sample_weights(sample_weights = None):
     """
     Apply the categorical cross entropy loss with arbitrary dimension sample weights.
     Note the sample weights MUST have the same shape as the model outputs.
     
     :Inputs:
     :param sample_weights: Sample weights to be applied to each input in the loss
     
     :Outputs:
     :param custom_cce: Categorical cross entropy loss function 
     """
 
     if sample_weights is None:
         sample_weights = 1.
     else:
         sample_weights = tf.convert_to_tensor(sample_weights, tf.float32)
 
         
     def custom_cce(y_true, y_pred):
         """
         Custom categorical cross entropy function that applies sample weights to each value without reshaping the sample weights to 2D.
         """
         epsilon = 1e-9
         y_true = tf.convert_to_tensor(y_true, tf.float32)
         y_pred = tf.convert_to_tensor(y_pred, tf.float32)
         
         model_out = tf.add(y_pred, epsilon)
         ce = tf.multiply(sample_weights, tf.multiply(y_true, -tf.math.log(model_out)))
         
         reduced_ce = tf.reduce_max(ce, axis = 1)
         return tf.reduce_mean(reduced_ce)
     return custom_cce


###### Some outdated functions that no longer with the current version of the experiments

#def load_all_models(keras, ml_model, model, method, rotation = 0, path = './Data/narr/christian_models'):
#    '''
#    Load all ML models associated with a single FD method and a single dataset
#    
#    Inputs:
#    :param keras: A boolean indicating whether a keras ML model is being loaded
#    :param ml_model: The name of the ML model being loaded
#    :param ra_model: The name of the reanalysis model used to train the ML model
#    :param method: The name of the FD identification method the ML model learned
#    :param rotations: List of rotations over which to load the data
#    :param path: Path to the ML models that will be loaded
#    
#    Outputs:
#    :param models: A list of all machine learning models loaded for a single FD identification method and trained on a single reanalysis
#    '''
#    
#    # Initialize the models
#    models = []
#    
#    # Load lat/lon labels
#    with open('%s/lat_lon_labels.pkl'%(path), 'rb') as fn:
#        lat_labels = pickle.load(fn)
#        lon_labels = pickle.load(fn)
#    
#    I = len(lat_labels)
#    J = len(lon_labels)
#    
#    # Load all models for each combination of lat and lon labels
#    for n in range(len(lat_labels)):
#            
#        # Generate the model filename
#        model_fbase = generate_model_fname_build(model, ml_model, method, rotation, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
#        model_fname = '%s/%s/%s/%s'%(path, ml_model, method, model_fbase)
#        print(model_fname)
#        
#        model = load_single_model(model_fname, keras)
#        
#        models.append(model)
#    
#            
#    return models
#
## Function to make predictions for all models in a rotation (and average them together/take the standard deviation)
#def make_predictions(data, lat, lon, probabilities, threshold, keras, ml_model, ra_model, 
#                     method, rotations, label, path = './Data/narr/christian_models'):
#    '''
#    Function designed make predictions of FD for a full dataset for a given ML model. Predictions are the average over all rotations and the standard deviation.
#    
#    Inputs:
#    :param data: Data used to make FD predictions. Must be in a NVar x time x space format
#    :param lat: The latitude coordinates corresponding to data
#    :param lon: The longitude coordinates corresponding to data
#    :param probabilities: Boolean indicating whether to use return the average probabilistic predictions (true), or average yes/no predictions (false)
#    :param theshold: The probability threshold above which FD is said to occur
#    :param keras: A boolean indicating whether a keras ML model is being used
#    :param ml_model: The name of the ML model being used
#    :param ra_model: The name of the reanalysis model used to train the ML model
#    :param method: The name of the FD identification method the ML model learned
#    :param rotations: List of rotations over which to load the data
#    :param label: The experiment label of the ML models
#    :param path: Path to the ML models that will be loaded
#    
#    Outputs:
#    :param pred: The mean FD predictions of FD
#    :param pred_var: The variation in FD probability predictions across all rotations
#    '''
#    
#        
#    # Reshape lat and lon into 1D arrays
#    print('Initializing some values')
#    I, J = lat.shape
#    lat1d = lat.reshape(I*J, order = 'F')
#    lon1d = lon.reshape(I*J, order = 'F')
#    
#    # Load in lat/lon labels
#    lat_labels = np.arange(-90, 90+5, 5)
#    lon_labels = np.arange(-180, 180+5, 5)
#    
#    I_lab = len(lat_labels)
#    J_lab = len(lon_labels)
#    
#    # Remove NaNs?
#    if np.invert(keras):
#        data[np.isnan(data)] = -995
#    
#    # Initialize the prediction variables
#    NVar, T, IJ = data.shape
#    
#    # data_reshaped = data.reshape(NVar, T, I*J, order = 'F')
#    
#    pred = np.ones((T, I*J)) * np.nan
#    pred_var = np.ones((T, I*J)) * np.nan
#    
#    # Split the dataset into regions
#    print('Splitting data into regions')
#    data_split = []
#    lat_lab = []
#    lon_lab = []
#    for i in range(I_lab-1):
#        for j in range(J_lab-1):
#            ind = np.where( ((lat1d >= lat_labels[i]) & (lat1d <= lat_labels[i+1])) & ((lon1d >= lon_labels[j]) & (lon1d <= lon_labels[j+1])) )[0]
#            
#            # Not all datasets are global; remove sets where there is no data
#            if len(ind) < 1: 
#                continue
#                
#            lat_lab.append(lat_labels[i])
#            lon_lab.append(lon_labels[j])
#            
#            data_split.append(data[:,:,ind])
#            
#    print('There are %d regions.'%len(data_split))
#            
#            
#    # Begin making predictions
#    print('Loading models and making predictions')
#    for n in range(len(data_split)):
#        ind = np.where( ((lat1d >= lat_lab[n]) & (lat1d <= lat_lab[n]+5)) & ((lon1d >= lon_lab[n]) & (lon1d <= lon_lab[n]+5)) )[0]
#        
#        pred_tmp = []
#        for rot in rotations:
#            # Generate the model filename
#            model_fbase = generate_model_fname_build(ra_model, label, method, rot, [lat_lab[n], lat_lab[n]+5], [lon_lab[n], lon_lab[n]+5])
#            model_fname = '%s/%s/%s/%s/%s'%(path, ra_model, ml_model, method, model_fbase)
#            if keras:
#                test_name = model_fname
#            else:
#                test_fname = '%s.pkl'%model_fname
#
#
#            # Check if model exists (it will not if there are no land points)
#            if np.invert(os.path.exists(test_fname)):
#                continue
#            
#            model = load_single_model(model_fname, keras)
#            
#            # Code to make prediction depends on whether a keras model is used
#            if keras:
#                pred_tmp.append(model.predict(data_split[n]))
#            else:
#                NVar, T, IJ_tmp = data_split[n].shape
#                
#                tmp_data = data_split[n].reshape(NVar, T*IJ_tmp, order = 'F')
#                if (ml_model.lower() == 'svm') | (ml_model.lower() == 'support_vector_machine'):
#                    # Note SVMs do not have a predict_proba option
#                    tmp = model.predict(tmp_data.T)
#                    pred_tmp.append(tmp.reshape(T, IJ_tmp, order = 'F'))
#                
#                else:
#                    tmp = model.predict_proba(tmp_data.T)
#
#                    # Check if the model only predicts 0s
#                    only_zeros = tmp.shape[1] <= 1
#                    if only_zeros:
#                        pred_tmp.append(np.zeros((T,len(ind))))
#                    else:
#                        pred_tmp.append(tmp[:,1].reshape(T, IJ_tmp, order = 'F'))
#                    
#        # For sea values, pred_tmp will be empty. Continue to the next region if this happens
#        if len(pred_tmp) < 1:
#            continue
#                    
#        # Take the average of the probabilistic predictions across all rotations
#        pred[:,ind] = np.nanmean(np.stack(pred_tmp, axis = -1), axis = -1)
#
#        # Take the standard deviation of the probabilistic predictions across all rotations
#        pred_var[:,ind] = np.nanstd(np.stack(pred_tmp, axis = -1), axis = -1)
#            
#    # Turn the mean probabilistic predictions into true/false?
#    if np.invert(probabilities):
#        pred = np.where(pred >= threshold, 1, pred)
#        pred = np.where(pred < threshold, 0, pred) # Performing this twice preserves NaN values as not available
#        
#    # Turn the predictions into 3D data
#    pred = pred.reshape(T, I, J, order = 'F')
#    pred_var = pred_var.reshape(T, I, J, order = 'F')
#    
#    return pred, pred_var


