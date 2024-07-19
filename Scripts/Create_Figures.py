#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 17:44:20 2022

@author: stuartedris

This script contains functions used to create the maps and figures to display and visualize the results of the ML models    

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory

TODO:
- Update figures for global scale
- Might add a function to display how forests are making predictions
- Might add a function to display certain NN layers to see how it is identifying FD
- Adjust case_studies to display more than 1 type of growing seaons
- New Figures:
  - Spigatti plot for learning curves
  - Combine climatology with time series

  
- NOTE: TICK and LABEL SIZES HAVE BEEN MODIFIED: TEST THEM

# Note: New functions to include: display_case_study_map_full
"""


#%%
##############################################

# Library impots
import os, sys, warnings
import gc
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from itertools import product
from scipy import stats
from scipy import interpolate
from scipy import signal
from scipy.special import gamma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sklearn

# Import custom scripts
from Raw_Data_Processing import *
from Calculate_Indices import *


#%%
##############################################

# Function to display a set of maps

def display_metric_map(data, lat, lon, methods, metric, cmin, cmax, cint, model, 
                       label, dataset = 'valid', reverse = False, globe = False, path = './Figures'):
    '''
    Display a column of maps, one for each method of FD intendification
    
    Inputs:
    :param data: List of data to be plotted
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param methods: List of methods used to identify FD
    :param metric: The name of the metric being plotted
    :param cmin, cmax, cint: The minimum, maximum, and intervals values respectively for the colorbar
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param reverse: Boolean indicating whether to reverse the colorbar or not
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    
    cname = 'coolwarm_r' if reverse else 'coolwarm'
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 18], nrows = len(methods), ncols = 1, 
                             subplot_kw = {'projection': fig_proj})
    
    if np.invert(globe):
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
        fig.suptitle(model.upper(), y = 0.925, size = 22)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
        fig.suptitle(model.upper(), y = 0.925, size = 22)
        
        
    for m, method in enumerate(methods):
        if len(methods) == 1: # For test cases where only 1 method is examined, the sole axis cannot be subscripted
            ax = axes
            change_pos = 0.2
        else:
            ax = axes[m]
            change_pos = 0.0

        if globe:
            change_pos = change_pos + 0.2
        
        
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        if np.invert(globe):
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            ax.add_feature(cfeature.STATES)
            ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
            ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        else:
            # Ocean covers and "masks" data outside a landmass
            ax.coastlines(edgecolor = 'black', zorder = 3)
            ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
        
        # Adjust the ticks
        ax.set_xticks(LonLabel, crs = fig_proj)
        ax.set_yticks(LatLabel, crs = fig_proj)
        
        ax.set_yticklabels(LatLabel, fontsize = 20)
        ax.set_xticklabels(LonLabel, fontsize = 20)
        
        ax.xaxis.set_major_formatter(LonFormatter)
        ax.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax.pcolormesh(lon, lat, data[m], vmin = cmin, vmax = cmax, cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the extent
        if np.invert(globe):
            ax.set_extent([-130, -65, 23.5, 48.5])
        else:
            ax.set_extent([-179, 179, -60, 75])
        
        # Add method label
        ax.set_ylabel(method.title(), size = 20, labelpad = 45.0, rotation = 0)
        
        # Add a colorbar at the end
        if m == (len(methods)-1):
            cbax = fig.add_axes([0.775 + change_pos, 0.10, 0.020, 0.80])
            cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
            cbar.set_ticks(np.round(np.arange(cmin, cmax+cint, cint*10), 2)) # Set a total of 10 ticks
            for i in cbar.ax.yaxis.get_ticklabels():
                i.set_size(22)
            cbar.ax.set_ylabel(metric.upper(), fontsize = 22)
            
    # Save the figure
    filename = '%s_%s_%s_maps.png'%(metric, label, dataset)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


# Function to display all the metrics in a single figure
def display_metric_map_new(data_list, lat, lon, methods, metrics, cmin, cmax, cint, model, 
                       label, dataset = 'valid', reverse = False, globe = False, path = './Figures'):
    '''
    Display a set of maps, one for each method of FD intendification, and each column of a set of metrics
    
    Inputs:
    :param data_list: List of data to be plotted. Formatted as an outer list, for each metric, and an inner list for each method (data inside the inner list)
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param methods: List of methods used to identify FD
    :param metrics: List of metrics being plotted
    :param cmin, cmax, cint: The minimum, maximum, and intervals values respectively for the colorbar
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param reverse: Boolean indicating whether to reverse the colorbar or not
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to

    Outputs:
    Figure metrics displayed in space for multiple FD identification methods will be created and saved
    '''
    
    
    # Set colorbar information
    cmin = cmin; cmax = cmax; cint = cint
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    
    cname = 'coolwarm_r' if reverse else 'coolwarm'
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [16, 24], nrows = len(methods), ncols = len(metrics),
                             subplot_kw = {'projection': fig_proj})
    
    if np.invert(globe):
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.85)
        fig.suptitle(model.upper(), y = 0.705, size = 16)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = -0.85)
        fig.suptitle(model.upper(), y = 0.705, size = 16)
        

    for n, metric in enumerate(metrics):
        for m, method in enumerate(methods):
            if len(methods) == 1: # For test cases where only 1 method is examined, the sole axis cannot be subscripted
                ax = axes[n]
                change_pos = 0.2
            else:
                ax = axes[m,n]
                change_pos = 0.0
            
            
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
            if np.invert(globe):
                # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
                ax.add_feature(cfeature.STATES)
                ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
                ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
            else:
                # Ocean covers and "masks" data outside a landmass
                ax.coastlines(edgecolor = 'black', zorder = 3)
                ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
            
            # Adjust the ticks
            ax.set_xticks(LonLabel, crs = fig_proj)
            ax.set_yticks(LatLabel, crs = fig_proj)

            # Add labels for the left most and bottom most plots (to prevent overlapping)
            if n == 0:
                ax.set_yticklabels(LatLabel, fontsize = 16)
            else:
                ax.tick_params(labelleft = False)

            if m == len(methods)-1:
                ax.set_xticklabels(LonLabel, fontsize = 16)
            else:
                ax.tick_params(labelbottom = False)
            
            ax.xaxis.set_major_formatter(LonFormatter)
            ax.yaxis.set_major_formatter(LatFormatter)
            
            # Plot the data
            cs = ax.pcolormesh(lon, lat, data_list[n][m], vmin = cmin, vmax = cmax, cmap = cmap, transform = data_proj, zorder = 1)
            
            # Set the extent
            if np.invert(globe):
                ax.set_extent([-130, -65, 23.5, 48.5])
            else:
                ax.set_extent([-179, 179, -60, 75])

            # Add a title
            if m == 0:
                ax.set_title(metric.upper(), size = 16)
            
            # Add method label
            if n == 0:
                ax.set_ylabel(method.title(), size = 16, labelpad = 35.0, rotation = 0)
        
        # Add a colorbar at the end
        if m == (len(methods)-1):
            cbax = fig.add_axes([0.935 + change_pos, 0.32, 0.020, 0.36])
            cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
            cbar.set_ticks(np.round(np.arange(cmin, cmax+cint, cint*10), 2)) # Set a total of 10 ticks
            for i in cbar.ax.yaxis.get_ticklabels():
                i.set_size(16)
            cbar.ax.set_ylabel('Metric [unitless]', fontsize = 16)
            
    # Save the figure
    filename = '%s_%s_metrics_maps.png'%(label, dataset)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to create a confusion matrix for each FD method
def display_confusion_matrix(true, pred, labels, label_names, methods, path = './', savename = 'tmp.png'):
    '''
    Create and display a row of confusion matrices for a set of true and predicted labels. 
    
    Heavily based off of the DisplayConfusionMatrix function in sklearn (slightly modified to display multiple confusion matrices).

    Inputs:
    :param true: List of true labels (each entry in the list is a set of true labels)
    :param pred: List of predicted labels (each entry in the list is a set of predicted labels)
    :param labels: List of different categorical labels for true and pred (e.g., labels = [0, 1] for binary, [0, 1, 2] for three classes, etc.)
    :param label_names: List of names for each label (e.g., label_names = ['No FD', 'FD'])
    :param methods: List of different methods/set of labels (one method/type for each set of labels in true/pred)
    :param path: Path to directory the confusion matrices will be saved to
    :param savename: Filename the confusion matrices will be saved to

    Outputs:
    PNG file of a row of confusion matrices will be saved
    '''


    # Entry for the colorbar to be plotted
    cmin = 0; cmax = 1; cint = 0.05
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    
    cmap  = plt.get_cmap(name = 'Blues', lut = nlevs)

    # Number of classes/labels
    n_classes = len(labels)

    # Start making the plot
    fig, axes = plt.subplots(figsize = [20, 4], nrows = 1, ncols = len(methods))
    plt.subplots_adjust(wspace = 0.1)

    # Make a plot for each method/set of labels
    for m, method in enumerate(methods):
        ax = axes[m]
        
        # Determine the confusion matrix (this is normalized sense most values used in this project are large - order of 1e6 and normalized values have more meaning)
        cm = sklearn.metrics.confusion_matrix(true[m].flatten(), pred[m].flatten(), labels = labels, normalize = 'all')

        # Make the confusion matrix
        im = ax.imshow(cm, vmin = cmin, vmax = cmax, cmap = cmap, interpolation = 'nearest')

        # Add values into the confusion matrix
        thresh = (cm.max() + cm.min())/2
        for i, j in product(range(n_classes), range(n_classes)):
            # Color of the value (matches the background colorbar, but max/min color entry depending on value
            color = im.cmap(nlevs) if cm[i, j] < thresh else im.cmap(0)

            # Add the text
            text = format(cm[i,j], '0.2g')
            ax.text(j, i, text, ha = 'center', va = 'center', color = color, fontsize = 14)
            
        #sklearn.metrics.ConfusionMatrixDisplay.from_predictions(true.flatten(), pred[m].flatten(), labels = labels, display_labels = label_names, 
        #                                                        ax = ax, cmap = cmap, colorbar = False, normalize = 'all',
        #                                                        im_kw = {'vmin': cmin, 'vmax': cmax}, text_kw = {'fontsize': 12})

        # Set the ticks
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))

        # Tick labels
        ax.set_xticklabels(label_names, fontsize = 14)

        ax.set_xlabel('Predicted Label', fontsize = 14)

        # Tick/axes labels on the y-axis should only be for the left most plot (to prevent overlapping onto other plots)
        if m == 0:
            ax.set_yticklabels(label_names, fontsize = 14)
            ax.set_ylabel('True Label', fontsize = 14)
        else:
            ax.set_yticklabels('')
            ax.set_ylabel('')

        # Put the colorbar at the end; put it on a separate axis so it does not shrink the last plot
        if m == len(methods)-1:
            cbax = fig.add_axes([0.915, 0.13, 0.015, 0.72])
            cbar = fig.colorbar(im, cax = cbax, orientation = 'vertical')
            cbar.set_ticks(np.round(np.arange(cmin, cmax+cint, (cmax-cmin)/10), 2)) # Set a total of 10 ticks
            for i in cbar.ax.yaxis.get_ticklabels():
                i.set_size(14)
            #fig.colorbar(im, ax = ax)

        # Add a title for each method/set of labels
        ax.set_title(method, fontsize = 14)

    # Save the plot
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

#%%
##############################################

# Function to display a set of ROC curves

def display_roc_curves(tprs, fprs, tprs_var, fprs_var, methods, model, label, dataset = 'valid', path = './Figures'):
    '''
    Display the (spatially averaged) ROC with the (spatially averaged) variation (across rotations). There will be 1 ROC curve per method
    
    Inputs:
    :param tprs: List of TPR values corresponding to each method
    :param fprs: List of FPR values corresponding to each method
    :param tprs_var: List of TPR variation values corresponding to each method
    :param fprs_var: List of FPR variation values corresponding to each method
    :param methods: List of methods used to identify FD
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [10, 10*len(methods)], nrows = len(methods), ncols = 1)
    
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
    fig.suptitle(model.upper(), y = 0.925, size = 22)
    
    for m, method in enumerate(methods):
        if len(methods) == 1: # For test cases where only 1 method is examined, the sole axis cannot be subscripted
            ax = axes
        else:
            ax = axes[m]
        
        fpr = fprs[m]
        tpr = tprs[m]
        
        fpr_var = fprs_var[m]
        tpr_var = tprs_var[m]
        
        # Sort the data accorinding to the FPR
        ind = fpr.argsort()
        
        fpr = fpr[ind]
        tpr = tpr[ind]
        
        fpr_var = fpr_var[ind]
        tpr_var = tpr_var[ind]
        
        # Use a running average to smooth out the curve a little
        runmean = 50
        
        fpr = np.convolve(fpr, np.ones((runmean,))/runmean)[(runmean-1):]
        tpr = np.convolve(tpr, np.ones((runmean,))/runmean)[(runmean-1):]
        
        fpr_var = np.convolve(fpr_var, np.ones((runmean,))/runmean)[(runmean-1):]
        tpr_var = np.convolve(tpr_var, np.ones((runmean,))/runmean)[(runmean-1):]
        
        # Plot the ROC curve
        ax.plot(fpr, tpr, 'b', linewidth = 2)
        
        # Plot the variation
        ax.fill_between(fpr, tpr-tpr_var, tpr+tpr_var, alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        
        ax.plot([0,1], [0,1], 'k-', linewidth = 1)
        
        # Set the label
        ax.set_ylabel(method.title(), fontsize = 22)
        
        # Set the ticks
        # ax.set_xticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        # ax.set_yticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
        
        # Set the tick size
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(22)
            
    # Save the figure
    filename = '%s_%s_ROC_curves.png'%(label, dataset)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


# Function to display the ROC curves on a single plot
def display_roc_curves_new(tprs, fprs, tprs_var, fprs_var, methods, model, label, dataset = 'valid', path = './Figures'):
    '''
    Display the (spatially averaged) ROC with the (spatially averaged) variation (across rotations). There will be 1 ROC curve per method


    #### NOTE: tprs_var and fprs_var are currently redundant (the shading was removed to prevent strong overlap); they were left in in the event the shading may be determined useful 
    
    Inputs:
    :param tprs: List of TPR values corresponding to each method
    :param fprs: List of FPR values corresponding to each method
    :param tprs_var: List of TPR variation values corresponding to each method
    :param fprs_var: List of FPR variation values corresponding to each method
    :param methods: List of methods used to identify FD
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [10, 10], nrows = 1, ncols = 1)

    ax = axes
    
    for m, method in enumerate(methods):
        
        fpr = fprs[m]
        tpr = tprs[m]
        
        fpr_var = fprs_var[m]
        tpr_var = tprs_var[m]
        
        # Sort the data accorinding to the FPR
        ind = fpr.argsort()
        
        fpr = fpr[ind]
        tpr = tpr[ind]
        
        fpr_var = fpr_var[ind]
        tpr_var = tpr_var[ind]
        
        # Use a running average to smooth out the curve a little
        runmean = 50
        
        fpr = np.convolve(fpr, np.ones((runmean,))/runmean)[(runmean-1):]
        tpr = np.convolve(tpr, np.ones((runmean,))/runmean)[(runmean-1):]
        
        fpr_var = np.convolve(fpr_var, np.ones((runmean,))/runmean)[(runmean-1):]
        tpr_var = np.convolve(tpr_var, np.ones((runmean,))/runmean)[(runmean-1):]
        
        # Calculate the AUC for the ROC curves to be displayed in the legend
        fpr_sorted, tpr_sorted = (list(sorted_list) for sorted_list in zip(*sorted(zip(fpr, tpr))))
        fpr_sorted = np.array(fpr_sorted)
        tpr_sorted = np.array(tpr_sorted)
        auc = sklearn.metrics.auc(fpr_sorted, tpr_sorted)
        
        # Plot the ROC curve
        ax.plot(fpr, tpr, linewidth = 2, label = '%s: AUC = %4.2f'%(method, auc))
        
        # Plot the variation
        #ax.fill_between(fpr, tpr-tpr_var, tpr+tpr_var, alpha = 0.5)

    # Add the legend
    ax.legend(loc = 'lower right', fontsize = 18)

    # Add a straight line to denote 0.5 AUC/no skill
    ax.plot([0,1], [0,1], 'k-', linewidth = 1)
        
    # Set the label
    ax.set_title('ROC Curve', fontsize = 18)
        
    # Set the ticks
    # ax.set_xticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
    # ax.set_yticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
        
    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)
            
    # Save the figure
    filename = '%s_%s_ROC_curves.png'%(label, dataset)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
#%%
##############################################

# Function to display how the metrics vary in time
def display_metrics_in_time(data_list, methods, time, metrics, model = 'rf', path = './'):
    '''
    Display a set of metrics for multiple FD identification methods in time (for test datasets, this is equivalent to metrics across each rotation)

    Inputs:
    :param data_list: Array of data to be plotted (formatted as time x n_methods x n_metrics)
    :param methods: List of FD identification methods
    :param time: Array of datatimes corresponding to time stamps for data_list
    :param metrics: List of metrics to be plotted
    :param model: ML model the set of metrics is evaluating
    :param path: Path to save the figure to

    Outputs:
    Figure of metrics in time saved
    '''

    # Make the plot
    fig, axes = plt.subplots(figsize = [18, 12], nrows = 2, ncols = 2)

    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.3, hspace = 0.3)

    # Make a plot for each metric
    row = 0
    col = 0
    for n, metric in enumerate(metrics):

        # 2 x 2 plot; adjust the row/col label based on metric
        if row == 2:
            col = col + 1
            row = 0

        ax = axes[row, col]
        
        # Set the title
        ax.set_title(metric.upper(), fontsize = 16)

        # Make the plots
        for m, method in enumerate(methods):
            ax.plot(time, data_list[:,m,n], linestyle = '-', linewidth = 1.5, marker = 'o', label = method)
        
        # Make a legend
        ax.legend(loc = 'upper right', fontsize = 16)
        
        # Set the labels
        ax.set_ylabel(metric.upper(), fontsize = 16)
        ax.set_xlabel('Time', fontsize = 16)
        
        
        # Set the tick sizes
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(16)

        row = row + 1
        
    # Save the figure
    filename = '%s_metric_performance_across_time.png'%(model)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to display a set of learning curves
def display_learning_curves(loss, loss_var, methods, metric, model, label, path = './Figures'):
    '''
    Display the (spatially averaged) learning curves with the (spatially averaged) variation (across rotations). There will be 1 learning curve per method
    
    Inputs:
    :param loss: List of loss values corresponding to each method
    :param loss_var: List of loss variations corresponding to each method
    :param methods: List of methods used to identify FD
    :param metric: The name of the metric being plotted
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    epochs = len(loss[0][0])
    
    # Create the figure
    fig, axes = plt.subplots(figsize = [12, 16], nrows = len(methods), ncols = 1, 
                             subplot_kw = {'projection': fig_proj})
    
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
    fig.suptitle(model, y = 0.945, size = 22)
    
    for m, method in enumerate(methods):
        ax = axes[m]
        
        # Plot the ROC curve
        ax.plot(epochs, loss[m][0], 'b', linewidth = 2, label = 'Training')
        ax.plot(epochs, loss[m][1], 'orange', linestyle = '.-', linewidth = 2, label = 'Validation')
        
        ax.legend(loc='top right', fontsize = 20)
        
        # Plot the variation
        ax.fill_between(epochs, loss[m][0]-loss_var[m][0], loss[m][0]+loss_var[m][0], alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        ax.fill_between(epochs, loss[m][1]-loss_var[m][1], loss[m][1]+loss_var[m][1], alpha = 0.5, edgecolor = 'orange', facecolor = 'orange')
        
        # Set the label
        ax.set_ylabel(method, fontsize = 22)
        
        # Set the ticks
        # ax.set_xticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        # ax.set_yticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
        
        # Set the tick size
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(22)
            
    # Save the figure
    filename = '%s_%s_learning_curves.png'%(metric, label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to display a set of feature importance
def display_feature_importance(fimportance, fimportance_var, Nsample, feature_names, methods, model, label, kind = 'importance', path = './Figures'):
    '''
    Display a bargraph showing the (spatially averaged) importance of each feature with corresponding variation
    
    Inputs:
    :param fimportance: List of feature importances
    :param fimportance_var: List of variation in feature importances
    :param Nsample: Int. Number of rotations used in fimportance (i.e., number of samples in the mean of fimportance and std deviation calculations in fimportance_var)
    :param feature_names: List of names for each feature
    :param methods: List of methods used to identify FD
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param kind: String indicating whether feature importance or contribution is being plotted (must be either 'importance' or 'contribution')
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    # Initialize some values for the plot
    if len(fimportance) < 2:
        N = len(fimportance[0])
        width = 0.8
        ind = np.arange(N)
        width_space = 0
    else:
        N = len(fimportance[0])
        width = 0.15
        ind = np.arange(N)
        width_space = width
    
    # Capitilize the first letter in the methods
    methods = [method.title() for method in methods]
    
    # Create the plot
    fig, ax = plt.subplots(figsize = [18, 14])
    
    # Set the title
    ax.set_title(model.upper(), fontsize = 22)
    
    bars = []
    
    # Plot the bars
    for m, method in enumerate(methods):
        # Add the error bars?
        if fimportance_var is None:
            bar = ax.bar(ind+width_space*m, fimportance[m], width = width)
        else:
            # Determine the 95% confidence intervals for the error bars
            t = stats.t.ppf(1-0.05, Nsample-1)
            conf_int = t*fimportance_var[m]/np.sqrt(Nsample-1)

            # Bar plots with 95% confidence interval error bars
            bar = ax.bar(ind+width_space*m, fimportance[m], width = width, yerr = conf_int, error_kw = {'elinewidth':2, 'capsize':6, 'barsabove':True})
        bars.append(bar)
        
    # Add the legend
    ax.legend(bars, methods, fontsize = 22)
        
    # Set the labels
    if kind == 'importance':
        ax.set_ylabel('Feature Importance', fontsize = 22)
    elif kind == 'contribution':
        ax.set_ylabel('Feature Contribution', fontsize = 22)
    
    # Set the ticks
    ax.set_xticks(ind + 2*width_space)#-width/1)
    ax.set_xticklabels(feature_names)
    
    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)
        
    # Save the figure
    filename = '%s_feature_%s.png'%(label, kind)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to display a set of time series
def display_time_series(data_true, data_pred, data_true_var, data_pred_var, time, var_name, model, label, path = './Figures'):
    '''
    Display a time series for a set of prediction and true data
    
    Inputs:
    :param data_true: The data of true values whose time series will be plotted
    :param data_pred: The data of predicted values whose time series will be plotted
    :param data_true_var: The variation of the true values
    :param data_pred_var: The variation of the predicted values
    :param time: The datetimes corresponding to each entry in data_true/data_pred
    :param var_name: The name of the variable whose time series is being plotted
    :param metric: The name of the metric being plotted
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    # Determine how often to plot an errorbar (once a year, assuming numerous years are plotted)
    years = np.array([date.year for date in time])
    Ny = len(np.unique(years))
    Ntime = len(data_true)
    N = int(Ntime/Ny)
    
    # Create the plot
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the %s for the %s'%(var_name, model), fontsize = 22)
    
    # Make the plots color = 'r', linestyle = '-', linewidth = 1, label = 'True values'
    ax.errorbar(time, data_true, yerr = data_true_var, capsize = 3, errorevery = 3*N, 
                color = 'r', linestyle = '-', linewidth = 1.5, label = 'True values')
    ax.errorbar(time, data_pred, yerr = data_pred_var,  capsize = 3, errorevery = 3*N, 
                color = 'b', linestyle = '-.', linewidth = 1.5, label = 'Predicted values')
    
    # Make a legend
    ax.legend(loc = 'upper right', fontsize = 20)
    
    # Set the labels
    ax.set_ylabel(var_name, fontsize = 22)
    ax.set_xlabel('Time', fontsize = 22)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(20)
        
    # Save the figure
    filename = '%s_%s_time_series.png'%(var_name, label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)

#%%
##############################################
# Function to create time series for FD prone regions across the globe
def display_time_series_region(data_true, data_pred, data_true_var, data_pred_var, time, var_name, model, label, one_year = False, path = './Figures'):
    '''
    Display a time series for a set of prediction and true data
    
    Inputs:
    :param data_true: The data of true values whose time series will be plotted
    :param data_pred: The data of predicted values whose time series will be plotted; could be a list of predictands for multiple time series
    :param data_true_var: The variation of the true values
    :param data_pred_var: The variation of the predicted values
    :param time: The datetimes corresponding to each entry in data_true/data_pred
    :param var_name: The name of the variable whose time series is being plotted
    :param metric: The name of the metric being plotted
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param one_year: Boolean indicating whether the plotted time series is for one year (i.e., seasonality)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''

    labels = ['Ada Predicted Labels', 'RNN Predicted Labels']
    colors = ['b', 'k']
    plot_type = 'long_term_ts'

    # Create 1 year of datetimes?
    if one_year:
        year = 1998
        tmp = []
        for month in time:
            tmp.append(datetime(year, month, 1))
            if month == 12:
                year = year + 1

        time = np.array(tmp)
        plot_type = 'seasonality'
                
    
    # Create the plot
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title(label, fontsize = 26)
    
    # Make the plots color = 'r', linestyle = '-', linewidth = 1, label = 'True values'
    ax.plot(time, data_true, 'r-', linewidth = 1.5, label = 'True Values')
    if type(data_pred) is list:
        for data, data_label, color in zip(data_pred, labels, colors):
            ax.plot(time, data, color = color, linestyle = '-.', linewidth = 1.5, label = data_label)
    else:
        ax.plot(time, data_pred, 'b-.', linewidth = 1.5, label = 'Predicted Values')
    #ax.fill_between(time, data_true-data_true_var, data_true+data_true_var, alpha = 0.4, edgecolor = 'r', facecolor = 'r')
    #ax.fill_between(time, data_pred-data_pred_var, data_pred+data_pred_var, alpha = 0.4, edgecolor = 'b', facecolor = 'b')
    # ax.errorbar(time, data_true, yerr = data_true_var, capsize = 3, errorevery = 3*N, 
    #             color = 'r', linestyle = '-', linewidth = 1.5, label = 'True values')
    # ax.errorbar(time, data_pred, yerr = data_pred_var,  capsize = 3, errorevery = 3*N, 
    #             color = 'b', linestyle = '-.', linewidth = 1.5, label = 'Predicted values')
    
    # Make a legend
    ax.legend(loc = 'upper right', fontsize = 26)
    
    # Set the labels
    ax.set_ylabel(var_name, fontsize = 26)
    ax.set_xlabel('Time', fontsize = 26)

    if one_year:
        ax.set_ylim([0, 60])
    else:
        ax.set_ylim([0, 80])
    
    # Set the ticks
    if one_year:
        pass
        ax.xaxis.set_major_formatter(DateFormatter('%b'))
    else:
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(26)
        
    # Save the figure
    filename = '%s_%s_%s_time_series_shading.png'%(plot_type, model, label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################
# Function to create barplots of FD coverage for every month
def fd_coverage_barplots(fd, dates, mask, labels, grow_season = False, years = None, months = None, days = None, path = './', savename_bar = 'tmp.png'):
    '''
    Create a time series showing the average FD coverage in a year, and a bar plot showing the average the how many grid points experience FD onset in each month

    Inputs:
    :param fd: FD data to be plotted. time x lat x lon format
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param mask: Land-sea mask for the dataset (1 = land, 0 = sea)
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param path: Path the figures will be saved to
    :param savename_bar: Filename the barplot will be saved to

    Outputs:
    Two plots of a time series and bar plot will be created and saved
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])

    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    dates_grow = dates[ind]
    years_grow = years[ind]
    months_grow = months[ind]
    days_grow = days[ind]

    # Isolate datetimes for a single year
    ind = np.where(years_grow == 2001)[0]
    one_year = dates_grow[ind]
    year_months = np.array([date.month for date in one_year])

    # Date format for the tick labels
    DateFMT = DateFormatter('%b')

    fd_list = []
    
    # Calculate the average number of rapid intensifications and flash droughts in a year
    for flash_drought in fd:

        # Get the data size
        T, I, J = flash_drought.shape
        
        # Determine the time series
        fd_ts = np.nansum(flash_drought.reshape(T, I*J), axis = -1)
        mask_ts = np.nansum(mask)
        
        # Calculate the average and standard deviation of FD coverage for each pentad in a year
        fd_mean = []
        fd_std = []
        for date in one_year:
            y_ind = np.where((date.month == months_grow) & (date.day == days_grow))[0]
            
            fd_mean.append(np.nansum(fd_ts[y_ind]))
            fd_std.append(np.nanstd(fd_ts[y_ind]))
        
        fd_mean = np.array(fd_mean)
        fd_std = np.array(fd_std)
    
        # Determine the average number of grids for each month
        months_unique = np.unique(months_grow)
        fd_month_mean = []
        fd_month_std = []
        for month in months_unique:
            y_ind = np.where(month == year_months)[0]
            fd_month_mean.append(np.nansum(fd_mean[y_ind])/np.nansum(fd_mean))
            fd_month_std.append(np.nanmean(fd_std[y_ind]))

        fd_list.append(fd_month_mean)

    # Obtain the labels for each month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names_grow = []
    for m, month in enumerate(months_unique):
        month_names_grow.append(month_names[int(month-1)])
    

    N = len(fd)
    if (N%2) == 0:
        width = 0.42
        offset = width*N/4
    else:
        width = 0.16
        offset = width*np.floor(N/2)
        ind = np.arange(N)
    
    # Bar plot of average number of FD in each month
    fig = plt.figure(figsize = [14.4, 8])
    ax = fig.add_subplot(1,1,1)
    
    # Make the bar plot
    m = 0
    for flash_drought, label in zip(fd_list, labels):
        print(flash_drought, np.nansum(flash_drought))
        ax.bar(months_unique+width*m-offset, flash_drought, width = width, edgecolor = 'k', label = label)#, yerr = fd_month_std)
        m = m + 1

    ax.legend(fontsize = 24)
    
    # Set the title
    ax.set_title('Average Month of FD Occurance', fontsize = 24)
    
    # Set the axis labels
    ax.set_xlabel('Time', size = 24)
    ax.set_ylabel('Percentage of FD Occurance', size = 24)
    
    # Set the ticks
    ax.set_xticks(months_unique, month_names_grow)
    #ax.xaxis.set_major_formatter(DateFMT)

    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(24)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename_bar), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################
# Function to plot a global frequency climatology with boxes over FD prone regions
def display_climatology_map_with_boxes(data, lat, lon, borders, title = 'tmp', cbar_label = 'tmp', globe = False, 
                                       cmin = -20, cmax = 80, cint = 1, cticks = np.arange(0, 90, 10), new_colorbar = True, cbar_months = False,
                                       path = './', savename = 'tmp.png'):
    '''
    Create a map plot of FD climatology

    Inputs:
    :param data: FD map to be plotted
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param title: Title of the plot
    :param globe: Boolean indicating whether the data is global
    :param cmin, cmax, cint: The minimum, maximum, and interval of the values in the colorbar
    :param cticks: List or 1D array of the values to make the ticks in the colorbar
    :param new_colorbar: Boolean indicating whether to make/use a new, adjusted colorbar (separate from the raw one)
    :param cbar_months: Boolean indicating whether to label the colorbar with months instead of values
    :param path: Path the figures will be saved to
    :param savename: Filename of the figure to be saved to

    Outputs:
    Map of FD climatology will be made and saved
    '''
    #### Create the Plot ####
    
    # Set colorbar information
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

    # Get the normalized color values
    vmin = 0 if cmin < 0 else cmin
    v = cmin if cmin < 0 else 0
    norm = mcolors.Normalize(vmin = vmin, vmax = cmax)

    # Create a new/adjust the colorbar?
    if np.invert(cbar_months):
        
        # Generate the colors from the orginal color map in range from [0, cmax]
        colors = cmap(np.linspace(1 - (cmax - vmin)/(cmax - v), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
        if new_colorbar:
            colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
        else:
            colors[:1,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
        
        # Create a new colorbar cut from the colors in range [0, cmax.]
        ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
        
        colorsNew = cmap(np.linspace(0, 1, cmap.N))
        if new_colorbar:
            colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
        cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)
    
    # Shapefile information
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
        CountriesReader = shpreader.Reader(CountriesSHP)
    
        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Lonitude and latitude tick information
    if globe:
        lat_int = 15
        lon_int = 40
    else:
        lat_int = 10
        lon_int = 20
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    
    # Create the plots
    fig = plt.figure(figsize = [12, 10])
    
    
    # Flash Drought plot
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)
    
    # Set the flash drought title
    ax.set_title(title, size = 14)
    
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        ax.add_feature(cfeature.STATES)
        ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        ax.coastlines(edgecolor = 'black', zorder = 3)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax.set_yticklabels(LatLabel, fontsize = 14)
    ax.set_xticklabels(LonLabel, fontsize = 14)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the flash drought data
    if globe:
        cs = ax.pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                           cmap = cmap, transform = data_proj, zorder = 1)
    else:
        cs = ax.pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                           cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent to the U.S.
    if globe:
        ax.set_extent([-179, 179, -60, 75])
    else:
        ax.set_extent([-130, -65, 23.5, 48.5])

    for border in borders:
        height = border[1] - border[0]
        width = border[3] - border[2]
        min_lat = border[0]
        if border[2] > 180:
            min_lon = border[2] - 360
        else:
            min_lon = border[2]
        ax.add_patch(mpatches.Rectangle(xy = [min_lon, min_lat], width = width, height = height, 
                                        facecolor = 'none', edgecolor = 'blue', linewidth = 2, transform = fig_proj, zorder = 4))
    
    
    # Set the colorbar size and location
    if globe:
        cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
    else:
        cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])
    
    # Create the colorbar
    if new_colorbar:
        cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')
    else:
        cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, norm = norm, orientation = 'vertical')
    
    # Set the colorbar label
    cbar.ax.set_ylabel(cbar_label, fontsize = 14)
    
    # Set the colorbar ticks
    if cbar_months:
        cbar.set_ticks([0.5, 1.3, 2.3, 3.15, 4.2, 5.05, 6.0, 6.9, 7.8, 8.75, 9.7, 10.7, 11.6])
        cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 14)
    else:
        cbar.set_ticks(cticks)
        cbar.ax.set_yticklabels(cticks, fontsize = 14)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to display the learning curves
def display_learning_curve(lc, lc_var, metric_names, plot_var, model, label, path = './Figures'):
    '''
    Display the learning curve that is stored in fname
    
    :param lc: Ditionary of learning curves (each entry in the list is one metric)
    :param lc_var: Ditionary of variation of learning curves (each entry in the list is one metric)
    :param metric_names: The name of the metric whose learning curve is being plotted
    :param plot_var: Boolean indicating whether to plot the variation in the learning cerves
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    for n, metric in enumerate(metric_names):
        # Create the plot
        fig, ax = plt.subplots(figsize = [12, 8])
    
        # Set the title
        ax.set_title('Learning curve of the %s for the %s'%(metric, model), fontsize = 22)
    
        # Display the loss
        ax.plot(lc['%s'%metric], 'b-', label = 'Training')
        ax.plot(lc['val_%s'%metric], 'r--', label = 'Validation')
        
        if plot_var:
            ax.fill_between(range(len(lc['%s'%metric])), lc['%s'%metric]-lc_var['%s'%metric], lc['%s'%metric]+lc_var['%s'%metric], 
                            alpha = 0.5, edgecolor = 'b', facecolor = 'r')
            ax.fill_between(range(len(lc['val_%s'%metric])), lc['val_%s'%metric]-lc_var['val_%s'%metric], lc['val_%s'%metric]+lc_var['val_%s'%metric], 
                            alpha = 0.5, edgecolor = 'b', facecolor = 'r')
        
        ax.legend(fontsize = 20)
        
        # Set the labels
        ax.set_ylabel(metric, fontsize = 22)
        ax.set_xlabel('Epochs', fontsize = 22)
        
        # Set the tick sizes
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(20)
    
        # Save the figure
        filename = '%s_%s_%s_learning_curve.png'%(model, label, metric)
        plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
        plt.show(block = False)



#%%
##############################################

# Function to display a set of case study maps
def display_case_study_maps(data, lon, lat, time, year_list, method, label, dataset = 'narr', 
                            globe = False, path = './Figures', grow_season = False, pred_type = 'true', years = None, months = None):
    '''
    Create a set of case study maps for a given set of years
    
    Inputs:
    :param data: The data the case study is being made for
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param time: The datetimes corresponding to each entry in data
    :param year_list: List of years corresponding to each year to make a case study for
    :param method: The FD identification method used to collect data
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to
    :param grow_season: A boolean indicating whether data has been subsetted to the growing season
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in time])
        
    if months == None:
        months = np.array([date.month for date in time])


    # Initialize a few values
    T, I, J = data.shape
    if grow_season:
        NMonths = 7
    else:
        NMonths = 12
    case_studies = []

    for y in year_list:
        cs = np.zeros((I, J)) # Initialize a placeholder variable
        for m in range(NMonths):
            if grow_season:
                mon = m + 4
            else:
                mon = m
                
            ind = np.where( (years == y) & (months == mon) )[0]
            cs = np.where(((np.nansum(data[ind,:,:], axis = 0) != 0 ) & (cs == 0)), mon, cs) # Points where there prediction for the 
                                                                                             # month is nonzero (FD is predicted) and 
                                                                                             # cs does not have a value already, are given a value of m. 
                                                                                             # cs is left alone otherwise.
        # Remove sea values
        # cs = apply_mask(cs, mask)
        case_studies.append(cs)

    

    # Set colorbar information
    # cmin = 0; cmax = 12; cint = 1
    cmin = 3; cmax = 10; cint = 1
    clevs = np.arange(cmin, cmax + cint, cint)
    norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

    # Shapefile information
    if np.invert(globe):
        # ShapeName = 'Admin_1_states_provinces_lakes_shp'
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60

    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)

    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()


    # Create a figure for each case study/year
    for y, year in enumerate(year_list):
        # Create the plots
        fig = plt.figure(figsize = [12, 10])


        # Flash Drought plot
        ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

        # Set the flash drought title
        ax.set_title('Flash Drought for %s'%year, size = 22)

        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
        if np.invert(globe):
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            ax.add_feature(cfeature.STATES)
            ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
            ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
        else:
            # Ocean covers and "masks" data outside the U.S.
            ax.coastlines(edgecolor = 'black', zorder = 3)
            ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

        # Adjust the ticks
        ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
        ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

        ax.set_yticklabels(LatLabel, fontsize = 20)
        ax.set_xticklabels(LonLabel, fontsize = 20)

        ax.xaxis.set_major_formatter(LonFormatter)
        ax.yaxis.set_major_formatter(LatFormatter)

        # Plot the flash drought data
        cs = ax.pcolormesh(lon, lat, case_studies[y], vmin = cmin, vmax = cmax,
                           cmap = cmap, transform = data_proj, zorder = 1)

        # Set the map extent to the U.S.
        if np.invert(globe):
            ax.set_extent([-130, -65, 23.5, 48.5])
        else:
            ax.set_extent([-179, 179, -65, 80])

        
        # Set the colorbar size and location
        if np.invert(globe):
            cbax = fig.add_axes([0.925, 0.30, 0.020, 0.40])
        else:
            cbax = fig.add_axes([0.925, 0.32, 0.020, 0.36])
        cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, norm = norm, orientation = 'vertical')

        # Set the colorbar ticks
        cbar.set_ticks([3.5, 4.3, 5.2, 6.05, 6.9, 7.75, 8.7, 9.5])
        cbar.ax.set_yticklabels(['No FD', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct'], fontsize = 22)
        for i in cbar.ax.yaxis.get_ticklabels():
            i.set_size(20)
        
        # Set the colorbar ticks
        # cbar.set_ticks(np.arange(0, 12+1, 1))
        # cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 16)
        # cbar.set_ticks(np.arange(1, 11+1, 1.48))
        

        # Save the figure
        plt.savefig('%s/%s_%s_%s_case_study_%s_%s.png'%(path,label,method,year, pred_type, dataset), bbox_inches = 'tight')
        plt.show(block = False)


def display_case_study_maps_full(data_true, data_pred, data_attribution, feature_names, lon, lat, time, year_list, methods, label, dataset = 'narr', 
                                 globe = False, path = './', grow_season = False, years = None, months = None):
    '''
    Create a set of case study maps for a given set of years
    
    Inputs:
    :param data_true: The data of true labels the case study is being made for
    :param data_pred: The data of predicted labels the case study is being made for
    :param data_attribution: The feature attributions for the predictions
    :param feature_names: List of strings giving the names of the features
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param time: The datetimes corresponding to each entry in data
    :param year_list: List of years corresponding to each year to make a case study for
    :param methods: List of FD identification methods used to collect data
    :param label: A label used to distinguish the experiment
    :param dataset: The type of ML dataset that is being trained (must br train, valid, or test)
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path to the directory the maps will be saved to
    :param grow_season: A boolean indicating whether data has been subsetted to the growing season
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    Figures (one for each case study year) showing the onset month for the case study for each FD identification method, 
    for true and predicted labels, and feature attribution for each pentad will be made and saved
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in time])
        
    if months == None:
        months = np.array([date.month for date in time])


    # Initialize a few values
    T, I, J = data_true[0].shape
    if grow_season:
        NMonths = 7
    else:
        NMonths = 12

    case_studies_true = []
    case_studies_pred = []

    for y in year_list:
        cs_true = np.zeros((I, J, len(methods))) # Initialize a placeholder variable
        cs_pred = np.zeros((I, J, len(methods))) # Initialize a placeholder variable
        
        if globe:
            if (y == 2015) | (y == 2018):
                NH = False
            else:
                NH = True
        else:
            NH = True

        # Determine the onset month of FD for the case study year
        for m in range(NMonths):
            if grow_season:
                if NH:
                    mon = m + 4
                else:
                    mon = (m + 9) if ((m+9) <= 12) else (m - 3)
            else:
                mon = m

            for method in range(len(methods)):
                ind = np.where( (years == y) & (months == mon) )[0]
                cs_true[:,:,method] = np.where(((np.nansum(data_true[method][ind,:,:], axis = 0) != 0 ) & (cs_true[:,:,method] == 0)), mon, cs_true[:,:,method])
                # Points where there prediction for the month is nonzero (FD is predicted) and 
                # cs does not have a value already, are given a value of m. 
                # cs is left alone otherwise.

                cs_pred[:,:,method] = np.where(((np.nansum(data_pred[method][ind,:,:], axis = 0) != 0 ) & (cs_pred[:,:,method] == 0)), mon, cs_pred[:,:,method])

        # Remove sea values
        # cs = apply_mask(cs, mask)
        
        case_studies_true.append(cs_true)
        case_studies_pred.append(cs_pred)

    

    # Set colorbar information
    # cmin = 0; cmax = 12; cint = 1
    cmin = 3; cmax = 10; cint = 1
    clevs = np.arange(cmin, cmax + cint, cint)
    norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'Paired', lut = nlevs)

    # Get the colors for the colormap
    colors = cmap.colors
    print(colors)

    colors[0,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

    # Shapefile information
    if np.invert(globe):
        # ShapeName = 'Admin_1_states_provinces_lakes_shp'
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 10
    else:
        lat_int = 30
        lon_int = 60

    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)

    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()


    # Create a figure for each case study/year
    for y, year in enumerate(year_list):
        y_ind = np.where(year == years)[0]

        # Determine domain of focus for the specific year
        if globe:
            if year == 2001:
                # India
                wspace = -0.68
                wspace_sub = -0.87
                
                lon_min = 62
                lon_max = 100
                lat_min = 0
                lat_max = 40
                month_names = ['No FD', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
                time_offset = timedelta(days = 0)
                
            elif year == 2010:
                # Russia
                wspace = -0.55
                wspace_sub = -0.74
                
                lon_min = 28
                lon_max = 52
                lat_min = 44
                lat_max = 59

                month_names = ['No FD', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
                time_offset = timedelta(days = 0)
        
            elif year == 2015:
                # South Africa
                # wspace = -0.66
                # wspace_sub = -0.84
                
                # lon_min = 10
                # lon_max = 40
                # lat_min = -35
                # lat_max = -10

                # month_names = ['No FD', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
                # time_offset = timedelta(days = 153)

                # Amazon
                wspace = -0.59
                wspace_sub = -0.78
                
                lon_min = -76
                lon_max = -39
                lat_min = -24
                lat_max = 0

                month_names = ['No FD', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
                time_offset = timedelta(days = 153)
                
            elif year == 2016:
                # Eastern Africa
                wspace = -0.66
                wspace_sub = -0.84
                
                lon_min = 33
                lon_max = 55
                lat_min = 0
                lat_max = 20

                month_names = ['No FD', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
                time_offset = timedelta(days = 0)
                
            elif year == 2018:
                # Southeast Australia
                wspace = -0.71
                wspace_sub = -0.89
                
                lon_min = 142
                lon_max = 154
                lat_min = -42
                lat_max = -28
                
                month_names = ['No FD', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
                time_offset = timedelta(days = 153)
        else:
            month_names = ['No FD', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
            time_offset = timedelta(days = 0)
        
            if year == 1988:
                wspace = -0.58
                wspace_sub = -0.77
                lon_min = -115
                lon_max = -80
                lat_min = 30
                lat_max = 50
                
            elif year == 2000:
                wspace = -0.615
                wspace_sub = -0.80
                lon_min = -105
                lon_max = -79
                lat_min = 28
                lat_max = 45
    
            elif year == 2003:
                wspace = -0.67
                wspace_sub = -0.85
                lon_min = -107
                lon_max = -83
                lat_min = 30
                lat_max = 50
                
            elif year == 2011:
                wspace = -0.55
                wspace_sub = -0.74
                lon_min = -110
                lon_max = -78
                lat_min = 25
                lat_max = 42
                
            elif year == 2012:
                wspace = -0.68
                wspace_sub = -0.855
                lon_min = -105
                lon_max = -82
                lat_min = 30
                lat_max = 50
                
            elif year == 2017:
                wspace = -0.415
                wspace_sub = -0.60
                lon_min = -118
                lon_max = -95
                lat_min = 40
                lat_max = 50
                
            elif year == 2019:
                wspace = -0.48
                wspace_sub = -0.67
                lon_min = -107
                lon_max = -74
                lat_min = 25
                lat_max = 40
                
            else:
                wspace = 0.1
                hspace = 0.2
                lat_min = np.nanmin(lat)
                lat_max = np.nanmax(lat)
                lon_min = np.nanmin(lon)
                lon_max = np.nanmax(lon)

        # Subset the attribution so only the attribution for the domain is displayed
        # data_attribution_sub = []
        # for m in range(len(methods)):
        #     tmp_value = []
        #     for fn in range(len(feature_names)):
        #         tmp, _, _ = subset_data(data_attribution[m][:,:,:,fn], lat, lon, 
        #                                 LatMin = lat_min, LatMax = lat_max, 
        #                                 LonMin = lon_min, LonMax = lon_max)
        #         tmp_value.append(tmp)
                
        #     tmp_value = np.stack(tmp_value, axis = -1)
        #     data_attribution_sub.append(tmp_value)
            
        
        # Create the plots
        #fig, axes = plt.subplots(figsize = [36, 10*len(methods)], nrows = len(methods), ncols = 3, 
        #                         subplot_kw = {'projection': fig_proj})

        # Create a set of nested grid specs so the space between the maps and the time series can be adjusted (only way found so far to display all three parts properly)
        fig = plt.figure(figsize = [54, 15*len(methods)])
        gs = fig.add_gridspec(2*len(methods), 3, left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = wspace, hspace = 0.12, width_ratios = [3, 3, 1])
        gs1 = gridspec.GridSpecFromSubplotSpec(2*len(methods), 2, subplot_spec = gs[:,:2], wspace = wspace_sub)

        #plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
        fig.suptitle(year, y = 0.985, size = 30)

        for m, method in enumerate(methods):
            #ax1 = axes[m,0]; ax2 = axes[m,1]; ax3 = axes[m,2]

            #ax1.remove()
            ax1 = fig.add_subplot(gs1[m, 0], projection = fig_proj)

            # Set the flash drought title
            if m == 0:
                ax1.set_title('True FD', size = 30)
    
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
            if np.invert(globe):
                # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
                ax1.add_feature(cfeature.STATES)
                ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
                ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
            else:
                # Ocean covers and "masks" data outside the U.S.
                ax1.coastlines(edgecolor = 'black', zorder = 3)
                ax1.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
            # Adjust the ticks
            ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
            ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())

            ax1.set_yticklabels(LatLabel, fontsize = 30)

            if m == len(methods)-1:
                ax1.set_xticklabels(LonLabel, fontsize = 30)
            else:
                ax1.tick_params(labelbottom = False)
    
            ax1.xaxis.set_major_formatter(LonFormatter)
            ax1.yaxis.set_major_formatter(LatFormatter)

            # Set the label
            ax1.set_ylabel(method, fontsize = 30, labelpad = 35.0, rotation = 0)
    
            # Plot the flash drought data
            cs = ax1.pcolormesh(lon, lat, case_studies_true[y][:,:,m], vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)
    
            # Set the map extent to the U.S.
            ax1.set_extent([lon_min, lon_max, lat_min, lat_max])



            ax2 = fig.add_subplot(gs1[m, 1], projection = fig_proj)
            
            # Set the flash drought title
            if m == 0:
                ax2.set_title('Predicted FD', size = 30)
    
            # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
            ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
            if np.invert(globe):
                # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
                ax2.add_feature(cfeature.STATES)
                ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
                ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
            else:
                # Ocean covers and "masks" data outside the U.S.
                ax2.coastlines(edgecolor = 'black', zorder = 3)
                ax2.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
            # Adjust the ticks
            ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
            ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())

            ax2.tick_params(labelleft = False)
            if m == len(methods)-1:
                ax2.set_xticklabels(LonLabel, fontsize = 30)
            else:
                ax2.tick_params(labelbottom = False)
    
            ax2.xaxis.set_major_formatter(LonFormatter)
            ax2.yaxis.set_major_formatter(LatFormatter)
    
            # Plot the flash drought data
            cs = ax2.pcolormesh(lon, lat, case_studies_pred[y][:,:,m], vmin = cmin, vmax = cmax,
                                cmap = cmap, transform = data_proj, zorder = 1)
    
            # Set the map extent to the U.S.
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])

            # Average the attribution over space for the time series plot
            # T, I, J, Nf = data_attribution_sub[m].shape
            # ts = np.nanmean(data_attribution_sub[m].reshape(T, I*J, Nf), axis = 1)

            ax3 = fig.add_subplot(gs[m, 2])#*len(methods))

            # Add the title for the feature attribution
            if m == 0:
                Nfeatures = data_attribution[0].shape[-1]
                if Nfeatures > 3:
                    ax3.set_title('Average Attribution (SHAP Value)', fontsize = 30)
                else:
                    ax3.set_title('Areal Coverage (%)', fontsize = 30)

            # Plot the attribution data
            for fn, fname in enumerate(feature_names):
                ax3.plot(time[y_ind]+time_offset, data_attribution[m][y,:,fn], linestyle = '-', linewidth = 1.5, marker = 'o', label = fname)
        
            # Make a legend
            ax3.legend(loc = 'upper right', fontsize = 20)
            
            # Set the labels
            if m == (len(methods)-1):
                ax3.set_xlabel('Time', fontsize = 30)

            # Set the axis
            ax3.xaxis.set_major_formatter(DateFormatter('%b'))
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Round the labels to 2 decimals to fit better
            
            # Set the tick sizes
            for i in ax3.xaxis.get_ticklabels() + ax3.yaxis.get_ticklabels():
                i.set_size(20)
    
            
        # Set the colorbar size and location
        if np.invert(globe):
            cbax = fig.add_axes([0.915, 0.60, 0.010, 0.20])
        else:
            cbax = fig.add_axes([0.925, 0.32, 0.020, 0.36])
        cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, norm = norm, orientation = 'vertical')

        # Set the colorbar ticks
        cbar.set_ticks([3.5, 4.3, 5.2, 6.05, 6.9, 7.75, 8.7, 9.5])
        cbar.ax.set_yticklabels(month_names, fontsize = 22)
        for i in cbar.ax.yaxis.get_ticklabels():
            i.set_size(30)
        
        # Set the colorbar ticks
        # cbar.set_ticks(np.arange(0, 12+1, 1))
        # cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 16)
        # cbar.set_ticks(np.arange(1, 11+1, 1.48))
        

        # Save the figure
        plt.savefig('%s/%s_%s_case_study_%s.png'%(path,label,year,dataset), bbox_inches = 'tight')
        plt.show(block = False)
    
#%%
##############################################
    
# Function to generate a generic time series
def make_map(var, lat, lon, var_name = 'tmp', model = 'narr', globe = False, path = './Figures/', savename = 'timeseries.png'):
    '''
    Create a save a generic map of var
    
    Inputs:
    :param var: 2D array of the variable to be mapped
    :param lat: 2D array of latitudes
    :param lon: 2D array of longitudes
    :param var_name: String. Name of the variable being plotted
    :param model: String. Name of the reanalysis model the data comes from
    :param globe: Bool. Indicates whether the map will be a global one or not (non-global maps are fixed on the U.S.)
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    
    # Set colorbar information
    cmin = 0; cmax = 1; cint = 0.05
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    
    cname = 'Reds'
    cmap  = plt.get_cmap(name = cname, lut = nlevs)
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

        CountriesReader = shpreader.Reader(CountriesSHP)

        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
        
    # Create the plots
    fig = plt.figure(figsize = [12, 10])


    # Flash Drought plot
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    # Set the flash drought title
    ax.set_title('%s for %s'%(var_name,model), size = 22)

    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax.add_feature(cfeature.STATES)
        ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        # Ocean covers and "masks" data outside the U.S.
        ax.coastlines(edgecolor = 'black', zorder = 3)
        ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)

    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

    ax.set_yticklabels(LatLabel, fontsize = 20)
    ax.set_xticklabels(LonLabel, fontsize = 20)

    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)

    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, var, vmin = cmin, vmax = cmax,
                       cmap = cmap, transform = data_proj, zorder = 1)

    # Set the map extent to the U.S.
    if np.invert(globe):
        ax.set_extent([-130, -65, 23.5, 48.5])
    else:
        ax.set_extent([-179, 179, -65, 80])


    # Set the colorbar size and location
    if np.invert(globe):
        cbax = fig.add_axes([0.925, 0.30, 0.020, 0.40])
    else:
        cbax = fig.add_axes([0.925, 0.32, 0.020, 0.36])
    cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, orientation = 'vertical')
    cbar.ax.set_ylabel(var_name, fontsize = 22)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(20)
        
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


# Function to generate a generic map
def make_timeseries(var, time, var_name = 'tmp', model = 'narr', path = './Figures/', savename = 'timeseries.png'):
    '''
    Create and save a generic time series plot for some variable var
    
    Inputs:
    :param var: Vector/1D array of the variable to be plotted
    :param time: 1D array of datetimes of time stamps for var
    :param var_name: String. Name of the variable being plotted
    :param model: String. Name of the reanalysis model the data comes from
    :param path: String. Output path to where the figure will be saved
    :param savename: String. Filename the figure will be saved as
    '''
    # Plot the figure
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the %s for the %s'%(var_name, model), fontsize = 22)
    
    # Make the plots
    ax.plot(time, var, 'r-', linewidth = 2)

    
    # Set the labels
    ax.set_ylabel(var_name, fontsize = 22)
    ax.set_xlabel('Time', fontsize = 22)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(20)
        
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
# Function to calculate the threat score
def display_threat_score(true, pred, lat, lon, time, mask, model = 'narr', label = 'christian', globe = False, path = './Figures/'):
    '''
    Calculate and display the threatscore and equitable threat score in space and in time. Note the threat score is also called the 
    critical success index (CSI).
    
    Inputs:
    :param true: 3D array of true values
    :param pred: 3D array of predicted values
    :param lat: Gridded latitude values corresponding to true/pred
    :param lon: Gridded longitude values corresponding to true/pred
    :param time: The datetimes corresponding to each entry in true/pred
    :param mask: The land-sea mask for true/pred
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    
    # Determine the hits, misses, and false alarms
    hits = np.where((true == 1) & (pred == 1), 1, 0)
    misses = np.where((true == 1) & (pred == 0), 1, 0)
    false_alarms = np.where((true == 0) & (pred == 1), 1, 0)
    correct_negatives = np.where((true == 0) & (pred == 0), 1, 0)
    
    # Initialize variables
    T, I, J = true.shape
    
    ts_s = np.ones((T)) * np.nan
    ets_s = np.ones((T)) * np.nan
    
    ts_t = np.ones((I, J)) * np.nan
    ets_t = np.ones((I, J)) * np.nan
    
    total_s = np.nansum(mask) # Total number of data points in space
    total_t = T # Total number of data points in time
    
    # Calculate the spatial number of hits/misses/false alarms in space
    hits_s = np.nansum(hits, axis = -1)
    hits_s = np.nansum(hits_s, axis = -1)

    misses_s = np.nansum(misses, axis = -1)
    misses_s = np.nansum(misses_s, axis = -1)

    false_alarms_s = np.nansum(false_alarms, axis = -1)
    false_alarms_s = np.nansum(false_alarms_s, axis = -1)
    

    # Calculate the total threat score in space for each time stamp
    ts_s = hits_s/(hits_s + misses_s + false_alarms_s)
    
    # Calculate the total equitable threat score in space for each time stamp
    hits_rand_s = (hits_s + misses_s)*(hits_s + false_alarms_s)/total_s
    
    ets_s = (hits_s - hits_rand_s)/(hits_s + misses_s + false_alarms_s - hits_rand_s)
    
    
    
    # Determine the equitable threat score in time
    hits_t = np.nansum(hits, axis = 0)
    misses_t = np.nansum(misses, axis = 0)
    false_alarms_t = np.nansum(false_alarms, axis = 0)
    
    # Calculate the total threat score in time for each grid point
    ts_t = hits_t/(hits_t + misses_t + false_alarms_t)
    
    # Calculate the total equitable threat score in time for each grid point
    hits_rand_t = (hits_t + misses_t)*(hits_t + false_alarms_t)/total_t
    
    ets_t = (hits_t - hits_rand_t)/(hits_t + misses_t + false_alarms_t - hits_rand_t)
        

        
    # Create and save a threat scores time series
    filename = 'threat_score_%s_time_series.png'%(label)
    make_timeseries(ts_s, time, var_name = 'Threat Score', model = model, path = path, savename = filename)
    
        
    # Create and save an equitable threat scores time series
    filename = 'equitable_threat_score_%s_time_series.png'%(label)
    make_timeseries(ets_s, time, var_name = 'Equitable Threat Score', model = model, path = path, savename = filename)

    
    # Plot the threat scores in space and save the plot
    filename = 'threat_score_%s_map.png'%(label)
    make_map(ts_t, lat, lon, var_name = 'Threat Score', model = model, globe = globe, path = path, savename = filename)
        

    # Plot the equitable threat scores in space and save the plot
    filename = 'equitable_threat_score_%s_map.png'%(label)
    make_map(ets_t, lat, lon, var_name = 'Equitable Threat Score', model = model, globe = globe, path = path, savename = filename)
    
    # Overall performance
    ts_list = []
    ets_list = []
    true_skill_list = []
    
    years = np.array([date.year for date in time])
    total = total_s * np.nansum(years == 2000) # total points across 1 year
    
    for year in np.unique(years):
        ind = np.where(year == years)[0]
        hits_tmp = np.nansum(hits[ind,:,:])
        misses_tmp = np.nansum(misses[ind,:,:])
        false_alarms_tmp = np.nansum(false_alarms[ind,:,:])
        correct_negatives_tmp = np.nansum(correct_negatives[ind,:,:])
        
        hits_rand = (hits_tmp + misses_tmp)*(hits_tmp + false_alarms_tmp)/total
        
        ts_list.append(hits_tmp/(hits_tmp + misses_tmp))
        ets_list.append((hits_tmp - hits_rand)/(hits_tmp + misses_tmp + false_alarms_tmp - hits_rand))
        true_skill_list.append(hits_tmp/(hits_tmp + misses_tmp) - false_alarms_tmp/(false_alarms_tmp + correct_negatives_tmp))
    
    Nsample = len(ts_list)
    t = stats.t.ppf(1-0.05, Nsample-1)
    
    ts = np.nanmean(ts_list)
    ts_std = np.nanstd(ts_list)
    ts_interval = t*ts_std/np.sqrt(Nsample-1)
    
    ets = np.nanmean(ets_list)
    ets_std = np.nanstd(ets_list)
    ets_interval = t*ets_std/np.sqrt(Nsample-1)
    
    true_skill = np.nanmean(true_skill_list)
    true_skill_std = np.nanstd(true_skill_list)
    true_skill_interval = t*true_skill_std/np.sqrt(Nsample-1)
    
    #hits = np.nansum(hits)
    #misses = np.nansum(misses)
    #false_alarms = np.nansum(false_alarms)
    #correct_negatives = np.nansum(correct_negatives)
    
    #ts = hits/(hits + misses + false_alarms)
    
    #total = total_s * total_t # All points
    #hits_rand = (hits + misses)*(hits + false_alarms)/total
    #ets = (hits - hits_rand)/(hits + misses + false_alarms - hits_rand)
    
    #true_skill = hits/(hits + misses) - false_alarms/(false_alarms + correct_negatives)
    
    print('Overall CSI/Threat score performance for %s is %4.2f (%f)'%(label, ts, ts_interval))
    print('Overall Gilbert skill score/Equitable threat score performance for %s is %4.2f (%f)'%(label, ets, ets_interval))
    print('Overall True skill statistic performance for %s is %4.2f (%f)'%(label, true_skill, true_skill_interval))

    
    
# Function to calculate the threat score
def display_far(true, pred, lat, lon, time, model = 'narr', label = 'christian', globe = False, path = './Figures/'):
    '''
    Calculate and display the false alarm ratio (FAR) in space and in time
    
    Inputs:
    :param true: 3D array of true values
    :param pred: 3D array of predicted values
    :param lat: Gridded latitude values corresponding to true/pred
    :param lon: Gridded longitude values corresponding to true/pred
    :param time: The datetimes corresponding to each entry in true/pred
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    
    # Determine the hits, misses, and false alarms
    hits = np.where((true == 1) & (pred == 1), 1, 0)
    misses = np.where((true == 1) & (pred == 0), 1, 0)
    false_alarms = np.where((true == 0) & (pred == 1), 1, 0)
    
    # Initialize variables
    T, I, J = true.shape
    
    far_s = np.ones((T)) * np.nan
    
    far_t = np.ones((I, J)) * np.nan
    
    # Calculate the spatial number of hits/misses/false alarms in space
    hits_s = np.nansum(hits, axis = -1)
    hits_s = np.nansum(hits_s, axis = -1)

    misses_s = np.nansum(misses, axis = -1)
    misses_s = np.nansum(misses_s, axis = -1)

    false_alarms_s = np.nansum(false_alarms, axis = -1)
    false_alarms_s = np.nansum(false_alarms_s, axis = -1)

    # Calculate the FAR score in space for each time stamp
    far_s = false_alarms_s/(hits_s + false_alarms_s)
    
    
    # Determine the equitable threat score in time
    hits_t = np.nansum(hits, axis = 0)
    misses_t = np.nansum(misses, axis = 0)
    false_alarms_t = np.nansum(false_alarms, axis = 0)
    
    # Calculate the FAR score in time for each grid point
    far_t = false_alarms_t/(hits_t + false_alarms_t)
        
        
    # Create and save a FAR time series
    filename = 'far_%s_time_series.png'%(label)
    make_timeseries(far_s, time, var_name = 'FAR', model = model, path = path, savename = filename)
    
        
    # Plot the FAR in space and save the plot
    filename = 'far_%s_map.png'%(label)
    make_map(far_t, lat, lon, var_name = 'FAR', model = model, globe = globe, path = path, savename = filename)
    
    # Overall performance
    far_list = []
    years = np.array([date.year for date in time])
    for year in np.unique(years):
        ind = np.where(year == years)[0]
        hits_tmp = np.nansum(hits[ind,:,:])
        misses_tmp = np.nansum(misses[ind,:,:])
        false_alarms_tmp = np.nansum(false_alarms[ind,:,:])
        far_list.append(false_alarms_tmp/(hits_tmp + false_alarms_tmp))
    
    Nsample = len(far_list)
    t = stats.t.ppf(1-0.05, Nsample-1)
    
    far = np.nanmean(far_list)
    far_std = np.nanstd(far_list)
    far_interval = t*far_std/np.sqrt(Nsample-1)
    
    #hits = np.nansum(hits)
    #misses = np.nansum(misses)
    #false_alarms = np.nansum(false_alarms)
    #far = false_alarms/(hits + false_alarms)
    
    print('Overall FAR performance for %s is %4.2f (%f)'%(label, far, far_interval))

    
    
# Function to calculate the threat score
def display_pod(true, pred, lat, lon, time, model = 'narr', label = 'christian', globe = False, path = './Figures/'):
    '''
    Calculate and display the probability of detection (POD) in space and in time
    
    Inputs:
    :param true: 3D array of true values
    :param pred: 3D array of predicted values
    :param lat: Gridded latitude values corresponding to true/pred
    :param lon: Gridded longitude values corresponding to true/pred
    :param time: The datetimes corresponding to each entry in true/pred
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    
    # Determine the hits, misses, and false alarms
    hits = np.where((true == 1) & (pred == 1), 1, 0)
    misses = np.where((true == 1) & (pred == 0), 1, 0)
    false_alarms = np.where((true == 0) & (pred == 1), 1, 0)
    
    # Initialize variables
    T, I, J = true.shape
    
    pod_s = np.ones((T)) * np.nan
    
    pod_t = np.ones((I, J)) * np.nan
    
    
    # Calculate the spatial number of hits/misses/false alarms in space
    hits_s = np.nansum(hits, axis = -1)
    hits_s = np.nansum(hits_s, axis = -1)

    misses_s = np.nansum(misses, axis = -1)
    misses_s = np.nansum(misses_s, axis = -1)

    false_alarms_s = np.nansum(false_alarms, axis = -1)
    false_alarms_s = np.nansum(false_alarms_s, axis = -1)

    # Calculate the POD score in space for each time stamp
    pod_s = hits_s/(hits_s + misses_s)
    
    
    # Determine the equitable threat score in time
    hits_t = np.nansum(hits, axis = 0)
    misses_t = np.nansum(misses, axis = 0)
    false_alarms_t = np.nansum(false_alarms, axis = 0)
    
    # Calculate the POD score in time for each grid point
    pod_t = hits_t/(hits_t + misses_t)
        

    # Create and save a POD time series
    filename = 'pod_%s_time_series.png'%(label)
    make_timeseries(pod_s, time, var_name = 'POD', model = model, path = path, savename = filename)
    
        
    # Plot the POD in space and save the plot
    filename = 'pod_%s_map.png'%(label)
    make_map(pod_t, lat, lon, var_name = 'POD', model = model, globe = globe, path = path, savename = filename)
    
    # Overall performance
    pod_list = []
    years = np.array([date.year for date in time])
    for year in np.unique(years):
        ind = np.where(year == years)[0]
        hits_tmp = np.nansum(hits[ind,:,:])
        misses_tmp = np.nansum(misses[ind,:,:])
        false_alarms_tmp = np.nansum(false_alarms[ind,:,:])
        pod_list.append(hits_tmp/(hits_tmp + misses_tmp))
    
    Nsample = len(pod_list)
    t = stats.t.ppf(1-0.05, Nsample-1)
    
    pod = np.nanmean(pod_list)
    pod_std = np.nanstd(pod_list)
    pod_interval = t*pod_std/np.sqrt(Nsample-1)
    #hits = np.nansum(hits)
    #misses = np.nansum(misses)
    #false_alarms = np.nansum(false_alarms)
    #pod = hits/(hits + misses)
    
    print('Overall POD performance for %s is %4.2f (%f)'%(label, pod, pod_interval))


#%%
##############################################

# Function for making a trio of confusion matrix maps
def display_confusion_matrix_maps(true, pred, lat, lon, method, globe = False, path = './', savename = 'tmp.png'):
    '''
    Display a map of confusion matrix inputs to see how they vary across space

    Inputs:
    :param true: True labels in a time x lat x lon format
    :param pred: Predicted labels in a time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param method: FD identification method used to get the labels
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path to the directory the maps will be saved to
    :param savename: Filename the figure will be saved to

    Outputs:
    A figure showing three maps of confusion matrix values will be created and saved
    '''

    # Initialize some variables
    T, I, J = true.shape
    
    YoN = np.zeros((T, I*J)) * np.nan # YoN stands for Yes or No, that is correct identification or false positive/negative.
    
    FP = np.zeros((T, I*J)) * np.nan # NP stands for false positive.
    
    FN = np.zeros((T, I*J)) * np.nan # FN stands for false negative. 

    # Reshape values
    true = true.reshape(T, I*J, order = 'F')
    pred = pred.reshape(T, I*J, order = 'F')
    
    for ij in range(I*J):
        for t in range(T):
            # Find the spots where the true and predicted labels agree
            if ((true[t,ij] == 1) & (pred[t,ij] == 1)) | ((true[t,ij] == 0) & (pred[t,ij] == 0)):
                YoN[t,ij] = 1
            # For the remaining points (where they do not agree) set to 0
            else:
                YoN[t,ij] = 0
                
            # Find specifically false positives (false alarms)
            if (true[t,ij] == 0) & (pred[t,ij] == 1):
                FP[t,ij] = 1
            # Set remaining values to 0
            else:
                FP[t,ij] = 0
                
            # Find the false negatives (misses)
            if (true[t,ij] == 1) & (pred[t,ij] == 0):
                FN[t,ij] = 1
            # Set remaining values to 0
            else:
                FN[t,ij] = 0
    
    # Reshape the data back to their original state
    true = true.reshape(T, I, J, order = 'F')
    pred = pred.reshape(T, I, J, order = 'F')
    
    YoN = YoN.reshape(T, I, J, order = 'F')
    
    FP = FP.reshape(T, I, J, order = 'F')
    FN = FN.reshape(T, I, J, order = 'F')
    
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60
    
    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)
    
    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()
    
    # Colorbar information
    cmax = 100; cmin = 0; cint = 5
    clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
    nlevs = len(clevs) - 1
    cmap = plt.get_cmap(name = 'Reds', lut = nlevs)
    
    # Additional shapefiles for removing non-US countries
    if globe:
        pass
    else:
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
        
        CountriesReader = shpreader.Reader(CountriesSHP)
        
        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Projection informatino
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Create the plot
    fig, axes = plt.subplots(figsize = [12, 20], nrows = 1, ncols = 3, 
                                 subplot_kw = {'projection': ccrs.PlateCarree()})
    
    # Adjust some figure parameters
    plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.10, hspace = -0.73)
    
    ax1 = axes[0]; ax2 = axes[1]; ax3 = axes[2]
    
    # Set the main figure title
    fig.suptitle('Spatial Distribution of Confusion Matrix Entries', fontsize = 22, y = 0.590)
    
    
    
    # Left plot; Correct identification frequency
    
    # Add features
    ax1.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax1.add_feature(cfeature.STATES)
        ax1.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax1.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        # Ocean covers and "masks" data outside the U.S.
        ax1.coastlines(edgecolor = 'black', zorder = 3)
        ax1.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
    # Set a local title
    ax1.set_title('Frequency of Agreement', fontsize = 16)
    
    # Set the tick information
    ax1.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax1.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax1.set_yticklabels(LatLabel, fontsize = 16)
    ax1.set_xticklabels(LonLabel, fontsize = 16)
    
    ax1.xaxis.set_major_formatter(LonFormatter)
    ax1.yaxis.set_major_formatter(LatFormatter)
    
    ax1.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                    labelsize = 16, bottom = True, top = True, left = True,
                    right = True, labelbottom = True, labeltop = False,
                    labelleft = True, labelright = False)

    
    ax1.set_ylabel(method, fontsize = 16, labelpad = 35.0, rotation = 0)
    
    # Plot the data
    cs = ax1.pcolormesh(lon, lat, np.nanmean(YoN[:,:,:], axis = 0)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())
    
    # Set the map extent over the U.S.
    if np.invert(globe):
        ax1.set_extent([-130, -65, 23.5, 48.5])
    else:
        ax1.set_extent([-179, 179, -60, 75])
    
    
    
    
    # Center plot: False positive frequency
    
    # Add features
    ax2.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax2.add_feature(cfeature.STATES)
        ax2.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax2.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        # Ocean covers and "masks" data outside the U.S.
        ax2.coastlines(edgecolor = 'black', zorder = 3)
        ax2.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
    # Set a local title
    ax2.set_title('Frequency of False Positives', fontsize = 16)
    
    # Set the tick information
    ax2.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax2.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax2.set_yticklabels(LatLabel, fontsize = 16)
    ax2.set_xticklabels(LonLabel, fontsize = 16)
    
    ax2.xaxis.set_major_formatter(LonFormatter)
    ax2.yaxis.set_major_formatter(LatFormatter)
    
    ax2.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                    labelsize = 16, bottom = True, top = True, left = True,
                    right = True, labelbottom = True, labeltop = False,
                    labelleft = False, labelright = False)
    
    # Plot the data
    cs = ax2.pcolormesh(lon, lat, np.nanmean(FP[:,:,:], axis = 0)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())
    
    # Set the map extent over the U.S.
    if np.invert(globe):
        ax2.set_extent([-130, -65, 23.5, 48.5])
    else:
        ax2.set_extent([-179, 179, -60, 75])
    
    
    
    
    
    # Right plot: False negative frequency
    
    # Add features
    ax3.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax3.add_feature(cfeature.STATES)
        ax3.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax3.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        # Ocean covers and "masks" data outside the U.S.
        ax3.coastlines(edgecolor = 'black', zorder = 3)
        ax3.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
    # Set a local title
    ax3.set_title('Frequency of False Negatives', fontsize = 16)
    
    # Set the tick information
    ax3.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax3.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax3.set_yticklabels(LatLabel, fontsize = 16)
    ax3.set_xticklabels(LonLabel, fontsize = 16)
    
    ax3.xaxis.set_major_formatter(LonFormatter)
    ax3.yaxis.set_major_formatter(LatFormatter)
    
    ax3.tick_params(axis = 'both', which = 'major', length = 2, width = 1,
                    labelsize = 16, bottom = True, top = True, left = True,
                    right = True, labelbottom = True, labeltop = False,
                    labelleft = False, labelright = True)
    
    # Plot the data
    cs = ax3.pcolormesh(lon, lat, np.nanmean(FN[:,:,:], axis = 0)*100, vmin = 0, vmax = 100, cmap = cmap, transform = ccrs.PlateCarree())
    
    # Set the map extent over the U.S.
    if np.invert(globe):
        ax3.set_extent([-130, -65, 23.5, 48.5])
    else:
        ax3.set_extent([-179, 179, -60, 75])
    
    # Set the colorbar location and size
    cbax = fig.add_axes([0.10, 0.435, 0.80, 0.012])
    
    # Create the colorbar
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
    
    # Set the colorbar ticks and labels
    cbar.set_ticks(np.round(np.arange(0, 100+20, 20)))
    cbar.ax.set_xticklabels(np.round(np.arange(0, 100+20, 20)), fontsize = 18)
    
    cbar.ax.set_xlabel('Frequency of Agreement/Error Type (%)', fontsize = 18)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)

#%%
##############################################

# Function to calculate and plot composite mean difference in one
def display_difference_map(true, pred, lat, lon, method, label, globe = False, path = './'):
    '''
    Calculate and plot the composite mean difference between a set of true and predicted labels

    Inputs:
    :param true: True labels in a time x lat x lon format
    :param pred: Predicted labels in a time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param method: FD identification method used to get the labels
    :paran label: Identifier to this experiement used in the filename
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param path: Path to the directory the maps will be saved to

    Outputs:
    A figure displaying the composite mean difference between the true and false labels will be created and saved
    '''

    # Determine the composite mean difference
    comp_diff, comp_diff_pval = composite_difference(pred, true)

    # Plot the composite difference
    display_stat_map(comp_diff, lon, lat, title = 'True - Predictions for %s'%method, pval = None, globe = globe,
                cmin = -0.10, cmax = 0.10, cint = 0.01, path = path, savename = '%s_%s_difference_map.png'%(label, method))

    display_stat_map(comp_diff, lon, lat, title = 'True - Predictions for %s Significance'%method, pval = comp_diff_pval, globe = globe,
                cmin = 0, cmax = 1, cint = 1, path = path, savename = '%s_%s_difference_significance_map.png'%(label, method))



# Function for the composite mean difference
def composite_difference(x, y, N = 5000):
    '''
    Calculates the composite mean difference in space between two sets of 3D data and determines the statistical significance
    at each grid point using the Monte-Carlo method.
    
    Inputs:
    :param x: One of the input data. A 3D array (lat x lon x time or lon x lat x time format)
    :param y: One of the input data. A 3D array (lat x lon x time or lon x lat x time format)
    :param N: The number of iterations used in the Monte-Carlo method for significance testing
    
    Outputs:
    :param comp_diff: The composite mean (in time) difference between each grid point in x and y
    :param pval: the p-value for CompDiff for each grid point from the Monte-Carlo bootstrapping method
    '''
    
    print('Calculating the composite mean difference')
    # Get the data sizes. 
    T, I, J = x.shape
    
    # Reshape the data into space x time arrays. For simplicity, focus on the mean to collapse the time dimension
    x_comp = np.nanmean(x.reshape(T, I*J, order = 'F'), axis = 0)
    y_comp = np.nanmean(y.reshape(T, I*J, order = 'F'), axis = 0)
    
    # Calculate the composite mean difference
    comp_diff = y_comp - x_comp
    
    # Next perform Monty - Carlo testing (with N iterations) to determine statistical significance
    print('Calculating the significance of the composite mean difference')
    
    # Initialize the index
    random_state = np.random.RandomState()
    ind = np.arange(I*J)
    # ind = np.random.randint(0, I*J, (N, I*J))
    
    comp_diff_mc = np.ones((I*J, N)) * np.nan
    
    # Calculate the composite mean difference N times with randomized data
    for n in range(N):
        ij = random_state.choice(ind, size = I*J, replace = False)
        comp_diff_mc[:,n] = y_comp - x_comp[ij]
    
    # Calculate the p-value
    pval = np.array([stats.percentileofscore(comp_diff_mc[ij,:], comp_diff[ij])/100 for ij in range(comp_diff.size)])
    
    # Reorder the desired variables into lon x lat
    comp_diff = comp_diff.reshape(I, J, order = 'F')
    pval = pval.reshape(I, J, order = 'F')
    
    print('Done \n')
    return comp_diff, pval


# Function for plotting a statistics map
def display_stat_map(x, lon, lat, title = 'Title', y = 0.68, pval = None, alpha = 0.05, globe = False,
            cmin = -1.0, cmax = 1.0, cint = 0.1, savename = 'tmp.png', path = './'):
    '''
    Create a single map (in general to display the correlation coefficient or composite mean difference). If p-value data
    is entered, this function also plots statistical significance to a desired significance level (default is 5%). The created figure is saved.
    
    Inputs:
    :param x: The gridded data (assumed statistical) to be plotted
    :param lon: The gridded longitude data associated with x
    :param lat: The gridded latitude data associated with x
    :param title: The title for the figure
    :param y: The vertical location of the title
    :param pval: The p-values for x
    :param alpha: The desired significance level for plotting statistical significance (in decimal form)
    :param cmin, cmax, cint: The minimum and maximum values for the colorbars and the interval between values on the colorbar (cint)
    :param globe: Boolean indicating whether the maps are of the globe (true) or CONUS (false)
    :param savename: The filename the figure will be saved to
    :param path: The path to the directory the figure will be saved to
               
    Outputs:
    A figure of the static is created and saved
    '''
    
    # Lonitude and latitude tick information
    if np.invert(globe):
        lat_int = 10
        lon_int = 20
    else:
        lat_int = 30
        lon_int = 60

    LatLabel = np.arange(-90, 90, lat_int)
    LonLabel = np.arange(-180, 180, lon_int)

    LonFormatter = cticker.LongitudeFormatter()
    LatFormatter = cticker.LatitudeFormatter()

    # Colorbar information
    if pval is None:
        clevs = np.arange(cmin, np.round(cmax+cint, 2), cint) # The np.round removes floating point error in the adition (which is then ceiled in np.arange)
        nlevs = len(clevs) - 1
        cmap = plt.get_cmap(name = 'RdBu_r', lut = nlevs)
        
        # Get the normalized color values
        norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
        # # Generate the colors from the orginal color map
        colors = cmap(np.linspace(0, 1, cmap.N))
        colors[int(nlevs/2-1):int(nlevs/2+1),:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
    
        # Create a new colorbar cut from the original colors with the white inserted in the middle
        cmap = mcolors.LinearSegmentedColormap.from_list('cut_RdBu_r', colors)
    else:
        cmin = 0; cmaxs = 1; cint = 1
        clevs = np.arange(cmin, cmax + cint, cint)
        nlevs = len(clevs)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"], 2)

    # Projection informatino
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()
    
    # Collect shapefile information for the U.S. and other countries
    if globe:
        pass
    else:
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
        
        CountriesReader = shpreader.Reader(CountriesSHP)
        
        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

    # Create the figure
    fig = plt.figure(figsize = [16, 18], frameon = True)
    fig.suptitle(title, y = y, size = 20)

    # Set the first part of the figure
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
        ax.add_feature(cfeature.STATES)
        ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        # Ocean covers and "masks" data outside the U.S.
        ax.coastlines(edgecolor = 'black', zorder = 3)
        ax.add_feature(cfeature.BORDERS, facecolor = 'none', edgecolor = 'black', zorder = 3)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = fig_proj)
    ax.set_yticks(LatLabel, crs = fig_proj)
    
    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the data
    if pval is None:
        cs = ax.pcolormesh(lon, lat, x[:,:], vmin = cmin, vmax = cmax,
                       cmap = cmap, transform = data_proj, zorder = 1)
    else:
        stipple = (pval < alpha/2) | (pval > (1-alpha/2)) # Create stipples for significant grid points
        # stipple = pval < alpha
        #ax.plot(lon[stipple][::3], lat[stipple][::3], 'o', color = 'Gold', markersize = 1.5, zorder = 1)
        cs = ax.pcolormesh(lon, lat, stipple, vmin = 0, vmax = 1, cmap = cmap, transform = fig_proj, zorder = 1)
    
    # Set the map extent
    if np.invert(globe):
        ax.set_extent([-130, -65, 23.5, 48.5])
    else:
        ax.set_extent([-179, 179, -60, 75])
    
    # Create the colorbar
    if pval is None:
        # Set the colorbar location and size
        cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])
        
        cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')
        
        # Adjsut the colorbar ticks
        for i in cbar.ax.get_xticklabels():
            i.set_size(18)
    
    # Save the figure.
    plt.savefig(path + '/' + savename, bbox_inches = 'tight')
    plt.show(block = False)

    
