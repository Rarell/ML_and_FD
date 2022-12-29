#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 17:44:20 2022

@author: stuartedris

This script contains functions used to create the maps and figures to display and visualize the results of the ML models    

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory

TODO:
- Might add a function to display how forests are making predictions
- Might add a function to display certain NN layers to see how it is identifying FD
- Adjust case_studies to display more than 1 type of growing seaons
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
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        fig.suptitle(model.upper(), y = 0.925, size = 18)
    else:
        plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.1, hspace = 0.2)
        fig.suptitle(model.upper(), y = 0.925, size = 18)
        
        
    for m, method in enumerate(methods):
        if len(methods) == 1: # For test cases where only 1 method is examined, the sole axis cannot be subscripted
            ax = axes
            change_pos = 0.2
        else:
            ax = axes[m]
            change_pos = 0.0
        
        
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
        ax.set_xticks(LonLabel, crs = fig_proj)
        ax.set_yticks(LatLabel, crs = fig_proj)
        
        ax.set_yticklabels(LatLabel, fontsize = 14)
        ax.set_xticklabels(LonLabel, fontsize = 14)
        
        ax.xaxis.set_major_formatter(LonFormatter)
        ax.yaxis.set_major_formatter(LatFormatter)
        
        # Plot the data
        cs = ax.pcolormesh(lon, lat, data[m], vmin = cmin, vmax = cmax, cmap = cmap, transform = data_proj, zorder = 1)
        
        # Set the extent
        if np.invert(globe):
            ax.set_extent([-130, -65, 23.5, 48.5])
        else:
            ax.set_extent([-179, 179, -65, 80])
        
        # Add method label
        ax.set_ylabel(method.title(), size = 14, labelpad = 45.0, rotation = 0)
        
        # Add a colorbar at the end
        if m == (len(methods)-1):
            cbax = fig.add_axes([0.775 + change_pos, 0.10, 0.020, 0.80])
            cbar = fig.colorbar(cs, cax = cbax, orientation = 'vertical')
            cbar.set_ticks(np.round(np.arange(cmin, cmax+cint, cint*10), 2)) # Set a total of 10 ticks
            for i in cbar.ax.yaxis.get_ticklabels():
                i.set_size(18)
            cbar.ax.set_ylabel(metric.upper(), fontsize = 18)
            
    # Save the figure
    filename = '%s_%s_%s_maps.png'%(metric, label, dataset)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
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
    fig.suptitle(model.upper(), y = 0.925, size = 18)
    
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
        ax.set_ylabel(method.title(), fontsize = 18)
        
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
    fig.suptitle(model, y = 0.945, size = 18)
    
    for m, method in enumerate(methods):
        ax = axes[m]
        
        # Plot the ROC curve
        ax.plot(epochs, loss[m][0], 'b', linewidth = 2, label = 'Training')
        ax.plot(epochs, loss[m][1], 'orange', linestyle = '.-', linewidth = 2, label = 'Validation')
        
        ax.legend(loc='top right', fontsize = 16)
        
        # Plot the variation
        ax.fill_between(epochs, loss[m][0]-loss_var[m][0], loss[m][0]+loss_var[m][0], alpha = 0.5, edgecolor = 'b', facecolor = 'b')
        ax.fill_between(epochs, loss[m][1]-loss_var[m][1], loss[m][1]+loss_var[m][1], alpha = 0.5, edgecolor = 'orange', facecolor = 'orange')
        
        # Set the label
        ax.set_ylabel(method, fontsize = 18)
        
        # Set the ticks
        # ax.set_xticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        # ax.set_yticks(np.round(np.arange(0, 1+0.2, 0.2), 1))
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
        
        # Set the tick size
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(18)
            
    # Save the figure
    filename = '%s_%s_learning_curves.png'%(metric, label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Function to display a set of feature importance
def display_feature_importance(fimportance, fimportance_var, feature_names, methods, model, label, path = './Figures'):
    '''
    Display a bargraph showing the (spatially averaged) importance of each feature with corresponding variation
    
    Inputs:
    :param fimportance: List of feature importances
    :param fimportance_var: List of variation in feature importances
    :param feature_names: List of names for each feature
    :param methods: List of methods used to identify FD
    :param model: The name of the reanalysis model the data is based on
    :param label: A label used to distinguish the experiment
    :param path: Path from the current directory to the directory the maps will be saved to
    '''
    
    # Initialize some values for the plot
    N = len(fimportance[0])
    width = 0.15
    ind = np.arange(N)
    
    # Capitilize the first letter in the methods
    methods = [method.title() for method in methods]
    
    # Create the plot
    fig, ax = plt.subplots(figsize = [18, 14])
    
    # Set the title
    ax.set_title('Feature Importance for the %s'%model.upper(), fontsize = 16)
    
    bars = []
    
    # Plot the bars
    for m, method in enumerate(methods):
        bar = ax.bar(ind+width*m, fimportance[m], width = width, yerr = fimportance_var[m])
        bars.append(bar)
        
    # Add the legend
    ax.legend(bars, methods, fontsize = 16)
        
    # Set the labels
    ax.set_ylabel('Feature Importance', fontsize = 16)
    
    # Set the ticks
    ax.set_xticks(ind + 2*width)#-width/1)
    ax.set_xticklabels(feature_names)
    
    # Set the tick size
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)
        
    # Save the figure
    filename = '%s_feature_importance.png'%(label)
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
    ax.set_title('Time Series of the %s for the %s'%(var_name, model), fontsize = 16)
    
    # Make the plots color = 'r', linestyle = '-', linewidth = 1, label = 'True values'
    ax.errorbar(time, data_true, yerr = data_true_var, capsize = 3, errorevery = 3*N, 
                color = 'r', linestyle = '-', linewidth = 1.5, label = 'True values')
    ax.errorbar(time, data_pred, yerr = data_pred_var,  capsize = 3, errorevery = 3*N, 
                color = 'b', linestyle = '-.', linewidth = 1.5, label = 'Predicted values')
    
    # Make a legend
    ax.legend(loc = 'upper right', fontsize = 16)
    
    # Set the labels
    ax.set_ylabel(var_name, fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Save the figure
    filename = '%s_%s_time_series.png'%(var_name, label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
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
        ax.set_title('Learning curve of the %s for the %s'%(metric, model), fontsize = 16)
    
        # Display the loss
        ax.plot(lc['%s'%metric], 'b-', label = 'Training')
        ax.plot(lc['val_%s'%metric], 'r--', label = 'Validation')
        
        if plot_var:
            ax.fill_between(range(len(lc['%s'%metric])), lc['%s'%metric]-lc_var['%s'%metric], lc['%s'%metric]+lc_var['%s'%metric], 
                            alpha = 0.5, edgecolor = 'b', facecolor = 'r')
            ax.fill_between(range(len(lc['val_%s'%metric])), lc['val_%s'%metric]-lc_var['val_%s'%metric], lc['val_%s'%metric]+lc_var['val_%s'%metric], 
                            alpha = 0.5, edgecolor = 'b', facecolor = 'r')
        
        ax.legend(fontsize = 16)
        
        # Set the labels
        ax.set_ylabel(metric, fontsize = 16)
        ax.set_xlabel('Epochs', fontsize = 16)
        
        # Set the tick sizes
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(16)
    
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
        ax.set_title('Flash Drought for %s'%year, size = 18)

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

        ax.set_yticklabels(LatLabel, fontsize = 18)
        ax.set_xticklabels(LonLabel, fontsize = 18)

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
        cbar.ax.set_yticklabels(['No FD', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct'], fontsize = 16)
        for i in cbar.ax.yaxis.get_ticklabels():
            i.set_size(18)
        
        # Set the colorbar ticks
        # cbar.set_ticks(np.arange(0, 12+1, 1))
        # cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 16)
        # cbar.set_ticks(np.arange(1, 11+1, 1.48))
        

        # Save the figure
        plt.savefig('%s/%s_%s_%s_case_study_%s_%s.png'%(path,label,method,year, pred_type, dataset), bbox_inches = 'tight')
        plt.show(block = False)
        
    
#%%
##############################################
    
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
        
        
    # Plot the threat scores in time
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the threat score for the %s'%(model), fontsize = 16)
    
    # Make the plots
    ax.plot(time, ts_s, 'r-', linewidth = 2, label = 'True values')

    
    # Set the labels
    ax.set_ylabel('Threat Score', fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Save the figure
    filename = 'threat_score_%s_time_series.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
    # Plot the ETS
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the equitable threat score for the %s'%(model), fontsize = 16)
    
    # Make the plots 
    ax.plot(time, ets_s, 'r-', linewidth = 2, label = 'True values')

    
    # Set the labels
    ax.set_ylabel('Equitable Threat Score', fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Save the figure
    filename = 'equitable_threat_score_%s_time_series.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    # Plot the threat scores in space
    
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
    ax.set_title('Threat Score for %s'%model, size = 18)

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

    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)

    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)

    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, ts_t, vmin = cmin, vmax = cmax,
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
    cbar.ax.set_ylabel('Threat Score', fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(18)
        
        
    # Save the figure
    filename = 'threat_score_%s_map.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
        
        
    # Create the plots
    fig = plt.figure(figsize = [12, 10])


    # Flash Drought plot
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    # Set the flash drought title
    ax.set_title('Equitable Threat Score for %s'%model, size = 18)

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

    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)

    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)

    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, ets_t, vmin = cmin, vmax = cmax,
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
    cbar.ax.set_ylabel('Equitable Threat Score', fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(18)
        
    # Save the figure
    filename = 'equitable_threat_score_%s_map.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
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
        
        
    # Plot the FAR scores in time
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the FAR score for the %s'%(model), fontsize = 16)
    
    # Make the plots
    ax.plot(time, far_s, 'r-', linewidth = 2, label = 'True values')

    
    # Set the labels
    ax.set_ylabel('FAR', fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Save the figure
    filename = 'far_%s_time_series.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
    # Plot the FAR scores in space
    
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
    ax.set_title('FAR Score for %s'%model, size = 18)

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

    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)

    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)

    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, far_t, vmin = cmin, vmax = cmax,
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
    cbar.ax.set_ylabel('FAR', fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(18)
        
        
    # Save the figure
    filename = 'far_%s_map.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
    
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
        
        
    # Plot the POD scores in time
    fig, ax = plt.subplots(figsize = [12, 8])
    
    # Set the title
    ax.set_title('Time Series of the POD for the %s'%(model), fontsize = 16)
    
    # Make the plots
    ax.plot(time, pod_s, 'r-', linewidth = 2, label = 'True values')

    
    # Set the labels
    ax.set_ylabel('POD', fontsize = 16)
    ax.set_xlabel('Time', fontsize = 16)
    
    # Set the ticks
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(16)
        
    # Save the figure
    filename = 'pod_%s_time_series.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    
    
    
    # Plot the POD scores in space
    
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
    ax.set_title('POD for %s'%model, size = 18)

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

    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)

    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)

    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, pod_t, vmin = cmin, vmax = cmax,
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
    cbar.ax.set_ylabel('POD', fontsize = 18)

    # Set the colorbar ticks
    for i in cbar.ax.yaxis.get_ticklabels():
        i.set_size(18)
        
        
    # Save the figure
    filename = 'pod_%s_map.png'%(label)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
        
        
    
