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
#   - Divide data into regions
#       - In DetermineParameters:
#           - Reorder data into 2D arrays
#           - Add overarching loop across all regions
#           - Update the Training, Validation, TPR, and FPR variable names and indices
#           - Update the print commands to include region
#           - Add a function call to create the regional map (see SESR_histograms.png for example)
#           - Create and add the function create the regional map from test.py
#       - Update the CreateSLModel function to divide data into regions
#       - Update the ModelPredictions function to divide data into regions
#   - Identify RI and drought seperately
#   - Add Li et al. FD method
#   - Add other datasets for more robustness and obtaining global scale analyses
#   - Might add other SL algorithms, and other types of NNs
#   - Might try a more effective approach to parallel processing for increased computation speed
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


#%%
# cell 1
#####################################
### Import some libraries ###########
#####################################

import os, sys, warnings
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

from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics

#%%
# cell 2
# Create a function to import the nc files

def LoadNC(SName, filename, path = './Data/Indices/'):
    '''
    This function loads .nc files, specifically raw NARR files. This function takes in the
    name of the data, and short name of the variable to load the .nc file. 
    
    Inputs:
    - SName: The short name of the variable being loaded. I.e., the name used
             to call the variable in the .nc file.
    - filename: The name of the .nc file.
    - sm: Boolean determining if soil moisture is being loaded (an extra variable and dimension, level,
          needs to be loaded).
    - path: The path from the current directory to the directory the .nc file is in.
    
    Outputs:
    - X: A directory containing all the data loaded from the .nc file. The 
         entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
         (space dimensions), 'date' contains the dates in a datetime variable
         (time dimension), 'month' 'day' are the numerical month
         and day value for the given time (time dimension), 'ymd' contains full
         datetime values, and 'SName' contains the variable (space and time demsions).
    '''
    
    # Initialize the directory to contain the data
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S'
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]

        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        X['time'] = time
        dates = np.asarray([datetime.strptime(time[d], DateFormat) for d in range(len(time))])
        
        X['date'] = dates
        X['year']  = np.asarray([d.year for d in dates])
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])

        # Collect the data itself
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

#%%
# cell 3
# Function to subset any dataset.
def SubsetData(X, Lat, Lon, LatMin, LatMax, LonMin, LonMax):
    '''
    This function is designed to subset data for any gridded dataset, including
    the non-simple grid used in the NARR dataset, where the size of the subsetted
    data is unknown. Note this function only makes square subsets with a maximum 
    and minimum latitude/longitude.
    
    Inputs:
    - X: The variable to be subsetted.
    - Lat: The gridded latitude data corresponding to X.
    - Lon: The gridded Longitude data corresponding to X.
    - LatMax: The maximum latitude of the subsetted data.
    - LatMin: The minimum latitude of the subsetted data.
    - LonMax: The maximum longitude of the subsetted data.
    - LonMin: The minimum longitude of the subsetted data.
    
    Outputs:
    - XSub: The subsetted data.
    - LatSub: Gridded, subsetted latitudes.
    - LonSub: Gridded, subsetted longitudes.
    '''
    
    # Collect the original sizes of the data/lat/lon
    I, J, T = X.shape
    
    # Reshape the data into a 2D array and lat/lon to a 1D array for easier referencing.
    X2D   = X.reshape(I*J, T, order = 'F')
    Lat1D = Lat.reshape(I*J, order = 'F')
    Lon1D = Lon.reshape(I*J, order = 'F')
    
    # Find the indices in which to make the subset.
    LatInd = np.where( (Lat1D >= LatMin) & (Lat1D <= LatMax) )[0]
    LonInd = np.where( (Lon1D >= LonMin) & (Lon1D <= LonMax) )[0]
    
    # Find the points where the lat and lon subset overlap. This comprises the subsetted grid.
    SubInd = np.intersect1d(LatInd, LonInd)
    
    # Next find, the I and J dimensions of subsetted grid.
    Start = 0 # The starting point of the column counting.
    Count = 1 # Row count starts at 1
    Isub  = 0 # Start by assuming subsetted column size is 0.
    
    for n in range(len(SubInd[:-1])): # Exclude the last value to prevent indexing errors.
        IndDiff = SubInd[n+1] - SubInd[n] # Obtain difference between this index and the next.
        if (n+2) == len(SubInd): # At the last value, everything needs to be increased by 2 to account for the missing indice at the end.
            Isub = np.nanmax([Isub, n+2 - Start]) # Note since this is the last indice, and this row is counted, there is no Count += 1.
        elif ( (IndDiff > 1) |              # If the difference is greater than 1, or if
             (np.mod(SubInd[n]+1,I) == 0) ):# SubInd is divisible by I, then a new row 
                                            # is started in the gridded array.
            Isub = np.nanmax([Isub, n+1 - Start]) # Determine the highest column count (may not be the same from row to row)
            Start = n+1 # Start the counting anew.
            Count = Count + 1 # Increment the row count by 1 as the next row is entered.
        else:
            pass
        
    # At the end, Count has the total number of rows in the subset.
    Jsub = Count
    
    # Next, the column size may not be the same from row to row. The rows with
    # with columns less than Isub need to be filled in. 
    # Start by finding how many placeholders are needed.
    PH = Isub * Jsub - len(SubInd) # Total number of needed points - number in the subset
    
    # Initialize the variable that will hold the needed indices.
    PlaceHolder = np.ones((PH)) * np.nan
    
    # Fill the placeholder values with the indices needed to complete a Isub x Jsub matrix
    Start = 0
    m = 0
    
    for n in range(len(SubInd[:-1])):
        # Identify when row changes occur.
        IndDiff = SubInd[n+1] - SubInd[n]
        if (n+2) == len(SubInd): # For the end of last row, an n+2 is needed to account for the missing index (SubInd[:-1] was used)
            ColNum = n+2-Start
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n+1] + np.arange(1, 1+Isub-ColNum)
            # Note this is the last value, so nothing else needs to be incremented up.
        elif ( (IndDiff > 1) | (np.mod(SubInd[n]+1,I) == 0) ):
            # Determine how many columns this row has.
            ColNum = n+1-Start
            
            # Fill the placeholder with the next index(ices) when the row has less than
            # the maximum number of columns (Isub)
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n] + np.arange(1, 1+Isub-ColNum)
            
            # Increment the placeholder index by the number of entries filled.
            m = m + Isub - ColNum
            Start = n+1
            
        
        else:
            pass
    
    # Next, convert the placeholders to integer indices.
    PlaceHolderInt = PlaceHolder.astype(int)
    
    # Add and sort the placeholders to the indices.
    SubIndTotal = np.sort(np.concatenate((SubInd, PlaceHolderInt), axis = 0))
    
    # The placeholder indices are technically outside of the desired subset. So
    # turn those values to NaN so they do not effect calculations.
    # (In theory, X2D is not the same variable as X, so the original dataset 
    #  should remain untouched.)
    X2D[PlaceHolderInt,:] = np.nan
    
    # Collect the subset of the data, lat, and lon
    XSub = X2D[SubIndTotal,:]
    LatSub = Lat1D[SubIndTotal]
    LonSub = Lon1D[SubIndTotal]
    
    # Reorder the data back into a 3D array, and lat and lon into gridded 2D arrays
    XSub = XSub.reshape(Isub, Jsub, T, order = 'F')
    LatSub = LatSub.reshape(Isub, Jsub, order = 'F')
    LonSub = LonSub.reshape(Isub, Jsub, order = 'F')
    
    # Return the the subsetted data
    return XSub, LatSub, LonSub


#%%
# cell 4
# Create a function to generate a range of datetimes
def DateRange(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
    - StartDate - A datetime. The starting date of the interval.
    - EndDate - A datetime. The ending date of the interval.
        
    Outputs:
    - All dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 
        

#%%
# cell 5
# Create functions to prepare training and label data
        
# Function to remove set number of columns
def ColumnRemoval(data, cols):

	data_new = data
	for c in reversed(cols):
		# Iterate in the reverse direction to avoid effecting data.
		# e.g., deleting col 5 can make col 10 turn into col 9
		# but removing col 10, col 5 is still col 5.
		data_new = np.delete(data_new, c, axis = -1) # deletes the column

	return data_new


# Function to seperate dataset based on percentage
def SplitData(train_data, target_data, per_seperation = 20.):

	# Determine how many indices need to be selected from the data.
	# Take the lower (rounded down) value, as more examples for training is useful
    I, J = train_data.shape
    num_indices = np.int(np.floor((per_seperation/100.) * I))

	# Since the examples are independent, it does not matter which data is set
	# to be training, which for testing/validation. Therefore, select them randomly
    np.random.seed(0)
    selection = np.random.choice(np.arange(I), size = num_indices, replace = False)
    train_selection = np.delete(np.arange(I), selection)
    
    xt_sel  = train_data[train_selection,:] # t refers to training
    xtv_sel = train_data[selection,:] # tv refers to testing/validation
    
    yt_sel  = target_data[train_selection] # t refers to training
    ytv_sel = target_data[selection] # tv refers to testing/validation
    
    return xt_sel, xtv_sel, yt_sel, ytv_sel


# Function to perform the preprocessing
def Preprocessing(training_data, target_data, cols):

	# seperate the training and target variables
	x_full   = training_data
	y_stable = target_data

	# Remove the non-predictive column (5) and column that gives the exact answer (13)
	x_filtered = ColumnRemoval(x_full, (cols))

	# Seperate 20% of the data for training and testing
	x_sel, x_test, y_sel, y_test = SplitData(x_filtered, y_stable, 20)

	# Split the data further for training and validation
	x_train, x_val, y_train, y_val = SplitData(x_sel, y_sel, 25)

	return x_train, x_val, x_test, x_sel, y_train, y_val, y_test, y_sel


#%%
# cell 6
# Create a function to evaluate the ML models
def EvaluateModel(Probs, y, N = 15, ProbThreshold = 0.5):
    '''
    
    '''
    
    # Get the datasize
    M = y.size
    
    # Start by calculating the confusion matrix and TPR and FPR
    
    #   Initialize the critical thresholds
    CritThresh = np.arange(0, 1+0.01, 0.01)
    
    #   Initialize the determinized matrix, confusion matrices, TPR, and FPR
    P  = np.zeros((M, CritThresh.size))
    CM = np.zeros((2, 2, CritThresh.size))
    TPR = np.zeros((CritThresh.size))
    FPR = np.zeros((CritThresh.size))
    
    #   Determinize the probabilities, calculate the confusion matrix, and calculate the TPR and FPR
    for j, ct in enumerate(CritThresh):
        # Determinize the probabilities (1 if the prob. pred. > critical threshold, and keep as 0 otherwise)
        ind = np.where(Probs >= ct)[0]
        
        P[ind,j] = 1
        
        # Calculate the confusion matrices (sum of true positive, true negative, false positive, and false negative)
        TP = (P[:,j] == 1) & (y == 1) # True positives
        TF = (P[:,j] == 0) & (y == 0) # True negatives
        FP = (P[:,j] == 1) & (y == 0) # False positives
        FN = (P[:,j] == 0) & (y == 1) # False negatives
        
        CM[0,0,j] = np.nansum(TP)
        CM[1,1,j] = np.nansum(TF)
        CM[0,1,j] = np.nansum(FP)
        CM[1,0,j] = np.nansum(FN)
        
        # Calculate the TPR and FPR
        TPR[j] = CM[0,0,j]/(CM[0,0,j] + CM[1,0,j])
        FPR[j] = 1 - CM[1,1,j]/(CM[1,1,j] + CM[0,1,j])
        
    
    #   The predicted values are the determined matrix for CritThresh = 0.5
    ind  = np.where(CritThresh == ProbThreshold)[0]
    PredCM = CM[:,:,ind[0]]
    
    yPred = P[:,ind[0]]
    
    
    # Calculate the residuals
    Residuals = y - yPred
    
    
    # Next, determine the cross-entropy of the model
    Ent  = -1 * (1/M) * np.nansum(y * np.log(Probs + 1e-5) + (1 - y) * np.log(1 - Probs + 1e-5))
    
    
    # Calculate various statistical performance metrics: the Adjusted-R^2, RMSE, Cp, AIC, and BIC
    
    #   Start by calculating some necessary components (RSS, TSS, and error variance)
    #   RSS
    RSS = np.nansum((y - yPred)**2)

    #   TSS
    TSS = np.nansum((y - np.nanmean(y))**2)
    
    #   Variation of Errors
    SigE = np.nanvar(Residuals)
    
    #   Adjusted R^2
    AdjustedR2 = 1 - (RSS/(M - N - 1))/(TSS/(M - 1))
    
    #   MSE
    RMSE = np.sqrt(RSS/M)
    
    #   Cp
    Cp = (1/M) * (RSS + 2 * N * SigE)
    
    #   AIC
    AIC = Cp/M
    
    #   BIC
    BIC = (1/M) * (RSS + np.log(M) * N * SigE)
    
    
    # Calculate some performance metrics from the confusion matric (Accuracy, Precision, Recall, F1-Score, Specificity, and Risk)
    #   Accuracy
    Accuracy = (PredCM[0,0] + PredCM[1,1])/(PredCM[0,0] + PredCM[1,1] + PredCM[1,0] + PredCM[0,1])
    
    #   Precision
    Precision = PredCM[0,0]/(PredCM[0,0] + PredCM[1,1])
    
    #   Recall (or sensitivity or TPR)
    Recall = TPR[ind[0]]
    
    #   F1-Score
    F1Score = (2 * Recall * Precision)/(Recall + Precision)
    
    #   Specificity (or true negative rate (TNR))
    Specificity = PredCM[1,1]/(PredCM[1,1] + PredCM[0,1])
    
    #   Risk
    Risk = (PredCM[0,1] + PredCM[1,0])/(PredCM[0,0] + PredCM[1,1] + PredCM[1,0] + PredCM[0,1])
    
    
    # Finally, calculate some performance metrics from the ROC curve (AUC, Youden Index, and minimum distance)
    #   AUC
    AUC = metrics.roc_auc_score(y, Probs)
    
    #   Youden index
    Youden = TPR - FPR
    
    #   Determine the maximum Youden index and corresponding index
    YoudInd = np.where(Youden == np.nanmax(Youden))[0]
    YoudThresh = CritThresh[YoudInd[0]]
    YoudenMax  = Youden[YoudInd[0]]
    
    #   Distance from leftmost point
    d = np.sqrt((1 - TPR)**2 + FPR**2)
    
    #   Determine the minimum distance
    dInd = np.where(d == np.nanmin(d))[0]
    dThresh = CritThresh[dInd[0]]
    dMin    = d[dInd[0]]
    
    
    # Return all the performance metrics
    return TPR, FPR, Ent, AdjustedR2, RMSE, Cp, AIC, BIC, Accuracy, Precision, Recall, F1Score, Specificity, Risk, AUC, YoudenMax, YoudThresh, dMin, dThresh

#%%
# cell
# Create some functions to generate maps of results

# Create a function to create climatology maps using SL predictions
def FDClimatologyMap(FD, lat, lon, AllYears, months, years, title = 'tmp', savename = 'tmp.png', OutPath = './Figures/'):
    '''
    
    '''
    I, J, T = FD.shape
    AnnFD = np.ones((I, J, AllYears.size)) * np.nan
    
    # Determine the average number of flash droughts in a year
    for y in range(AllYears.size):
        yInd = np.where( (AllYears[y] == years) & ((months >= 4) & (months <= 10)) )[0] # Second set of conditions ensures only growing season values
        
        # Calculate the mean number of flash drought for each year    
        AnnFD[:,:,y] = np.nanmean(FD[:,:,yInd], axis = -1)
        
        # Turn nonzero values to 1 (each year gets 1 count to the total)    
        AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                                AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
            
        
    # Calculate the percentage number of years with rapid intensifications and flash droughts
    PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/AllYears.size
    
    # Turn 0 values into nan
    PerAnnFD = np.where(PerAnnFD != 0, PerAnnFD, np.nan)
    
    #### Create the Plot ####

    # Set colorbar information
    cmin = -20; cmax = 80; cint = 1
    clevs = np.arange(-20, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    
    
    # Get the normalized color values
    norm = mcolors.Normalize(vmin = 0, vmax = cmax)
    
    # Generate the colors from the orginal color map in range from [0, cmax]
    colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
    colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
    
    # Create a new colorbar cut from the colors in range [0, cmax.]
    ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
    
    colorsNew = cmap(np.linspace(0, 1, cmap.N))
    colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
    cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)
    
    # Shapefile information
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Lonitude and latitude tick information
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
    ax.set_title(title, size = 18)
    
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    ax.add_feature(cfeature.STATES)
    ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
    ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, PerAnnFD*100, vmin = cmin, vmax = cmax,
                      cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent to the U.S.
    ax.set_extent([-130, -65, 23.5, 48.5])
    
    
    # Set the colorbar size and location
    cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])
    
    # Create the colorbar
    cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')
    
    # Set the colorbar label
    cbar.ax.set_ylabel('% of years with Flash Drought', fontsize = 18)
    
    # Set the colorbar ticks
    cbar.set_ticks(np.arange(0, 90, 10))
    cbar.ax.set_yticklabels(np.arange(0, 90, 10), fontsize = 16)
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    
# Create a function to create a case study map
def FDAnnualMaps(FD, lat, lon, CaseYear, months, years, title = 'tmp', savename = 'tmp.png', OutPath = './Figures/'):
    '''
    
    '''
    FDYear = np.zeros((I, J))
    
    NMonths = 12
    
    for m in range(NMonths):
        ind = np.where( (years == CaseYear) & (months == m) )[0]
        FDYear = np.where(((np.nansum(FD[:,:,ind], axis = -1) != 0 ) & (FDYear == 0)), m, FDYear) # Points where the prediction for the month is nonzero (FD is predicted) and 
                                                                                                  # FDYear does not have a value already, are given a value of m. FDYear is left alone otherwise.
        
    # Create a figure to plot this.

    # Set colorbar information
    # cmin = 0; cmax = 12; cint = 1
    cmin = 3; cmax = 10; cint = 1
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)
    
    # Shapefile information
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    ShapeName = 'admin_0_countries'
    CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
    CountriesReader = shpreader.Reader(CountriesSHP)
    
    USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
    NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Lonitude and latitude tick information
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
    ax.set_title(title, size = 18)
    
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    ax.add_feature(cfeature.STATES)
    ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
    ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax.set_yticklabels(LatLabel, fontsize = 18)
    ax.set_xticklabels(LonLabel, fontsize = 18)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the flash drought data
    cs = ax.pcolormesh(lon, lat, FDYear, vmin = cmin, vmax = cmax,
                      cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent to the U.S.
    ax.set_extent([-130, -65, 23.5, 48.5])
    
    
    # Set the colorbar size and location
    cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])
    
    # Create the colorbar
    cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, orientation = 'vertical')
    
    
    # Set the colorbar ticks
    #cbar.set_ticks(np.arange(5, 100+1, 10))
    cbar.set_ticks(np.arange(0.05, 1, 0.128))
    cbar.ax.set_yticklabels(['No FD', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct'], fontsize = 16)
    
    # Save the figure
    plt.savefig(OutPath + savename, bbox_inches = 'tight')
    plt.show(block = False)
    
    
    
#%%
# cell
# Create functions that will create, train, and make probabilistic predictions of SL models and output the weights of of each index

### Function for SL models
# Define a function to create and evaluate a decision tree model.
def RFModel(xTrain, yTrain, xVal, N_trees = 50, crit = 'gini', max_depth = 5, max_features = 10, NJobs = -1):
    '''


    '''
    
    # Make the random forest. # Note this is default set to run parallel across all CPUs
    RF = ensemble.RandomForestClassifier(n_estimators = N_trees, criterion = crit, max_depth = max_depth, max_features = max_features, bootstrap = True, oob_score = True, n_jobs = NJobs)
    
    # Train the tree
    RF.fit(xTrain, yTrain)
    
    # Make probabilistic predictions
    Prob = RF.predict_proba(xVal)
    
    # Get the parameters of the forest
    TrainingWeights = RF.feature_importances_

    return Prob, TrainingWeights


def SVMModel(xTrain, yTrain, xVal, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale', max_iter = 200):
    '''
    
    '''
    
    # Make the SVM.
    SVM = svm.SVR(C = RegParam, kernel = Kernel, gamma = Gamma)
    
    # Train the SVM
    SVM.fit(xTrain, yTrain)
    
    # Make probabilistic predictions
    Prob = SVM.predict_proba(xVal)
    
    return Prob


def ANNModel(xTrain, yTrain, xVal, layers = (15,), activation = 'relu', solver = 'adam', learn_rate = 'constant'):
    '''
    
    '''
    
    # Make the ANN.
    ANN = neural_network.MLPClassifier(hidden_layer_sizes = layers, activation = activation, solver = solver, learning_rate = learn_rate)
    
    # Train the ANN
    ANN.fit(xTrain, yTrain)
    
    # Make probabilistic predictions
    Prob = ANN.predict_proba(xVal)
    
    return Prob



#%%
# cell 
# Create a function to create SL models and output performance metrics to test parameters
def DetermineParameters(Train, Label, Model, GriddedRegion = 'USA', NJobs = -1):
    '''
    
    '''
    
    # First split the data based on year
    TrainInd = np.where( ( (sesr['year'] == 1979) | (sesr['year'] == 1980) | (sesr['year'] == 1981) | 
                          (sesr['year'] == 1982) | (sesr['year'] == 1990) | (sesr['year'] == 1991) | 
                          (sesr['year'] == 1992) | (sesr['year'] == 1995) | (sesr['year'] == 1996) |
                          (sesr['year'] == 1997) | (sesr['year'] == 1999) | (sesr['year'] == 2000) |
                          (sesr['year'] == 2002) | (sesr['year'] == 2004) | (sesr['year'] == 2008) |
                          (sesr['year'] == 2009) | (sesr['year'] == 2010) | (sesr['year'] == 2012) |
                          (sesr['year'] == 2013) | (sesr['year'] == 2014) | (sesr['year'] == 2015) |
                          (sesr['year'] == 2016) | (sesr['year'] == 2017) | (sesr['year'] == 2018) |
                          (sesr['year'] == 2020) ) & ((sesr['month'] >= 4) & (sesr['month'] <= 10)) )[0]
    
    ValInd   = np.where( ( (sesr['year'] == 1983) | (sesr['year'] == 1985) | (sesr['year'] == 1986) |
                          (sesr['year'] == 1994) | (sesr['year'] == 1998) | (sesr['year'] == 2005) |
                          (sesr['year'] == 2007) | (sesr['year'] == 2011) ) & ((sesr['month'] >= 4) & (sesr['month'] <= 10)) )[0]
    TestInd  = np.where( ( (sesr['year'] == 1984) | (sesr['year'] == 1987) | (sesr['year'] == 1988) |
                          (sesr['year'] == 1989) | (sesr['year'] == 1993) | (sesr['year'] == 2001) |
                          (sesr['year'] == 2003) | (sesr['year'] == 2006) | (sesr['year'] == 2019) ) &
                        ((sesr['month'] >= 4) & (sesr['month'] <= 10)) )[0]
    
    xTrain = Train[:,TrainInd,:]; yTrain = Label[:,TrainInd,:]
    xVal = Train[:,ValInd,:]; yVal = Label[:,ValInd,:]
    
    IJTrain, Ttrain, NVar = xTrain.shape
    IJVal, Tval, NVar   = xVal.shape
    IJVal, Tval, NMethods = yVal.shape
    
    # Reorder data into 2D matrices
    xTrain = xTrain.reshape(IJTrain*Ttrain, NVar, order = 'F')
    xVal   = xVal.reshape(IJVal*Tval, NVar, order = 'F')
    yTrain = yTrain.reshape(IJTrain*Ttrain, NMethods, order = 'F')
    yVal   = yVal.reshape(IJVal*Tval, NMethods, order = 'F')
    
    # Split the data into regions depending on scale of the dataset
    if GriddedRegion == 'USA':
        # Define the regions
        Regions = ['Northwest', 'West North Central', 'East North Central', 'Northeast', 'West', 'Southwest', 'South', 'Southeast']
        
        # Define the TPR and FPR to the number of regions
        TPRM1 = np.ones((100, Regions.size)) * np.nan
        TPRM2 = np.ones((100, Regions.size)) * np.nan
        TPRM3 = np.ones((100, Regions.size)) * np.nan
        TPRM4 = np.ones((100, Regions.size)) * np.nan
        
        FPRM1 = np.ones((100, Regions.size)) * np.nan
        FPRM2 = np.ones((100, Regions.size)) * np.nan
        FPRM3 = np.ones((100, Regions.size)) * np.nan
        FPRM4 = np.ones((100, Regions.size)) * np.nan
        
        # Initialize the dictonaries for split training and validation data.
        xTrainRegions = {}
        xValRegions = {}
        yTrainRegions = {}
        yValRegions = {}
        
        # Split the training data
        for t in Ttrain:
            xTrainRegions[Regions[0]] = xTrain[(lat >= 42) & (lat <= 50) & (lon >= -130) & (lon <= -111),:,:] # NW region
            xTrainRegions[Regions[1]] = xTrain[(lat >= 42) & (lat <= 50) & (lon >= -111) & (lon <= -94),:,:] # WNC region
            xTrainRegions[Regions[2]] = xTrain[(lat >= 38) & (lat <= 50) & (lon >= -94) & (lon <= -75.5),:,:] # EWC region
            xTrainRegions[Regions[3]] = xTrain[(lat >= 38) & (lat <= 50) & (lon >= -75.5) & (lon <= -65),:,:] # NE region
            xTrainRegions[Regions[4]] = xTrain[(lat >= 25) & (lat <= 42) & (lon >= -130) & (lon <= -114),:,:] # W region
            xTrainRegions[Regions[5]] = xTrain[(lat >= 25) & (lat <= 42) & (lon >= -114) & (lon <= -105),:,:] # SW region
            xTrainRegions[Regions[6]] = xTrain[(lat >= 25) & (lat <= 42) & (lon >= -105) & (lon <= -94),:,:] # S region
            xTrainRegions[Regions[7]] = xTrain[(lat >= 25) & (lat <= 38) & (lon >= -94) & (lon <= -65),:,:] # SE region
            
            yTrainRegions[Regions[0]] = yTrain[(lat >= 42) & (lat <= 50) & (lon >= -130) & (lon <= -111),:,:] # NW region
            yTrainRegions[Regions[1]] = yTrain[(lat >= 42) & (lat <= 50) & (lon >= -111) & (lon <= -94),:,:] # WNC region
            yTrainRegions[Regions[2]] = yTrain[(lat >= 38) & (lat <= 50) & (lon >= -94) & (lon <= -75.5),:,:] # EWC region
            yTrainRegions[Regions[3]] = yTrain[(lat >= 38) & (lat <= 50) & (lon >= -75.5) & (lon <= -65),:,:] # NE region
            yTrainRegions[Regions[4]] = yTrain[(lat >= 25) & (lat <= 42) & (lon >= -130) & (lon <= -114),:,:] # W region
            yTrainRegions[Regions[5]] = yTrain[(lat >= 25) & (lat <= 42) & (lon >= -114) & (lon <= -105),:,:] # SW region
            yTrainRegions[Regions[6]] = yTrain[(lat >= 25) & (lat <= 42) & (lon >= -105) & (lon <= -94),:,:] # S region
            yTrainRegions[Regions[7]] = yTrain[(lat >= 25) & (lat <= 38) & (lon >= -94) & (lon <= -65),:,:] # SE region
            
        # Split the validation data
        for t in Tval:
            xValRegions[Regions[0]] = xVal[(lat >= 42) & (lat <= 50) & (lon >= -130) & (lon <= -111),:,:] # NW region
            xValRegions[Regions[1]] = xVal[(lat >= 42) & (lat <= 50) & (lon >= -111) & (lon <= -94),:,:] # WNC region
            xValRegions[Regions[2]] = xVal[(lat >= 38) & (lat <= 50) & (lon >= -94) & (lon <= -75.5),:,:] # EWC region
            xValRegions[Regions[3]] = xVal[(lat >= 38) & (lat <= 50) & (lon >= -75.5) & (lon <= -65),:,:] # NE region
            xValRegions[Regions[4]] = xVal[(lat >= 25) & (lat <= 42) & (lon >= -130) & (lon <= -114),:,:] # W region
            xValRegions[Regions[5]] = xVal[(lat >= 25) & (lat <= 42) & (lon >= -114) & (lon <= -105),:,:] # SW region
            xValRegions[Regions[6]] = xVal[(lat >= 25) & (lat <= 42) & (lon >= -105) & (lon <= -94),:,:] # S region
            xValRegions[Regions[7]] = xVal[(lat >= 25) & (lat <= 38) & (lon >= -94) & (lon <= -65),:,:] # SE region
            
            yValRegions[Regions[0]] = yVal[(lat >= 42) & (lat <= 50) & (lon >= -130) & (lon <= -111),:,:] # NW region
            yValRegions[Regions[1]] = yVal[(lat >= 42) & (lat <= 50) & (lon >= -111) & (lon <= -94),:,:] # WNC region
            yValRegions[Regions[2]] = yVal[(lat >= 38) & (lat <= 50) & (lon >= -94) & (lon <= -75.5),:,:] # EWC region
            yValRegions[Regions[3]] = yVal[(lat >= 38) & (lat <= 50) & (lon >= -75.5) & (lon <= -65),:,:] # NE region
            yValRegions[Regions[4]] = yVal[(lat >= 25) & (lat <= 42) & (lon >= -130) & (lon <= -114),:,:] # W region
            yValRegions[Regions[5]] = yVal[(lat >= 25) & (lat <= 42) & (lon >= -114) & (lon <= -105),:,:] # SW region
            yValRegions[Regions[6]] = yVal[(lat >= 25) & (lat <= 42) & (lon >= -105) & (lon <= -94),:,:] # S region
            yValRegions[Regions[7]] = yVal[(lat >= 25) & (lat <= 38) & (lon >= -94) & (lon <= -65),:,:] # SE region
        
    else:
        pass
    
    # Next, Start performing SL models for each method.
    ##### Remember to add Li
    Methods = ['Christian', 'Noguera', 'Liu', 'Pendergrass', 'Otkin']
    
    for method in Methods:
        if method == 'Christian': # Christian et al. method uses SESR
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([0, 1]))
            ValData   = ColumnRemoval(xVal, cols = np.asarray([0, 1]))
            TrainLabel = yTrain[:,0]
            ValLabel   = yVal[:,0]
            
            NVarRemoved = 2
            
        elif method == 'Noguera': # Noguera et al method uses SPEI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([4, 5]))
            ValData   = ColumnRemoval(xVal, cols = np.asarray([4, 5]))
            TrainLabel = yTrain[:,1]
            ValLabel   = yVal[:,1]
            
            NVarRemoved = 2
        
        elif method == 'Li': # Li et al. method uses SEDI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([2, 3]))
            ValData   = ColumnRemoval(xVal, cols = np.asarray([2, 3]))
            TrainLabel = yTrain[:,2]
            ValLabel   = yVal[:,2]
            
            NVarRemoved = 2
            
        elif method == 'Liu': # Liu et al. method uses soil moisture
            TrainData = xTrain
            ValData   = xVal
            TrainLabel = yTrain[:,3]
            ValLabel   = yVal[:,3]
            
            NVarRemoved = 0
            
        elif method == 'Pendergrass': # Penndergrass et al. method uses EDDI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([9, 10]))
            ValData   = ColumnRemoval(xVal, cols = np.asarray([9, 10]))
            TrainLabel = yTrain[:,4]
            ValLabel   = yVal[:,4]
            
            NVarRemoved = 2
            
        else: # Otkin et al. Method uses FDII
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([14]))
            ValData   = ColumnRemoval(xVal, cols = np.asarray([14]))
            TrainLabel = yTrain[:,5]
            ValLabel   = yVal[:,5]
            
            NVarRemoved = 1
        
        print('Creating the ' + Model + 's for the ' + method + ' et al. method.')
        # Note on nomenclature: M# refers toa value(s) for Model #. So ProbM1 is probability values for Model 1, RMSEM2 is the RMSE for Model 2, etc. Names are generic to move across multiple SL models.
        if (Model == 'RF') | (Model == 'Random Forest'):
            # Past studies on predicting droughts with RFs have maintained default settings, while letting the number of trees vary from 10 to 50 to 100 to 200 to 1000.
            # For simplicity and consistency, follow this procedure for now.
            ProbM1, _ = RFModel(TrainData, TrainLabel, ValData, N_trees = 50, crit = 'gini', max_depth = None, max_features = 'auto', NJobs = NJobs)
            ProbM2, _ = RFModel(TrainData, TrainLabel, ValData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto', NJobs = NJobs)
            ProbM3, _ = RFModel(TrainData, TrainLabel, ValData, N_trees = 200, crit = 'gini', max_depth = None, max_features = 'auto', NJobs = NJobs)
            ProbM4, _ = RFModel(TrainData, TrainLabel, ValData, N_trees = 1000, crit = 'gini', max_depth = None, max_features = 'auto', NJobs = NJobs) # Note this last one can take a long time to run. It will only be excepted if it really outperforms the others.
            
            TextM1 = '50 tree random forest'
            TextM2 = '100 tree random forest'
            TextM3 = '200 tree random forest'
            TextM4 = '1000 tree random forest'
           
        elif (Model == 'SVM') | (Model == 'Support Vector Machine'):
            # Other studies are fairly consistent in using the radial basis function kernel, but do not detail other parameters. Modified parameter for this run will be kernal functions.
            # May come back to this and toy with other parameters
            
            ProbM1 = SVMModel(TrainData, TrainLabel, ValData, Kernel = 'linear', RegParam = 1.0, Gamma = 'scale')
            ProbM2 = SVMModel(TrainData, TrainLabel, ValData, Kernel = 'poly', RegParam = 1.0, Gamma = 'scale')
            ProbM3 = SVMModel(TrainData, TrainLabel, ValData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
            ProbM4 = SVMModel(TrainData, TrainLabel, ValData, Kernel = 'sigmoid', RegParam = 1.0, Gamma = 'scale')
            
            TextM1 = 'Linear SVM'
            TextM2 = 'Polynomial SVM'
            TextM3 = 'Radial basis SVM'
            TextM4 = 'Sigmoid SVM'
            
        elif (Model == 'ANN') | (Model == 'Nueral Network'):
            # Only one study gives the parameters used, which were 1 layerr with 14 and 15 neurons. Base parameter variation off of this.
            # May come back to this and toy with other parameters
            
            ProbM1 = ANNModel(TrainData, TrainLabel, ValData, layers = (15,))
            ProbM2 = ANNModel(TrainData, TrainLabel, ValData, layers = (25,))
            ProbM3 = ANNModel(TrainData, TrainLabel, ValData, layers = (15, 15))
            ProbM4 = ANNModel(TrainData, TrainLabel, ValData, layers = (25, 25))
            
            TextM1 = '1 layer, 15 node ANN'
            TextM2 = '1 layer, 25 node ANN'
            TextM3 = '2 layer, 15 node ANN'
            TextM4 = '2 layer, 25 node ANN'
            
            
        else: ##### Add more models here
            pass
       
        print('Evaluating the ' + Model + 's for the ' + method + ' et al. method.')
        TPRM1, FPRM1, EntM1, R2M1, RMSEM1, CpM1, AICM1, BICM1, AccM1, PrecM1, RecallM1, F1M1, SpecM1, RiskM1, AUCM1, YoudM1, YoudThreshM1, dM1, dThreshM1 = EvaluateModel(ProbM1[:,1], ValLabel, N = (NVar - NVarRemoved))
        TPRM2, FPRM2, EntM2, R2M2, RMSEM2, CpM2, AICM2, BICM2, AccM2, PrecM2, RecallM2, F1M2, SpecM2, RiskM2, AUCM2, YoudM2, YoudThreshM2, dM2, dThreshM2 = EvaluateModel(ProbM2[:,1], ValLabel, N = (NVar - NVarRemoved))
        TPRM3, FPRM3, EntM3, R2M3, RMSEM3, CpM3, AICM3, BICM3, AccM3, PrecM3, RecallM3, F1M3, SpecM3, RiskM3, AUCM3, YoudM3, YoudThreshM3, dM3, dThreshM3 = EvaluateModel(ProbM3[:,1], ValLabel, N = (NVar - NVarRemoved))
        TPRM4, FPRM4, EntM4, R2M4, RMSEM4, CpM4, AICM4, BICM4, AccM4, PrecM4, RecallM4, F1M4, SpecM4, RiskM4, AUCM4, YoudM4, YoudThreshM4, dM4, dThreshM4 = EvaluateModel(ProbM4[:,1], ValLabel, N = (NVar - NVarRemoved))
        
        # Output the performance statistics

        #   Cross-Entropy
        print('The ' + method + ' et al. ' + TextM1 + ' has a cross-entropy of: %4.2f' %EntM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a cross-entropy of: %4.2f' %EntM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a cross-entropy of: %4.2f' %EntM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a cross-entropy of: %4.2f' %EntM4)
        print('\n')
        
        #   Adjusted-R^2
        print('The ' + method + ' et al. ' + TextM1 + ' has an Adjusted-R^2 of: %4.2f' %R2M1)
        print('The ' + method + ' et al. ' + TextM2 + ' has an Adjusted-R^2 of: %4.2f' %R2M2)
        print('The ' + method + ' et al. ' + TextM3 + ' has an Adjusted-R^2 of: %4.2f' %R2M3)
        print('The ' + method + ' et al. ' + TextM4 + ' has an Adjusted-R^2 of: %4.2f' %R2M4)
        print('\n')
        
        #   RMSE
        print('The ' + method + ' et al. ' + TextM1 + ' has a RMSE of: %4.2f' %RMSEM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a RMSE of: %4.2f' %RMSEM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a RMSE of: %4.2f' %RMSEM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a RMSE of: %4.2f' %RMSEM4)
        print('\n')
        
        #   Cp
        print('The ' + method + ' et al. ' + TextM1 + ' has a Cp of: %4.2f' %CpM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Cp of: %4.2f' %CpM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Cp of: %4.2f' %CpM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Cp of: %4.2f' %CpM4)
        print('\n')
        
        #   AIC
        print('The ' + method + ' et al. ' + TextM1 + ' has a AIC of: %4.2f' %AICM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a AIC of: %4.2f' %AICM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a AIC of: %4.2f' %AICM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a AIC of: %4.2f' %AICM4)
        print('\n')
        
        #   BIC
        print('The ' + method + ' et al. ' + TextM1 + ' has a BIC of: %4.2f' %BICM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a BIC of: %4.2f' %BICM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a BIC of: %4.2f' %BICM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a BIC of: %4.2f' %BICM4)
        print('\n')
        
        #   Accuracy
        print('The ' + method + ' et al. ' + TextM1 + ' has a Accuracy of: %4.2f' %AccM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Accuracy of: %4.2f' %AccM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Accuracy of: %4.2f' %AccM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Accuracy of: %4.2f' %AccM4)
        print('\n')
        
        #   Precision
        print('The ' + method + ' et al. ' + TextM1 + ' has a Precision of: %4.2f' %PrecM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Precision of: %4.2f' %PrecM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Precision of: %4.2f' %PrecM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Precision of: %4.2f' %PrecM4)
        print('\n')
        
        #   Recall
        print('The ' + method + ' et al. ' + TextM1 + ' has a Recall of: %4.2f' %RecallM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Recall of: %4.2f' %RecallM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Recall of: %4.2f' %RecallM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Recall of: %4.2f' %RecallM4)
        print('\n')
        
        #   F1-Score
        print('The ' + method + ' et al. ' + TextM1 + ' has a F1-Score of: %4.2f' %F1M1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a F1-Score of: %4.2f' %F1M2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a F1-Score of: %4.2f' %F1M3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a F1-Score of: %4.2f' %F1M4)
        print('\n')
        
        #   Specificity
        print('The ' + method + ' et al. ' + TextM1 + ' has a Specificity of: %4.2f' %SpecM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Specificity of: %4.2f' %SpecM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Specificity of: %4.2f' %SpecM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Specificity of: %4.2f' %SpecM4)
        print('\n')
        
        #   Risk
        print('The ' + method + ' et al. ' + TextM1 + ' has a Risk of: %4.2f' %RiskM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has a Risk of: %4.2f' %RiskM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has a Risk of: %4.2f' %RiskM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has a Risk of: %4.2f' %RiskM4)
        print('\n')
        
        #   AUC
        print('The ' + method + ' et al. ' + TextM1 + ' has an AUC of: %4.2f' %AUCM1)
        print('The ' + method + ' et al. ' + TextM2 + ' has an AUC of: %4.2f' %AUCM2)
        print('The ' + method + ' et al. ' + TextM3 + ' has an AUC of: %4.2f' %AUCM3)
        print('The ' + method + ' et al. ' + TextM4 + ' has an AUC of: %4.2f' %AUCM4)
        print('\n')
        
        #   Youden Index
        print('The ' + method + ' et al. ' + TextM1 + ' has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudM1, YoudThreshM1))
        print('The ' + method + ' et al. ' + TextM2 + ' has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudM2, YoudThreshM2))
        print('The ' + method + ' et al. ' + TextM3 + ' has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudM3, YoudThreshM3))
        print('The ' + method + ' et al. ' + TextM4 + ' has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudM4, YoudThreshM4))
        print('\n')
        
        #   Distance from leftmost corner of ROC curve
        print('The ' + method + ' et al. ' + TextM1 + ' has a minimum distance of %4.2f at the threshold of %4.3f' %(dM1, dThreshM1))
        print('The ' + method + ' et al. ' + TextM2 + ' has a minimum distance of %4.2f at the threshold of %4.3f' %(dM2, dThreshM2))
        print('The ' + method + ' et al. ' + TextM3 + ' has a minimum distance of %4.2f at the threshold of %4.3f' %(dM3, dThreshM3))
        print('The ' + method + ' et al. ' + TextM4 + ' has a minimum distance of %4.2f at the threshold of %4.3f' %(dM4, dThreshM4))
        print('\n')
        
        
        # Finally output a ROC curve to finish evaluating the models
        fig = plt.figure(figsize = [14,14])
        ax = fig.add_subplot(1,1,1)
        
        #   Set the title
        ax.set_title('Receiver Operating Characteristic Curve for ' + Model + 's using the ' + method + ' et al. Method', fontsize = 24)
        
        #   Create the plots
        ax.plot(FPRM1, TPRM1, 'r-', linewidth = 2.0, label = TextM1)
        ax.plot(FPRM2, TPRM2, 'b-', linewidth = 2.0, label = TextM2)
        ax.plot(FPRM3, TPRM3, 'k-', linewidth = 2.0, label = TextM3)
        ax.plot(FPRM4, TPRM4, 'g-', linewidth = 2.0, label = TextM4)
        
        #   Set the legend
        ax.legend(loc = 'best', fontsize = 33)
        
        #   Set the figure limits and labels
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.02])
        
        ax.set_xlabel('False Positive Rate', fontsize = 22)
        ax.set_ylabel('True Positive Rate', fontsize = 22)
        
        #   Set the tick sizes
        for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            i.set_size(22)
            
        plt.show(block = False)
        
        
#%%
# cell
# Create a function to create the final SL models, based on the best performing parameters, and output their performance
def CreateSLModel(Train, Label, Model):
    '''
    
    '''
    
    # First split the data based on year
    TrainInd = np.where( ( (sesr['year'] == 1979) | (sesr['year'] == 1980) | (sesr['year'] == 1981) | 
                          (sesr['year'] == 1982) | (sesr['year'] == 1983) | (sesr['year'] == 1985) | 
                          (sesr['year'] == 1986) | (sesr['year'] == 1990) | (sesr['year'] == 1991) | 
                          (sesr['year'] == 1992) | (sesr['year'] == 1994) | (sesr['year'] == 1995) | 
                          (sesr['year'] == 1996) | (sesr['year'] == 1997) | (sesr['year'] == 1998) | 
                          (sesr['year'] == 1999) | (sesr['year'] == 2000) | (sesr['year'] == 2002) | 
                          (sesr['year'] == 2004) | (sesr['year'] == 2005) | (sesr['year'] == 2007) | 
                          (sesr['year'] == 2008) | (sesr['year'] == 2009) | (sesr['year'] == 2010) | 
                          (sesr['year'] == 2011) | (sesr['year'] == 2012) | (sesr['year'] == 2013) | 
                          (sesr['year'] == 2014) | (sesr['year'] == 2015) | (sesr['year'] == 2016) | 
                          (sesr['year'] == 2017) | (sesr['year'] == 2018) | (sesr['year'] == 2020) ) & 
                        ((sesr['month'] >= 4) & (sesr['month'] <= 10)) )[0]
    
    TestInd  = np.where( ( (sesr['year'] == 1984) | (sesr['year'] == 1987) | (sesr['year'] == 1988) |
                          (sesr['year'] == 1989) | (sesr['year'] == 1993) | (sesr['year'] == 2001) |
                          (sesr['year'] == 2003) | (sesr['year'] == 2006) | (sesr['year'] == 2019) ) &
                        ((sesr['month'] >= 4) & (sesr['month'] <= 10)) )[0]
    
    xTrain = Train[:,TrainInd,:]; yTrain = Label[:,TrainInd,:]
    xTest = Train[:,TestInd,:]; yTest = Label[:,TestInd,:]
    
    IJTrain, Ttrain, NVar = xTrain.shape
    IJTest, Ttest, NVar   = xTest.shape
    IJTest, Ttest, NMethods = yTest.shape
    
    # Reorder data into 2D matrices
    xTrain = xTrain.reshape(IJTrain*Ttrain, NVar, order = 'F')
    xTest  = xTest.reshape(IJTest*Ttest, NVar, order = 'F')
    yTrain = yTrain.reshape(IJTrain*Ttrain, NMethods, order = 'F')
    yTest  = yTest.reshape(IJTest*Ttest, NMethods, order = 'F')
    
    
    
    # Next, Start performing SL models for each method.
    ##### Remember to add Li
    Methods = ['Christian', 'Noguera', 'Liu', 'Pendergrass', 'Otkin']
    
    for method in Methods:
        if method == 'Christian': # Christian et al. method uses SESR
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([0, 1]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([0, 1]))
            TrainLabel = yTrain[:,0]
            TestLabel  = yTest[:,0]
            
            NVarRemoved = 2
            
        elif method == 'Noguera': # Noguera et al method uses SPEI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([4, 5]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([4, 5]))
            TrainLabel = yTrain[:,1]
            TestLabel  = yTest[:,1]
            
            NVarRemoved = 2
        
        elif method == 'Li': # Li et al. method uses SEDI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([2, 3]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([2, 3]))
            TrainLabel = yTrain[:,2]
            TestLabel  = yTest[:,2]
            
            NVarRemoved = 2
            
        elif method == 'Liu': # Liu et al. method uses soil moisture
            TrainData = xTrain
            TestData  = xTest
            TrainLabel = yTrain[:,3]
            TestLabel  = yTest[:,3]
            
            NVarRemoved = 0
            
        elif method == 'Pendergrass': # Penndergrass et al. method uses EDDI
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([9, 10]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([9, 10]))
            TrainLabel = yTrain[:,4]
            TestLabel  = yTest[:,4]
            
            NVarRemoved = 2
            
        else: # Otkin et al. Method uses FDII
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([14]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([14]))
            TrainLabel = yTrain[:,5]
            TestLabel  = yTest[:,5]
            
            NVarRemoved = 1
        
        print('Creating the ' + Model + 's for the ' + method + ' et al. method.')
        if (Model == 'RF') | (Model == 'Random Forest'):
            # Create random forest based on the best parameters
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities 
            if method == 'Christian':
                ChProb, ChWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                ChThresh = 0.02
                
            elif method == 'Noguera':
                NogProb, NogWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                NogThresh = 0.06
                
            elif method == 'Li':
                LiProb, LiWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                LiThresh = 0.05
                
            elif method == 'Liu':
                LiuProb, LiuWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                LiuThresh = 0.05
                
            elif method == 'Pendergrass':
                PeProb, PeWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                PeThresh = 0.01
                
            else:
                OtProb, OtWeights = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')
                
                OtThresh = 0.03
                
        elif (Model == 'SVM') | (Model == 'Support Vector Machine'):
            # Create SVM based on the best parameters
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities
            if method == 'Christian':
                ChProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                ChThresh = 0.02
                
            elif method == 'Noguera':
                NogProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                NogThresh = 0.06
                
            elif method == 'Li':
                LiProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                LiThresh = 0.05
                
            elif method == 'Liu':
                LiuProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                LiuThresh = 0.05
                
            elif method == 'Pendergrass':
                PeProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                PeThresh = 0.01
                
            else:
                OtProb = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
                
                OtThresh = 0.03
                
        elif (Model == 'ANN') | (Model == 'Nueral Network'):
            # Create ANN based on the best parameters
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities
            if method == 'Christian':
                ChProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15,))
                
                ChThresh = 0.01
                
            elif method == 'Noguera':
                NogProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15, 15))
                
                NogThresh = 0.04
                
            elif method == 'Li':
                LiProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15,))
                
                LiThresh = 0.03
                
            elif method == 'Liu':
                LiuProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15,))
                
                LiuThresh = 0.03
                
            elif method == 'Pendergrass':
                PeProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15, 15))
                
                PeThresh = 0.01
                
            else:
                OtProb = ANNModel(TrainData, TrainLabel, TestData, layers = (15, 15))
                
                OtThresh = 0.03

        else: ##### Add more models here
            pass
        
    print('Evaluating the models.')
    TPRCh, FPRCh, EntCh, R2Ch, RMSECh, CpCh, AICCh, BICCh, AccCh, PrecCh, RecallCh, F1Ch, SpecCh, RiskCh, AUCCh, YoudCh, YoudThreshCh, dCh, dThreshCh = EvaluateModel(ChProb[:,1], TestLabel, N = (NVar - NVarRemoved), ProbThreshold = ChThresh)
    TPRNog, FPRNog, EntNog, R2Nog, RMSENog, CpNog, AICNog, BICNog, AccNog, PrecNog, RecallNog, F1Nog, SpecNog, RiskNog, AUCNog, YoudNog, YoudThreshNog, dNog, dThreshNog = EvaluateModel(NogProb[:,1], TestLabel, N = (NVar - NVarRemoved), ProbThreshold = NogThresh)
    TPRLiu, FPRLiu, EntLiu, R2Liu, RMSELiu, CpLiu, AICLiu, BICLiu, AccLiu, PrecLiu, RecallLiu, F1Liu, SpecLiu, RiskLiu, AUCLiu, YoudLiu, YoudThreshLiu, dLiu, dThreshLiu = EvaluateModel(LiuProb[:,1], TestLabel, N = (NVar - NVarRemoved), ProbThreshold = LiuThresh)
    TPRPe, FPRPe, EntPe, R2Pe, RMSEPe, CpPe, AICPe, BICPe, AccPe, PrecPe, RecallPe, F1Pe, SpecPe, RiskPe, AUCPe, YoudPe, YoudThreshPe, dPe, dThreshPe = EvaluateModel(PeProb[:,1], TestLabel, N = (NVar - NVarRemoved), ProbThreshold = PeThresh)
    TPROt, FPROt, EntOt, R2Ot, RMSEOt, CpOt, AICOt, BICOt, AccOt, PrecOt, RecallOt, F1Ot, SpecOt, RiskOt, AUCOt, YoudOt, YoudThreshOt, dOt, dThreshOt = EvaluateModel(OtProb[:,1], TestLabel, N = (NVar - NVarRemoved), ProbThreshold = OtThresh)
    
    
    # Output the model performance
    
    #   Cross-Entropy
    print('The Christian et al. Method has a cross-entropy of: %4.2f' %EntCh)
    print('The Nogeura et al. Method has a cross-entropy of: %4.2f' %EntNog)
    print('The Liu et al. Method has a cross-entropy of: %4.2f' %EntLiu)
    print('The Pendergrass et al. Method has a cross-entropy of: %4.2f' %EntPe)
    print('The Otkin et al. Method has a cross-entropy of: %4.2f' %EntOt)
    print('\n')
    
    #   Adjusted-R^2
    print('The Christian et al. Method has an Adjusted-R^2 of: %4.2f' %R2Ch)
    print('The Nogeura et al. Method has an Adjusted-R^2 of: %4.2f' %R2Nog)
    print('The Liu et al. Method has an Adjusted-R^2 of: %4.2f' %R2Liu)
    print('The Pendergrass et al. Method has an Adjusted-R^2 of: %4.2f' %R2Pe)
    print('The Otkin et al. Method has an Adjusted-R^2 of: %4.2f' %R2Ot)
    print('\n')
    
    #   RMSE
    print('The Christian et al. Method has a RMSE of: %4.2f' %RMSECh)
    print('The Nogeura et al. Method has a RMSE of: %4.2f' %RMSENog)
    print('The Liu et al. Method has a RMSE of: %4.2f' %RMSELiu)
    print('The Pendergrass et al. Method has a RMSE of: %4.2f' %RMSEPe)
    print('The Otkin et al. Method has a RMSE of: %4.2f' %RMSEOt)
    print('\n')
    
    #   Cp
    print('The Christian et al. Method has a Cp of: %4.2f' %CpCh)
    print('The Nogeura et al. Method has a Cp of: %4.2f' %CpNog)
    print('The Liu et al. Method has a Cp of: %4.2f' %CpLiu)
    print('The Pendergrass et al. Method has a Cp of: %4.2f' %CpPe)
    print('The Otkin et al. Method has a Cp of: %4.2f' %CpOt)
    print('\n')
    
    #   AIC
    print('The Christian et al. Method has a AIC of: %4.2f' %AICCh)
    print('The Nogeura et al. Method has a AIC of: %4.2f' %AICNog)
    print('The Liu et al. Method has a AIC of: %4.2f' %AICLiu)
    print('The Pendergrass et al. Method has a AIC of: %4.2f' %AICPe)
    print('The Otkin et al. Method has a AIC of: %4.2f' %AICOt)
    print('\n')
    
    #   BIC
    print('The Christian et al. Method has a BIC of: %4.2f' %BICCh)
    print('The Nogeura et al. Method has a BIC of: %4.2f' %BICNog)
    print('The Liu et al. Method has a BIC of: %4.2f' %BICLiu)
    print('The Pendergrass et al. Method has a BIC of: %4.2f' %BICPe)
    print('The Otkin et al. Method has a BIC of: %4.2f' %BICOt)
    print('\n')
    
    #   Accuracy
    print('The Christian et al. Method has a Accuracy of: %4.2f' %AccCh)
    print('The Nogeura et al. Method has a Accuracy of: %4.2f' %AccNog)
    print('The Liu et al. Method has a Accuracy of: %4.2f' %AccLiu)
    print('The Pendergrass et al. Method has a Accuracy of: %4.2f' %AccPe)
    print('The Otkin et al. Method has a Accuracy of: %4.2f' %AccOt)
    print('\n')
    
    #   Precision
    print('The Christian et al. Method has a Precision of: %4.2f' %PrecCh)
    print('The Nogeura et al. Method has a Precision of: %4.2f' %PrecNog)
    print('The Liu et al. Method has a Precision of: %4.2f' %PrecLiu)
    print('The Pendergrass et al. Method has a Precision of: %4.2f' %PrecPe)
    print('The Otkin et al. Method has a Precision of: %4.2f' %PrecOt)
    print('\n')
    
    #   Recall
    print('The Christian et al. Method has a Recall of: %4.2f' %RecallCh)
    print('The Nogeura et al. Method has a Recall of: %4.2f' %RecallNog)
    print('The Liu et al. Method has a Recall of: %4.2f' %RecallLiu)
    print('The Pendergrass et al. Method has a Recall of: %4.2f' %RecallPe)
    print('The Otkin et al. Method has a Recall of: %4.2f' %RecallOt)
    print('\n')
    
    #   F1-Score
    print('The Christian et al. Method has a F1-Score of: %4.2f' %F1Ch)
    print('The Nogeura et al. Method has a F1-Score of: %4.2f' %F1Nog)
    print('The Liu et al. Method has a F1-Score of: %4.2f' %F1Liu)
    print('The Pendergrass et al. Method has a F1-Score of: %4.2f' %F1Pe)
    print('The Otkin et al. Method has a F1-Score of: %4.2f' %F1Ot)
    print('\n')
    
    #   Specificity
    print('The Christian et al. Method has a Specificity of: %4.2f' %SpecCh)
    print('The Nogeura et al. Method has a Specificity of: %4.2f' %SpecNog)
    print('The Liu et al. Method has a Specificity of: %4.2f' %SpecLiu)
    print('The Pendergrass et al. Method has a Specificity of: %4.2f' %SpecPe)
    print('The Otkin et al. Method has a Specificity of: %4.2f' %SpecOt)
    print('\n')
    
    #   Risk
    print('The Christian et al. Method has a Risk of: %4.2f' %RiskCh)
    print('The Nogeura et al. Method has a Risk of: %4.2f' %RiskNog)
    print('The Liu et al. Method has a Risk of: %4.2f' %RiskLiu)
    print('The Pendergrass et al. Method has a Risk of: %4.2f' %RiskPe)
    print('The Otkin et al. Method has a Risk of: %4.2f' %RiskOt)
    print('\n')
    
    #   AUC
    print('The Christian et al. Method has an AUC of: %4.2f' %AUCCh)
    print('The Nogeura et al. Method has an AUC of: %4.2f' %AUCNog)
    print('The Liu et al. Method has an AUC of: %4.2f' %AUCLiu)
    print('The Pendergrass et al. Method has an AUC of: %4.2f' %AUCPe)
    print('The Otkin et al. Method has an AUC of: %4.2f' %AUCOt)
    print('\n')
    
    #   Youden Index
    print('The Christian et al. Method has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudCh, YoudThreshCh))
    print('The Nogeura et al. Method has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudNog, YoudThreshNog))
    print('The Liu et al. Method has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudLiu, YoudThreshLiu))
    print('The Pendergrass et al. Method has amaximum Youden index of %4.2f at the threshold of %4.3f' %(YoudPe, YoudThreshPe))
    print('The Otkin et al. Method has amaximum Youden index of %4.2f at the threshold of %4.3f' %(YoudOt, YoudThreshOt))
    print('\n')
    
    #   Distance from leftmost corner of ROC curve
    print('The Christian et al. Method has a minimum distance of %4.2f at the threshold of %4.3f' %(dCh, dThreshCh))
    print('The Nogeura et al. Method has a minimum distance of %4.2f at the threshold of %4.3f' %(dNog, dThreshNog))
    print('The Liu et al. Method has a minimum distance of %4.2f at the threshold of %4.3f' %(dLiu, dThreshLiu))
    print('The Pendergrass et al. Method has a minimum distance of %4.2f at the threshold of %4.3f' %(dPe, dThreshPe))
    print('The Otkin et al. Method has a minimum distance of %4.2f at the threshold of %4.3f' %(dOt, dThreshOt))
    print('\n')
    
    # Feature Importance
    if (Model == 'RF') | (Model == 'Random Forest'):
        print('The feature importance for the Christian et al. Method is:', ChWeights)
        print('The feature importance for the Noguera et al. Method is:', NogWeights)
        print('The feature importance for the Liu et al. Method is:', LiuWeights)
        print('The feature importance for the Pendergrass et al. Method is:', PeWeights)
        print('The feature importance for the Otkin et al. Method is:', OtWeights)
        print('\n')
    else:
        pass
    
    
    # Finally, create and save a ROC curve
    fig = plt.figure(figsize = [14,14])
    ax = fig.add_subplot(1,1,1)
    
    #   Set the title
    ax.set_title('Receiver Operating Characteristic Curve for ' + Model + 's', fontsize = 24)
    
    #   Create the plots
    ax.plot(FPRCh, TPRCh, 'r-', linewidth = 2.0, label = 'Christian et al. 2019 Method')
    ax.plot(FPRNog, TPRNog, 'b-', linewidth = 2.0, label = 'Noguera et al. 2020 Method')
    ax.plot(FPRLiu, TPRLiu, 'k-', linewidth = 2.0, label = 'Liu et al. 2020 Method')
    ax.plot(FPRPe, TPRPe, 'g-', linewidth = 2.0, label = 'Pendergrass et al. 2020 Method')
    ax.plot(FPROt, TPROt, 'c-', linewidth = 2.0, label = 'Otkin et al. 2021 Method')
    
    #   Set the legend
    ax.legend(loc = 'best', fontsize = 33)
    
    #   Set the figure limits and labels
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    
    ax.set_xlabel('False Positive Rate', fontsize = 22)
    ax.set_ylabel('True Positive Rate', fontsize = 22)
    
    #   Set the tick sizes
    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(22)
    
    plt.savefig('./Figures/' + Model + '_ROC.png', bbox_inches = 'tight')
    plt.show(block = False)
    

#%%
# cell
    
# Define a function to create climatology and case study maps for the Christian and Otkin methods (since they have good performance and they have good climatologies to compare)
def ModelPredictions(Train, Label, Model, lat, lon, Mask, months, years):
    '''
    
    '''
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    
    
    xTrain = Train[:,ind,:]; yTrain = Label[:,ind,:]
    xTest  = Train[:,:,:]
    
    IJTrain, Ttrain, NVar = xTrain.shape
    IJTrain, Ttrain, NMethods = yTrain.shape
    
    IJTest, T, NVar = xTest.shape
    
    # Reorder data into 2D matrices
    xTrain = xTrain.reshape(IJTrain*Ttrain, NVar, order = 'F')
    yTrain = yTrain.reshape(IJTrain*Ttrain, NMethods, order = 'F')
    
    xTest = xTest.reshape(IJTest*T, NVar, order = 'F')
    
    I, J = lon.shape
    
    # Next, Start performing SL models for each method.
    ##### Remember to add Li
    Methods = ['Christian', 'Liu', 'Otkin']
    
    CaseYears = [1988, 2000, 2003, 2011, 2012, 2017, 2019]
    
    AllYears = np.unique(years)
    
    for method in Methods:
        if method == 'Christian': # Christian et al. method uses SESR
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([0, 1]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([0, 1]))
            TrainLabel = yTrain[:,0]
            
            NVarRemoved = 2
            FullMethod = 'Christian et al. 2019'
            
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities 
            Thresh = 0.02
            
        elif method == 'Liu': # Christian et al. method uses SESR
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([ ]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([ ]))
            TrainLabel = yTrain[:,3]
            
            NVarRemoved = 2
            FullMethod = 'Christian et al. 2019'
            
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities 
            Thresh = 0.03
            
        else: # Otkin et al. Method uses FDII
            TrainData = ColumnRemoval(xTrain, cols = np.asarray([14]))
            TestData  = ColumnRemoval(xTest, cols = np.asarray([14]))
            TrainLabel = yTrain[:,5]
            
            NVarRemoved = 1
            FullMethod = 'Otkin et al. 2021'
            
            # The Probability threshholds are based on the maximum Youden and minimum distance probabilities
            Thresh = 0.03
            
        # Create the model
        if (Model == 'RF') | (Model == 'Random Forest'):
            # Create random forest based on the best parameters
            # Train the model with all the data now that it has been tested and predict FD for all datapoints
            Prob, _ = RFModel(TrainData, TrainLabel, TestData, N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto', NJobs = -1)
            FullModel = '100 Tree Random Forest'
            
        elif (Model == 'SVM') | (Model == 'Support Vector Machine'):
            # Create SVM based on the best parameters
            # Train the model with all the data now that it has been tested and predict FD for all datapoints
            Prob = SVMModel(TrainData, TrainLabel, TestData, Kernel = 'rbf', RegParam = 1.0, Gamma = 'scale')
            FullModel = 'Radial Basis SVM'
            
        elif (Model == 'ANN') | (Model == 'Nueral Network'):
            # Create ANN based on the best parameters
            # Train the model with all the data now that it has been tested and predict FD for all datapoints
            Prob = ANNModel(TrainData, TrainLabel, TestData, layers = (15, 15))
            FullModel = '2 layer 15 node ANN'
        else:
            pass # Add more models
        
        # Create the predictions based on the thressholds
        Predictions = np.where(Prob[:,1] >= Thresh, 1, 0)
        
        # Next reshape the data back into a 2D array
        Predictions = Predictions.reshape(IJTrain, T, order = 'F')
        
        # Readd sea datapoints to recreate the full map.
        FullPred = np.ones((I*J, T)) * np.nan
        ij_land = 0
        for ij in range(I*J):
            if Mask[ij] == 0:
                FullPred[ij,:] = np.nan
            else:
                FullPred[ij,:] = Predictions[ij_land,:]
                
                ij_land = ij_land + 1
                
        # Reshape the data into a full 3D map
        FullPred = FullPred.reshape(I, J, T, order = 'F')
        
        
        # Create the climatology map 'Percent of Years from 1979 - 2020 with Otkin et al. 2021 Flash Drought' + '\n' + 'as Predicted by a 100 Tree Random Forest'
        FDClimatologyMap(FullPred, lat, lon, AllYears, months, years, title = 'Percent of Years from 1979 - 2020 with ' + method + ' Flash Drought' + '\n' + 'as Predicted by a ' + FullModel, savename = method + '_' + Model + '_Predicted_Climatology.png')
        
        # Create the case study maps
        for cy in CaseYears:
            FDAnnualMaps(FullPred, lat, lon, cy, months, years, title = 'Flash Drought for ' + str(cy) + ' Predicted by a ' + FullModel + ' for the ' + FullMethod + ' Method', savename = method + '_' + Model + '_CaseStudy' + str(cy) + '.png')


#%%
# cell 7
# Load in the index data

path = './Data/Indices/'

sesr  = LoadNC('sesr', 'sesr.NARR.CONUS.pentad.nc', path = path)
sedi  = LoadNC('sedi', 'sedi.NARR.CONUS.pentad.nc', path = path)
spei  = LoadNC('spei', 'spei.NARR.CONUS.pentad.nc', path = path)
sapei = LoadNC('sapei', 'sapei.NARR.CONUS.pentad.nc', path = path)
eddi  = LoadNC('eddi', 'eddi.NARR.CONUS.pentad.nc', path = path)
smi   = LoadNC('smi', 'smi.NARR.CONUS.pentad.nc', path = path)
sodi  = LoadNC('sodi', 'sodi.NARR.CONUS.pentad.nc', path = path)
fdii  = LoadNC('fdii', 'fdii.NARR.CONUS.pentad.nc', path = path)

# Load the FD_INT variable as well.
FDInt  = LoadNC('ric', 'fd_int.NARR.CONUS.pentad.nc', path = path)

#%%
# cell 8
# Load the flash drought data

path = './Data/FD_Data/'

ChFD  = LoadNC('chfd', 'ChristianFD.NARR.CONUS.pentad.nc', path = path)
NogFD = LoadNC('nogfd', 'NogueraFD.NARR.CONUS.pentad.nc', path = path)
# LiFD  = LoadNC('lifd', 'LiFD.NARR.CONUS.pentad.nc', path = path)
LiuFD = LoadNC('liufd', 'LiuFD.NARR.CONUS.pentad.nc', path = path)
PeFD  = LoadNC('pegfd', 'PendergrassFD.NARR.CONUS.pentad.nc', path = path)
OtFD  = LoadNC('otfd', 'OtkinFD.NARR.CONUS.pentad.nc', path = path)


#%%
# cell 9
# Load the mask data incase it is needed


# Create a function to load 2D data
def load2Dnc(filename, SName, path = './Data/'):
    '''
    This function loads 2 dimensional .nc files (e.g., the lat or lon files/
    only spatial files). Function is simple as these files only contain the raw data.
    
    Inputs:
    - filename: The filename of the .nc file to be loaded.
    - SName: The short name of the variable in the .nc file (i.e., the name to
             call when loading the data)
    - Path: The path from the present direction to the directory the file is in.
    
    Outputs:
    - var: The main variable in the .nc file.
    '''
    
    with Dataset(path + filename, 'r') as nc:
        var = nc.variables[SName][:,:]
        
    return var


# Load the mask data
mask = load2Dnc('land.nc', 'land')
lat = load2Dnc('lat_narr.nc', 'lat') # Dataset is lat x lon
lon = load2Dnc('lon_narr.nc', 'lon') # Dataset is lat x lon

# Turn positive lon values into negative
for i in range(len(lon[:,0])):
    ind = np.where( lon[i,:] > 0 )[0]
    lon[i,ind] = -1*lon[i,ind]

# Turn mask from time x lat x lon into lat x lon x time
T, I, J = mask.shape

maskNew = np.ones((I, J, T)) * np.nan
maskNew[:,:,0] = mask[0,:,:] # No loop is needed since the time dimension has length 1

# Subset the data to the same subset as the criteria data
LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65
maskSub, LatSub, LonSub = SubsetData(maskNew, lat, lon, LatMin = LatMin, LatMax = LatMax,
                                     LonMin = LonMin, LonMax = LonMax)



#%%
# cell 10
# Interpolate missing values and replace sea values with 0. I.e., remove NaNs
I, J, T = sesr['sesr'].shape

x    = np.arange(-6.5, 6.5, (13/T)) # A variable covering the range of all indices with 1 entry for each time step
xFDI = np.arange(0, 70, (70/T)) # A variable covering the range of all FDII values with 1 entry for each time step
xBin = np.arange(0, 2, (2/T)) # A binary variable for all FD labels

# Repalce NaNs in the sea with 0
sesr['sesr'][maskSub[:,:,0] == 0] = 0
sedi['sedi'][maskSub[:,:,0] == 0] = 0
spei['spei'][maskSub[:,:,0] == 0] = 0
sapei['sapei'][maskSub[:,:,0] == 0] = 0
eddi['eddi'][maskSub[:,:,0] == 0] = 0
smi['smi'][maskSub[:,:,0] == 0] = 0
sodi['sodi'][maskSub[:,:,0] == 0] = 0
fdii['fdii'][maskSub[:,:,0] == 0] = 0

ChFD['chfd'][maskSub[:,:,0] == 0] = 0
NogFD['nogfd'][maskSub[:,:,0] == 0] = 0
# LiFD['lifd'][maskSub[:,:,0] == 0] = 0
LiuFD['liufd'][maskSub[:,:,0] == 0] = 0
PeFD['pegfd'][maskSub[:,:,0] == 0] = 0
OtFD['otfd'][maskSub[:,:,0] == 0]= 0


for i in range(I):
    for j in range(J):
        SESRind = np.isfinite(sesr['sesr'][i,j,:])
        if (maskSub[i,j,0] == 0):
            continue
        
        SESRind = np.isfinite(sesr['sesr'][i,j,:])
        SEDIind = np.isfinite(sedi['sedi'][i,j,:])
        SPEIind = np.isfinite(spei['spei'][i,j,:])
        SAPEIind = np.isfinite(sapei['sapei'][i,j,:])
        EDDIind = np.isfinite(eddi['eddi'][i,j,:])
        SMIind = np.isfinite(smi['smi'][i,j,:])
        SODIind = np.isfinite(sodi['sodi'][i,j,:])
        FDIIind = np.isfinite(fdii['fdii'][i,j,:])
        
        ChFDind = np.isfinite(ChFD['chfd'][i,j,:])
        NogFDind = np.isfinite(NogFD['nogfd'][i,j,:])
        # LiFDind = np.isfinite(LiFD['lifd'][i,j,:])
        LiuFDind = np.isfinite(LiuFD['liufd'][i,j,:])
        PeFDind = np.isfinite(PeFD['pegfd'][i,j,:])
        OtFDind = np.isfinite(OtFD['otfd'][i,j,:])
        
        
        if np.nansum(SESRind) == 0:
            sesr['sesr'][i,j,:] = 0
        else:
            interp_sesr = interpolate.interp1d(x[SESRind], sesr['sesr'][i,j,SESRind], kind = 'linear', fill_value = 'extrapolate')
            sesr['sesr'][i,j,:] = interp_sesr(x)
        
        if np.nansum(SEDIind) == 0:
            sedi['sedi'][i,j,:] = 0            
        else:
            interp_sedi = interpolate.interp1d(x[SEDIind], sedi['sedi'][i,j,SEDIind], kind = 'linear', fill_value = 'extrapolate')
            sedi['sedi'][i,j,:] = interp_sedi(x)
        
        if np.nansum(SPEIind) == 0:
            spei['spei'][i,j,:] = 0
        else:
            interp_spei = interpolate.interp1d(x[SPEIind], spei['spei'][i,j,SPEIind], kind = 'linear', fill_value = 'extrapolate')
            spei['spei'][i,j,:] = interp_spei(x)
        
        if np.nansum(SAPEIind) == 0:
            sapei['sapei'][i,j,:] = 0
        else:
            interp_sapei = interpolate.interp1d(x[SAPEIind], sapei['sapei'][i,j,SAPEIind], kind = 'linear', fill_value = 'extrapolate')
            sapei['sapei'][i,j,:] = interp_sapei(x)
        
        if np.nansum(EDDIind) == 0:
            eddi['eddi'][i,j,:] = 0
        else:
            interp_eddi = interpolate.interp1d(x[EDDIind], eddi['eddi'][i,j,EDDIind], kind = 'linear', fill_value = 'extrapolate')
            eddi['eddi'][i,j,:] = interp_eddi(x)
        
        if np.nansum(SMIind) == 0:
            smi['smi'][i,j,:] = 0
        else:
            interp_smi  = interpolate.interp1d(x[SMIind], smi['smi'][i,j,SMIind], kind = 'linear', fill_value = 'extrapolate')
            smi['smi'][i,j,:] = interp_smi(x)
        
        if np.nansum(SODIind) == 0:
            sodi['sodi'][i,j,:] = 0
        else:
            interp_sodi = interpolate.interp1d(x[SODIind], sodi['sodi'][i,j,SODIind], kind = 'linear', fill_value = 'extrapolate')
            sodi['sodi'][i,j,:] = interp_sodi(x)
        
        if np.nansum(FDIIind) == 0:
            fdii['fdii'][i,j,:] = 0
        else:
            interp_fdii = interpolate.interp1d(xFDI[FDIIind], fdii['fdii'][i,j,FDIIind], kind = 'linear', fill_value = 'extrapolate')
            fdii['fdii'][i,j,:] = interp_fdii(xFDI)
        
        if np.nansum(ChFDind) == 0:
            ChFD['chfd'][i,j,:] = 0
        else:
            interp_ChFD  = interpolate.interp1d(xBin[ChFDind], ChFD['chfd'][i,j,ChFDind], kind = 'linear', fill_value = 'extrapolate')
            ChFD['chfd'][i,j,:] = interp_ChFD(xBin)
        
        if np.nansum(NogFDind) == 0:
            NogFD['nogfd'][i,j,:] = 0
        else:
            interp_NogFD = interpolate.interp1d(xBin[NogFDind], NogFD['nogfd'][i,j,NogFDind], kind = 'linear', fill_value = 'extrapolate')
            NogFD['nogfd'][i,j,:] = interp_NogFD(xBin)
        
        # if np.nansum(LiFDind) == 0:
        #     LiFD['lifd'][i,j,:] = 0
        # else:
        #     interp_LiFD = interpolate.interp1d(xBin[LiFDind], LiFD['lifd'][i,j,LiFDind], kind = 'linear', fill_value = 'extrapolate')
        #     LiFD['lifd'][i,j,:] = interp_LiFD(xBin)
        
        if np.nansum(LiuFDind) == 0:
            LiuFD['liufd'][i,j,:] = 0
        else:
            interp_LiuFD = interpolate.interp1d(xBin[LiuFDind], LiuFD['liufd'][i,j,LiuFDind], kind = 'linear', fill_value = 'extrapolate')
            LiuFD['liufd'][i,j,:] = interp_LiuFD(xBin)
        
        if np.nansum(PeFDind) == 0:
            PeFD['pegfd'][i,j,:] = 0
        else:
            interp_PeFD  = interpolate.interp1d(xBin[PeFDind], PeFD['pegfd'][i,j,PeFDind], kind = 'linear', fill_value = 'extrapolate')
            PeFD['pegfd'][i,j,:] = interp_PeFD(xBin)
        
        if np.nansum(OtFDind) == 0:
            OtFD['otfd'][i,j,:]= 0
        else:
            interp_OtFD  = interpolate.interp1d(xBin[OtFDind], OtFD['otfd'][i,j,OtFDind], kind = 'linear', fill_value = 'extrapolate')
            OtFD['otfd'][i,j,:]= interp_OtFD(xBin)
            

            
# Turn interpolated FD labels to 1 or 0
ChFD['chfd'][ChFD['chfd'] >= 0.5] = 1
NogFD['nogfd'][NogFD['nogfd'] >= 0.5] = 1
# LiFD['lifd'][LiFD['lifd'] >= 0.5] = 1
LiuFD['liufd'][LiuFD['liufd'] >= 0.5] = 1
PeFD['pegfd'][PeFD['pegfd'] >= 0.5] = 1
OtFD['otfd'][OtFD['otfd'] >= 0.5] = 1

ChFD['chfd'][ChFD['chfd'] < 0.5] = 0
NogFD['nogfd'][NogFD['nogfd'] < 0.5] = 0
# LiFD['lifd'][LiFD['lifd'] < 0.5] = 0
LiuFD['liufd'][LiuFD['liufd'] < 0.5] = 0
PeFD['pegfd'][PeFD['pegfd'] < 0.5] = 0
OtFD['otfd'][OtFD['otfd'] < 0.5] = 0
            
            

#%%
# cell 11
# Determine the mean change in each index (minus FDII which has its own intensification component pre-calculated) to get the intensification.

I, J, T = sesr['sesr'].shape

Dsesr     = np.ones((I, J, T)) * np.nan
MeanDsesr = np.ones((I, J, T)) * np.nan

Dsedi     = np.ones((I, J, T)) * np.nan
MeanDsedi = np.ones((I, J, T)) * np.nan

Dspei     = np.ones((I, J, T)) * np.nan
MeanDspei = np.ones((I, J, T)) * np.nan

Dsapei     = np.ones((I, J, T)) * np.nan
MeanDsapei = np.ones((I, J, T)) * np.nan

Deddi     = np.ones((I, J, T)) * np.nan
MeanDeddi = np.ones((I, J, T)) * np.nan

Dsmi     = np.ones((I, J, T)) * np.nan
MeanDsmi = np.ones((I, J, T)) * np.nan

Dsodi     = np.ones((I, J, T)) * np.nan
MeanDsodi = np.ones((I, J, T)) * np.nan


Dsesr[:,:,1:]  = sesr['sesr'][:,:,1:] - sesr['sesr'][:,:,:-1]
Dsedi[:,:,1:]  = sedi['sedi'][:,:,1:] - sedi['sedi'][:,:,:-1]
Dspei[:,:,1:]  = spei['spei'][:,:,1:] - spei['spei'][:,:,:-1]
Dsapei[:,:,1:] = sapei['sapei'][:,:,1:] - sapei['sapei'][:,:,:-1]
Deddi[:,:,1:]  = eddi['eddi'][:,:,1:] - eddi['eddi'][:,:,:-1]
Dsmi[:,:,1:]   = smi['smi'][:,:,1:] - smi['smi'][:,:,:-1]
Dsodi[:,:,1:]  = sodi['sodi'][:,:,1:] - sodi['sodi'][:,:,:-1]


# Determine the mean change in the indices
# The mean change is over the period of a month
for t in range(T):
    if (t == 0) | (t == 1): # This sets the first values to 0 (no change/mean in the first two pentads)
        MeanDsesr[:,:,t]  = 0
        MeanDsedi[:,:,t]  = 0
        MeanDspei[:,:,t]  = 0
        MeanDsapei[:,:,t] = 0
        MeanDeddi[:,:,t]  = 0
        MeanDsmi[:,:,t]   = 0
        MeanDsodi[:,:,t]  = 0
        
    elif t < 5:
        MeanDsesr[:,:,t]  = np.nanmean(Dsesr[:,:,:t], axis = -1)
        MeanDsedi[:,:,t]  = np.nanmean(Dsedi[:,:,:t], axis = -1)
        MeanDspei[:,:,t]  = np.nanmean(Dspei[:,:,:t], axis = -1)
        MeanDsapei[:,:,t] = np.nanmean(Dsapei[:,:,:t], axis = -1)
        MeanDeddi[:,:,t]  = np.nanmean(Deddi[:,:,:t], axis = -1)
        MeanDsmi[:,:,t]   = np.nanmean(Dsmi[:,:,:t], axis = -1)
        MeanDsodi[:,:,t]  = np.nanmean(Dsodi[:,:,:t], axis = -1)
        
    else:
        MeanDsesr[:,:,t]  = np.nanmean(Dsesr[:,:,t-5:t], axis = -1)
        MeanDsedi[:,:,t]  = np.nanmean(Dsedi[:,:,t-5:t], axis = -1)
        MeanDspei[:,:,t]  = np.nanmean(Dspei[:,:,t-5:t], axis = -1)
        MeanDsapei[:,:,t] = np.nanmean(Dsapei[:,:,t-5:t], axis = -1)
        MeanDeddi[:,:,t]  = np.nanmean(Deddi[:,:,t-5:t], axis = -1)
        MeanDsmi[:,:,t]   = np.nanmean(Dsmi[:,:,t-5:t], axis = -1)
        MeanDsodi[:,:,t]  = np.nanmean(Dsodi[:,:,t-5:t], axis = -1)


# Delete the no longer necessary variables to conserve space
del Dsesr
del Dsedi
del Dspei
del Dsapei
del Deddi
del Dsmi
del Dsodi


#%%
# Remove sea points to remove potential bias
Lat1D = sesr['lat'].reshape(I*J, order = 'F')
Lon1D = sesr['lon'].reshape(I*J, order = 'F')
Mask1D = maskSub.reshape(I*J*1, order = 'F')

ind = np.where(Mask1D[:] == 0)[0]

Lat1DnoSea = np.delete(Lat1D, ind)
Lon1DnoSea = np.delete(Lon1D, ind)

sesr2D = sesr['sesr'].reshape(I*J, T, order = 'F')
MDsesr2D = MeanDsesr.reshape(I*J, T, order = 'F')

sesr2D = np.delete(sesr2D, ind, axis = 0)
MDsesr2D = np.delete(MDsesr2D, ind, axis = 0)

sedi2D = sedi['sedi'].reshape(I*J, T, order = 'F')
MDsedi2D = MeanDsedi.reshape(I*J, T, order = 'F')

sedi2D = np.delete(sedi2D, ind, axis = 0)
MDsedi2D = np.delete(MDsedi2D, ind, axis = 0)

spei2D = spei['spei'].reshape(I*J, T, order = 'F')
MDspei2D = MeanDspei.reshape(I*J, T, order = 'F')

spei2D = np.delete(spei2D, ind, axis = 0)
MDspei2D = np.delete(MDspei2D, ind, axis = 0)

sapei2D = sapei['sapei'].reshape(I*J, T, order = 'F')
MDsapei2D = MeanDsapei.reshape(I*J, T, order = 'F')

sapei2D = np.delete(sapei2D, ind, axis = 0)
MDsapei2D = np.delete(MDsapei2D, ind, axis = 0)

eddi2D = eddi['eddi'].reshape(I*J, T, order = 'F')
MDeddi2D = MeanDeddi.reshape(I*J, T, order = 'F')

eddi2D = np.delete(eddi2D, ind, axis = 0)
MDeddi2D = np.delete(MDeddi2D, ind, axis = 0)

smi2D = smi['smi'].reshape(I*J, T, order = 'F')
MDsmi2D = MeanDsmi.reshape(I*J, T, order = 'F')

smi2D = np.delete(smi2D, ind, axis = 0)
MDsmi2D = np.delete(MDsmi2D, ind, axis = 0)

sodi2D = sodi['sodi'].reshape(I*J, T, order = 'F')
MDsodi2D = MeanDsodi.reshape(I*J, T, order = 'F')

sodi2D = np.delete(sodi2D, ind, axis = 0)
MDsodi2D = np.delete(MDsodi2D, ind, axis = 0)

fdii2D = fdii['fdii'].reshape(I*J, T, order = 'F')

fdii2D = np.delete(fdii2D, ind, axis = 0)



ChFD2D = ChFD['chfd'].reshape(I*J, T, order = 'F')
ChFD2D = np.delete(ChFD2D, ind, axis = 0)

NogFD2D = NogFD['nogfd'].reshape(I*J, T, order = 'F')
NogFD2D = np.delete(NogFD2D, ind, axis = 0)

# LiFD2D = LiFD['lifd'].reshape(I*J, T, order = 'F')
# LiFD2D = np.delete(LiFD2D, ind, axis = 0)

LiuFD2D = LiuFD['liufd'].reshape(I*J, T, order = 'F')
LiuFD2D = np.delete(LiuFD2D, ind, axis = 0)

PeFD2D = PeFD['pegfd'].reshape(I*J, T, order = 'F')
PeFD2D = np.delete(PeFD2D, ind, axis = 0)

OtFD2D = OtFD['otfd'].reshape(I*J, T, order = 'F')
OtFD2D = np.delete(OtFD2D, ind, axis = 0)

#%%
# cell 12
# Organize the data into two variables x (training) and y (label)
IJ, T = sesr2D.shape

NIndices = 15 # Number of indices. Each index (except FDII [see cell 9]) has 2 values. The index is the drought part, the mean change is the intensification part
NMethods = 6 # Number of flash drought identification methods being investigated

x = np.ones((IJ, T, NIndices)) * np.nan
y = np.ones((IJ, T, NMethods)) * np.nan

x[:,:,0] = sesr2D # The first column of x contains SESR
x[:,:,1] = MDsesr2D # The second column of x contains the mean change in SESR
x[:,:,2] = sedi2D # The third column of x contains SEDI
x[:,:,3] = MDsedi2D # The fourth column of x contains the mean change in SEDI
x[:,:,4] = spei2D # The fifth column of x contains SPEI
x[:,:,5] = MDspei2D # The sixth column of x contains the mean change in SPEI
x[:,:,6] = sapei2D # The seventh column of x contains SAPEI
x[:,:,7] = MDsapei2D # The eighth column of x contains the mean change in SAPEI
x[:,:,8] = eddi2D # The nineth column of x contains EDDI
x[:,:,9] = MDeddi2D # The tenth column of x contains the mean change in EDDI
x[:,:,10] = smi2D # The eleventh column of x contains SMI
x[:,:,11] = MDsmi2D # The twelveth column of x contains the mean change in SMI
x[:,:,12] = sodi2D # The thirteenth column of x contains SODI
x[:,:,13] = MDsodi2D # The forteenth column of x contains the mean change in SODI
x[:,:,14] = fdii2D # The fifteenth column of x contains FDII


y[:,:,0] = ChFD2D # The first column in y contains FD identified using the Christian et al. method
y[:,:,1] = NogFD2D # The second column in y contains FD identified using the Noguera et al. method
# y[:,:,2] = LiFD2D # The third column in y contains FD identified using the Li et al. method
y[:,:,3] = LiuFD2D # The fourth column in y contains FD identified using the Liu et al. method
y[:,:,4] = PeFD2D # The fifth column in y contains FD identified using the Pendergrass et al. method
y[:,:,5] = OtFD2D # The sixth column in y contains FD identified using the Otkin et al. method


# x = np.ones((IJ*T, NIndices)) * np.nan
# y = np.ones((IJ*T, NMethods)) * np.nan


# # Place the data in their respective arrays. For posterity, comments will record which columns contains what variable
# x[:,0]  = sesr2D.reshape(IJ*T, order = 'F') # The first column of x contains SESR
# x[:,1]  = MDsesr2D.reshape(IJ*T, order = 'F') # The second column of x contains the mean change in SESR
# x[:,2]  = sedi2D.reshape(IJ*T, order = 'F') # The third column of x contains SEDI
# x[:,3]  = MDsedi2D.reshape(IJ*T, order = 'F') # The fourth column of x contains the mean change in SEDI
# x[:,4]  = spei2D.reshape(IJ*T, order = 'F') # The fifth column of x contains SPEI
# x[:,5]  = MDspei2D.reshape(IJ*T, order = 'F') # The sixth column of x contains the mean change in SPEI
# x[:,6]  = sapei2D.reshape(IJ*T, order = 'F') # The seventh column of x contains SAPEI
# x[:,7]  = MDsapei2D.reshape(IJ*T, order = 'F') # The eighth column of x contains the mean change in SAPEI
# x[:,8]  = eddi2D.reshape(IJ*T, order = 'F') # The nineth column of x contains EDDI
# x[:,9]  = MDeddi2D.reshape(IJ*T, order = 'F') # The tenth column of x contains the mean change in EDDI
# x[:,10] = smi2D.reshape(IJ*T, order = 'F') # The eleventh column of x contains SMI
# x[:,11] = MDsmi2D.reshape(IJ*T, order = 'F') # The twelveth column of x contains the mean change in SMI
# x[:,12] = sodi2D.reshape(IJ*T, order = 'F') # The thirteenth column of x contains SODI
# x[:,13] = MDsodi2D.reshape(IJ*T, order = 'F') # The forteenth column of x contains the mean change in SODI
# x[:,14] = fdii2D.reshape(IJ*T, order = 'F') # The fifteenth column of x contains FDII



# y[:,0] = ChFD2D.reshape(IJ*T, order = 'F')  # The first column in y contains FD identified using the Christian et al. method
# y[:,1] = NogFD2D.reshape(IJ*T, order = 'F') # The second column in y contains FD identified using the Noguera et al. method
# # y[:,2] = LiFD2D.reshape(IJ*T, order = 'F')  # The third column in y contains FD identified using the Li et al. method
# y[:,3] = LiuFD2D.reshape(IJ*T, order = 'F') # The fourth column in y contains FD identified using the Liu et al. method
# y[:,4] = PeFD2D.reshape(IJ*T, order = 'F') # The fifth column in y contains FD identified using the Pendergrass et al. method
# y[:,5] = OtFD2D.reshape(IJ*T, order = 'F')  # The sixth column in y contains FD identified using the Otkin et al. method

# Delete the redundant 2D data to conserve space
del sesr2D
del MDsesr2D
del sedi2D
del MDsedi2D
del spei2D
del MDspei2D
del sapei2D
del MDsapei2D
del eddi2D
del MDeddi2D
del smi2D
del MDsmi2D
del sodi2D
del MDsodi2D
del fdii2D

del ChFD2D
del NogFD2D
# del LiFD2D
del LiuFD2D
del PeFD2D
del OtFD2D


# #%%
# # cell 13
# # Seperate the data into training, validation, and test sets [For now, this separation is random. May separate based on years later.]

# # Note separation is based on which method is being investigated
# # Method = 'Christian'
# # Method = 'Noguera'
# # Method = 'Li'
# # Method = 'Liu'
# # Method = 'Pendergrass'
# Method = 'Otkin'

# if Method == 'Christian': # Christian et al. method uses SESR
#     DelCols = np.asarray([0, 1])
#     FDInd = 0
# elif Method == 'Noguera': # Noguera et al method uses SPEI
#     DelCols = np.asarray([4, 5])
#     FDInd = 1
# elif Method == 'Li': # Li et al. method uses SEDI
#     DelCols = np.asarray([2, 3])
#     FDInd = 2
# elif Method == 'Liu': # Liu et al. method uses soil moisture
#     DelCols = np.asarray([ ])
#     FDInd = 3
# elif Method == 'Pendergrass': # Penndergrass et al. method uses EDDI
#     DelCols = np.asarray([9, 10])
#     FDInd = 4
# else: # Otkin et al. Method uses FDII
#     DelCols = np.asarray([14])
#     FDInd = 5

# # Separate the data, removing the index used in the FD identification method (this removes potential bias of that index outweighing the others)
# xTrain, xVal, xTest, xSel, yTrain, yVal, yTest, ySel = Preprocessing(x, y[:,FDInd], DelCols)

# # Determine the sizes of the test data
# ITrain = yTrain.shape[0]
# IVal   = yVal.shape[0]
# ITest  = yTest.shape[0]

# NVar = xTrain.shape[-1] # Number of variables

#%%
# cell
# Split the data

# Model testing for random forests

### Random Forests

# Run the models using parallel processing. 
### NOTE, This is designed to run all the cores on the computer for the quickest performance. Then the computer CANNOT be used while this is running.

DetermineParameters(x, y, Model = 'Random Forest', NJobs = -1)


# # Remove unnecessary variables to conserve space
# del xTrain, xVal
# del yTrain, yVal

# The best performing model for the Christian et al. method was the 100 tree RF (200 trees was better, but the improvement was minor). Largest Youden index was around 0.02
# The best performing model for the Noguera et al. method was the 100 tree RF (200 only had minor improvement). Largest Youden index was around 0.06
# The best performing model for the Li et al. method was the # tree RF.
# The best performing model for the Liu et al. method was the 100 tree RF (200 only had minor improvement). Largest Youden index was around 0.05
# The best performing model for the Pendergrass et al. method was the 100 tree RF. Largest Youden index was around 0.01
# The best performing model for the Otkin et al. method was the 100 tree RF (200 only had minor improvement). Largest Youden index was around 0.03

#%%
# cell
# With the model parameters tested, make and compare models for each FD method. Start with RFs
    
### Main results with RFs

# Run the models using parallel processing. 
### NOTE, This is designed to run all the cores on the computer for the quickest performance. Then the computer CANNOT be used while this is running.


CreateSLModel(x, y, 'Random Forest')

# Create some climatologies and case studies using the RF predictions to further examine performance
ModelPredictions(x, y, 'RF', sesr['lat'], sesr['lon'], Mask1D, sesr['month'], sesr['year'])



#%%
# cell

# Model testing for boosted trees
    
### Boosted Trees





#%%
# cell

# Model testing for SVMs

### SVMs
DetermineParameters(x, y, Model = 'SVM')

# Other studies are fairly consistent in using the radial basis function kernel, but do not detail other parameters. Modified parameter for this run will be kernal functions.
#   May come back to this and toy with other parameters



# The best performing model for the Christian et al. method was the SVM with # kernal. Largest Youden index was around 0.02
# The best performing model for the Noguera et al. method was the SVM with # kernal. Largest Youden index was around 0.06
# The best performing model for the Li et al. method was the SVM with # kernal.
# The best performing model for the Liu et al. method was the SVM with # kernal. Largest Youden index was around 0.05
# The best performing model for the Pendergrass et al. method was the SVM with # kernal. Largest Youden index was around 0.01
# The best performing model for the Otkin et al. method was the SVM with # kernal. Largest Youden index was around 0.03

#%%
# cell
# With the model parameters tested, make and compare models for each FD method. Start with SVMs
    
### Main results with SVMs

# Run the models using parallel processing. 
### NOTE, This is designed to run all the cores on the computer for the quickest performance. Then the computer CANNOT be used while this is running.


CreateSLModel(x, y, 'SVM')

# Create some climatologies and case studies using the RF predictions to further examine performance
ModelPredictions(x, y, 'SVM', sesr['lat'], sesr['lon'], Mask1D, sesr['month'], sesr['year'])


#%%
# cell

# Model testing for traditional NNs    

### Nueral Networks
DetermineParameters(x, y, Model = 'ANN')

# Other studies are fairly consistent in using the radial basis function kernel, but do not detail other parameters. Modified parameter for this run will be kernal functions.
#   May come back to this and toy with other parameters



# The best performing model for the Christian et al. method was the ANN with 1 layer and 15 nodes. Largest Youden index was around 0.01 (Note they all performed about the same, but this one is simpler)
# The best performing model for the Noguera et al. method was the ANN with 2 layers and 15 nodes. Largest Youden index was around 0.04 (Note they all performed about the same, but this one had the best Recall)
# The best performing model for the Li et al. method was the ANN with # layers and # nodes.
# The best performing model for the Liu et al. method was the ANN with 1 layers and 15 nodes. Largest Youden index was around 0.03 (Note they all performed about the same, but this one is simpler)
# The best performing model for the Pendergrass et al. method was the ANN with 2 layers and 15 nodes. Largest Youden index was around 0.01
# The best performing model for the Otkin et al. method was the ANN with 2 layers and 15 nodes. Largest Youden index was around 0.03


#%%
# cell
# With the model parameters tested, make and compare models for each FD method. Start with ANNs
    
### Main results with ANNs

# Run the models using parallel processing. 
### NOTE, This is designed to run all the cores on the computer for the quickest performance. Then the computer CANNOT be used while this is running.


CreateSLModel(x, y, 'ANN')

# Create some climatologies and case studies using the RF predictions to further examine performance
ModelPredictions(x, y, 'ANN', sesr['lat'], sesr['lon'], Mask1D, sesr['month'], sesr['year'])

#%%
# cell

# Model testing for wavelets

### Wavelets





#%%
# cell

# This was used to make a prelimary figure
# # More Examination of model performance.
# OtProb, _ = RFModel(xTrain, yTrain, x[:,:-1], N_trees = 100, crit = 'gini', max_depth = None, max_features = 'auto')


# # Make prediction for the Christian et al. method using the maximum Youden index
# Pred = np.where(OtProb[:,1] >= 0.05, 1, 0) # For all points where the probability is greater than or equal to 0.02, 1 is assigned, and 0 otherwise.

# # Reshape the data into a 2D array.
# Pred = Pred.reshape(IJ, T, order = 'F')

# # Replace sea datapoints to get a full array. 
# FullPred = np.ones((I*J, T)) * np.nan

# ij_land = 0
# for ij in range(I*J):
#     if Mask1D[ij] == 0:
#         FullPred[ij,:] = np.nan
#     else: # Locations where the mask is 1 (land), predictions are placed. Mask == 0 (sea) are left as NaN
#         FullPred[ij,:] = Pred[ij_land,:]
        
#         ij_land = ij_land+1
    
# FullPred = FullPred.reshape(I, J, T, order = 'F')
    

# # Next, determine a climatology for the predicted data.
# years  = np.unique(sesr['year'])

# AnnFD = np.ones((I, J, years.size)) * np.nan

# # Calculate the average number of rapid intensifications and flash droughts in a year
# for y in range(years.size):
#     yInd = np.where( (years[y] == sesr['year']) & ((sesr['month'] >= 4) & (sesr['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
#     # Calculate the mean number of flash drought for each year    
#     AnnFD[:,:,y] = np.nanmean(FullPred[:,:,yInd], axis = -1)
    
#     # Turn nonzero values to 1 (each year gets 1 count to the total)    
#     AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
#                             AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    

# # Calculate the percentage number of years with rapid intensifications and flash droughts
# PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size

# # Turn 0 values into nan
# PerAnnFD = np.where(PerAnnFD != 0, PerAnnFD, np.nan)

# #### Create the Plot ####

# # Set colorbar information
# cmin = -20; cmax = 80; cint = 1
# clevs = np.arange(-20, cmax + cint, cint)
# nlevs = len(clevs)
# cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)


# # Get the normalized color values
# norm = mcolors.Normalize(vmin = 0, vmax = cmax)

# # Generate the colors from the orginal color map in range from [0, cmax]
# colors = cmap(np.linspace(1 - (cmax - 0)/(cmax - cmin), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
# colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white

# # Create a new colorbar cut from the colors in range [0, cmax.]
# ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)

# colorsNew = cmap(np.linspace(0, 1, cmap.N))
# colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
# cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)

# # Shapefile information
# # ShapeName = 'Admin_1_states_provinces_lakes_shp'
# ShapeName = 'admin_0_countries'
# CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

# CountriesReader = shpreader.Reader(CountriesSHP)

# USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
# NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# # Lonitude and latitude tick information
# lat_int = 10
# lon_int = 20

# LatLabel = np.arange(-90, 90, lat_int)
# LonLabel = np.arange(-180, 180, lon_int)

# LonFormatter = cticker.LongitudeFormatter()
# LatFormatter = cticker.LatitudeFormatter()

# # Projection information
# data_proj = ccrs.PlateCarree()
# fig_proj  = ccrs.PlateCarree()



# # Create the plots
# fig = plt.figure(figsize = [12, 10])


# # Flash Drought plot
# ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# # Set the flash drought title
# ax.set_title('Percent of Years from 1979 - 2020 with Otkin et al. 2021 Flash Drought' + '\n' + 'as Predicted by a 100 Tree Random Forest', size = 18)

# # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
# ax.add_feature(cfeature.STATES)
# ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
# ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# # Adjust the ticks
# ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax.set_yticklabels(LatLabel, fontsize = 18)
# ax.set_xticklabels(LonLabel, fontsize = 18)

# ax.xaxis.set_major_formatter(LonFormatter)
# ax.yaxis.set_major_formatter(LatFormatter)

# # Plot the flash drought data
# cs = ax.pcolormesh(sesr['lon'], sesr['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj, zorder = 1)

# # Set the map extent to the U.S.
# ax.set_extent([-130, -65, 23.5, 48.5])


# # Set the colorbar size and location
# cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])

# # Create the colorbar
# cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')

# # Set the colorbar label
# cbar.ax.set_ylabel('% of years with Flash Drought', fontsize = 18)

# # Set the colorbar ticks
# cbar.set_ticks(np.arange(0, 90, 10))
# cbar.ax.set_yticklabels(np.arange(0, 90, 10), fontsize = 16)

# # Save the figure
# plt.savefig('./Figures/Preliminary_RF_PredictedClimatology.png', bbox_inches = 'tight')
# plt.show(block = False)




#%%
# cell

# This was used to make a prelimary figure
# Finally, produce a figure for 2017    

# This was used to make a prelimary figure
# Pred2017 = np.zeros((I, J))
# NMonths = 12

# for m in range(NMonths):
#     ind = np.where( (sesr['year'] == 2017) & (sesr['month'] == m) )[0]
#     Pred2017 = np.where(((np.nansum(FullPred[:,:,ind], axis = -1) != 0 ) & (Pred2017 == 0)), m, Pred2017) # Points where there prediction for the month is nonzero (FD is predicted) and 
#                                                                                                           # Pred2017 does not have a value already, are given a value of m. Pred2017 is left alone otherwise.
    
# # Remove sea values
# Pred2017[maskSub[:,:,0] == 0] = np.nan
    
    
# # Create a figure to plot this.

# # Set colorbar information
# # cmin = 0; cmax = 12; cint = 1
# cmin = 3; cmax = 10; cint = 1
# clevs = np.arange(cmin, cmax + cint, cint)
# nlevs = len(clevs)
# cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

# # Shapefile information
# # ShapeName = 'Admin_1_states_provinces_lakes_shp'
# ShapeName = 'admin_0_countries'
# CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

# CountriesReader = shpreader.Reader(CountriesSHP)

# USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
# NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# # Lonitude and latitude tick information
# lat_int = 10
# lon_int = 20

# LatLabel = np.arange(-90, 90, lat_int)
# LonLabel = np.arange(-180, 180, lon_int)

# LonFormatter = cticker.LongitudeFormatter()
# LatFormatter = cticker.LatitudeFormatter()

# # Projection information
# data_proj = ccrs.PlateCarree()
# fig_proj  = ccrs.PlateCarree()



# # Create the plots
# fig = plt.figure(figsize = [12, 10])


# # Flash Drought plot
# ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# # Set the flash drought title
# ax.set_title('Flash Drought for 2017 Predicted by a 100 Tree Random Forest using the Otkin et al. 2021 Method', size = 18)

# # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
# ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
# ax.add_feature(cfeature.STATES)
# ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
# ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)

# # Adjust the ticks
# ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
# ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())

# ax.set_yticklabels(LatLabel, fontsize = 18)
# ax.set_xticklabels(LonLabel, fontsize = 18)

# ax.xaxis.set_major_formatter(LonFormatter)
# ax.yaxis.set_major_formatter(LatFormatter)

# # Plot the flash drought data
# cs = ax.pcolormesh(sesr['lon'], sesr['lat'], Pred2017, vmin = cmin, vmax = cmax,
#                   cmap = cmap, transform = data_proj, zorder = 1)

# # Set the map extent to the U.S.
# ax.set_extent([-130, -65, 23.5, 48.5])


# # Set the colorbar size and location
# cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])

# # Create the colorbar
# cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, norm = norm, orientation = 'vertical')


# # Set the colorbar ticks
# # cbar.set_ticks(np.arange(0, 12+1, 1))
# # cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 16)
# # cbar.set_ticks(np.arange(1, 11+1, 1.48))
# cbar.set_ticks(np.arange(5, 100+1, 10))
# cbar.ax.set_yticklabels(['No FD', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct'], fontsize = 16)

# # Save the figure
# plt.savefig('./Figures/Preliminary_RF_CaseStudy2017.png', bbox_inches = 'tight')
# plt.show(block = False)




    






