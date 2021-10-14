#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:52:45 2021

@author: stuartedris


This is the main script for the employment of machine learning to identify flash drought study.
This script takes in indces calculated from the Calculate_Indices script (training data) and the 
identified flash drought in the Calculate_FD script (label data) and identifies flash drought
using those indices (minus the index used to calculate flash drought). Several models are employed
(decision trees to set the process up, boosted trees, random forests, SVMs, and nueral networks).
Models are run for each flash drought identification method. Output results are given in the final
models, ROC curves, tables of performance statistics, weights (contribution of each index), etc.

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory

"""


#%%
# cell 1
#####################################
### Import some libraries ###########
#####################################

import os, sys, warnings
import numpy as np
import pathos.multiprocessing as pmp
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
def EvaluateModel(Probs, y, N = 15):
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
    ind  = np.where(CritThresh == 0.5)[0]
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

x = np.ones((IJ*T, NIndices)) * np.nan
y = np.ones((IJ*T, NMethods)) * np.nan


# Place the data in their respective arrays. For posterity, comments will record which columns contains what variable
x[:,0]  = sesr2D.reshape(IJ*T, order = 'F') # The first column of x contains SESR
x[:,1]  = MDsesr2D.reshape(IJ*T, order = 'F') # The second column of x contains the mean change in SESR
x[:,2]  = sedi2D.reshape(IJ*T, order = 'F') # The third column of x contains SEDI
x[:,3]  = MDsedi2D.reshape(IJ*T, order = 'F') # The fourth column of x contains the mean change in SEDI
x[:,4]  = spei2D.reshape(IJ*T, order = 'F') # The fifth column of x contains SPEI
x[:,5]  = MDspei2D.reshape(IJ*T, order = 'F') # The sixth column of x contains the mean change in SPEI
x[:,6]  = sapei2D.reshape(IJ*T, order = 'F') # The seventh column of x contains SAPEI
x[:,7]  = MDsapei2D.reshape(IJ*T, order = 'F') # The eighth column of x contains the mean change in SAPEI
x[:,8]  = eddi2D.reshape(IJ*T, order = 'F') # The nineth column of x contains EDDI
x[:,9]  = MDeddi2D.reshape(IJ*T, order = 'F') # The tenth column of x contains the mean change in EDDI
x[:,10] = smi2D.reshape(IJ*T, order = 'F') # The eleventh column of x contains SMI
x[:,11] = MDsmi2D.reshape(IJ*T, order = 'F') # The twelveth column of x contains the mean change in SMI
x[:,12] = sodi2D.reshape(IJ*T, order = 'F') # The thirteenth column of x contains SODI
x[:,13] = MDsodi2D.reshape(IJ*T, order = 'F') # The forteenth column of x contains the mean change in SODI
x[:,14] = fdii2D.reshape(IJ*T, order = 'F') # The fifteenth column of x contains FDII



y[:,0] = ChFD2D.reshape(IJ*T, order = 'F')  # The first column in y contains FD identified using the Christian et al. method
y[:,1] = NogFD2D.reshape(IJ*T, order = 'F') # The second column in y contains FD identified using the Noguera et al. method
# y[:,2] = LiFD2D.reshape(IJ*T, order = 'F')  # The third column in y contains FD identified using the Li et al. method
y[:,3] = LiuFD2D.reshape(IJ*T, order = 'F') # The fourth column in y contains FD identified using the Liu et al. method
y[:,4] = PeFD2D.reshape(IJ*T, order = 'F') # The fifth column in y contains FD identified using the Pendergrass et al. method
y[:,5] = OtFD2D.reshape(IJ*T, order = 'F')  # The sixth column in y contains FD identified using the Otkin et al. method

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


#%%
# cell 13
# Seperate the data into training, validation, and test sets [For now, this separation is random. May separate based on years later.]

# Note separation is based on which method is being investigated
Method = 'Christian'
# Method = 'Noguera'
# Method = 'Li'
# Method = 'Liu'
# Method = 'Pendergrass'
# Method = 'Otkin'

if Method == 'Christian': # Christian et al. method uses SESR
    DelCols = np.asarray([0, 1])
    FDInd = 0
elif Method == 'Noguera': # Noguera et al method uses SPEI
    DelCols = np.asarray([4, 5])
    FDInd = 1
elif Method == 'Li': # Li et al. method uses SEDI
    DelCols = np.asarray([2, 3])
    FDInd = 2
elif Method == 'Liu': # Liu et al. method uses soil moisture
    DelCols = np.asarray([ ])
    FDInd = 3
elif Method == 'Pendergrass': # Penndergrass et al. method uses EDDI
    DelCols = np.asarray([9, 10])
    FDInd = 4
else: # Otkin et al. Method uses FDII
    DelCols = np.asarray([15])
    FDInd = 5

# Separate the data, removing the index used in the FD identification method (this removes potential bias of that index outweighing the others)
xTrain, xVal, xTest, xSel, yTrain, yVal, yTest, ySel = Preprocessing(x, y[:,FDInd], DelCols)

# Determine the sizes of the test data
ITrain = yTrain.shape[0]
IVal   = yVal.shape[0]
ITest  = yTest.shape[0]

NVar = xTrain.shape[-1] # Number of variables


#%%
# cell 14
# Begin ML methods. Start with basic decision trees to start iron out the process of refining the model, displaying results, etc.

# Decision Trees

### Note this is more of a test cell to create a means for deciding and comparing models

# Both 10 branch models performed equally well. Go with default (GINI)

# Define a function to create and evaluate a decision tree model.
def TreeModel(xTrain, yTrain, xVal, crit, max_depth):
    '''


    '''
    
    # Make the decision tree
    Tree = tree.DecisionTreeClassifier(criterion = crit, max_depth = max_depth)
    
    # Train the tree
    Tree.fit(xTrain, yTrain)
    
    # Make probabilistic predictions
    Prob = Tree.predict_proba(xVal)
    
    return Prob
    
# Create a couple of decision tree models to test
print('Creating models')
ProbGini5 = TreeModel(xTrain, yTrain, xVal, crit = 'gini', max_depth = 5)
ProbGini10 = TreeModel(xTrain, yTrain, xVal, crit = 'gini', max_depth = 10)
ProbEnt5 = TreeModel(xTrain, yTrain, xVal, crit = 'gini', max_depth = 5)
ProbEnt10 = TreeModel(xTrain, yTrain, xVal, crit = 'gini', max_depth = 10)



# Evaluate the models
# Note the probabilities have 2 columns. Column 1 is the probability of that row being 0, and column 1 is the probability of that row being 1. The latter is used for most of these calculations
print('Evaluating models')
TPRGini5, FPRGini5, EntGini5, R2Gini5, RMSEGini5, CpGini5, AICGini5, BICGini5, AccGini5, PrecGini5, RecallGini5, F1Gini5, SpecGini5, RiskGini5, AUCGini5, YoudGini5, YoudThreshGini5, dGini5, dThreshGini5 = EvaluateModel(ProbGini5[:,1], yVal, N = NVar)
TPRGini10, FPRGini10, EntGini10, R2Gini10, RMSEGini10, CpGini10, AICGini10, BICGini10, AccGini10, PrecGini10, RecallGini10, F1Gini10, SpecGini10, RiskGini10, AUCGini10, YoudGini10, YoudThreshGini10, dGini10, dThreshGini10 = EvaluateModel(ProbGini10[:,1], yVal, N = NVar)
TPREnt5, FPREnt5, EntEnt5, R2Ent5, RMSEEnt5, CpEnt5, AICEnt5, BICEnt5, AccEnt5, PrecEnt5, RecallEnt5, F1Ent5, SpecEnt5, RiskEnt5, AUCEnt5, YoudEnt5, YoudThreshEnt5, dEnt5, dThreshEnt5 = EvaluateModel(ProbEnt5[:,1], yVal, N = NVar)
TPREnt10, FPREnt10, EntEnt10, R2Ent10, RMSEEnt10, CpEnt10, AICEnt10, BICEnt10, AccEnt10, PrecEnt10, RecallEnt10, F1Ent10, SpecEnt10, RiskEnt10, AUCEnt10, YoudEnt10, YoudThreshEnt10, dEnt10, dThreshEnt10 = EvaluateModel(ProbEnt10[:,1], yVal, N = NVar)



# Output the performance statistics

#   Cross-Entropy
print('The GINI 5 branch tree has a cross-entropy of: %4.2f' %EntGini5)
print('The GINI 10 branch tree has a cross-entropy of: %4.2f' %EntGini10)
print('The Entropy 5 branch tree has a cross-entropy of: %4.2f' %EntEnt5)
print('The Entropy 10 branch tree has a cross-entropy of: %4.2f' %EntEnt10)
print('\n')

#   Adjusted-R^2
print('The GINI 5 branch tree has an Adjusted-R^2 of: %4.2f' %R2Gini5)
print('The GINI 10 branch tree has an Adjusted-R^2 of: %4.2f' %R2Gini10)
print('The Entropy 5 branch tree has an Adjusted-R^2 of: %4.2f' %R2Ent5)
print('The Entropy 10 branch tree has an Adjusted-R^2 of: %4.2f' %R2Ent10)
print('\n')

#   RMSE
print('The GINI 5 branch tree has a RMSE of: %4.2f' %RMSEGini5)
print('The GINI 10 branch tree has a RMSE of: %4.2f' %RMSEGini10)
print('The Entropy 5 branch tree has a RMSE of: %4.2f' %RMSEEnt5)
print('The Entropy 10 branch tree has a RMSE of: %4.2f' %RMSEEnt10)
print('\n')

#   Cp
print('The GINI 5 branch tree has a Cp of: %4.2f' %CpGini5)
print('The GINI 10 branch tree has a Cp of: %4.2f' %CpGini10)
print('The Entropy 5 branch tree has a Cp of: %4.2f' %CpEnt5)
print('The Entropy 10 branch tree has a Cp of: %4.2f' %CpEnt10)
print('\n')

#   AIC
print('The GINI 5 branch tree has a AIC of: %4.2f' %AICGini5)
print('The GINI 10 branch tree has a AIC of: %4.2f' %AICGini10)
print('The Entropy 5 branch tree has a AIC of: %4.2f' %AICEnt5)
print('The Entropy 10 branch tree has a AIC of: %4.2f' %AICEnt10)
print('\n')

#   BIC
print('The GINI 5 branch tree has a BIC of: %4.2f' %BICGini5)
print('The GINI 10 branch tree has a BIC of: %4.2f' %BICGini10)
print('The Entropy 5 branch tree has a BIC of: %4.2f' %BICEnt5)
print('The Entropy 10 branch tree has a BIC of: %4.2f' %BICEnt10)
print('\n')

#   Accuracy
print('The GINI 5 branch tree has a Accuracy of: %4.2f' %AccGini5)
print('The GINI 10 branch tree has a Accuracy of: %4.2f' %AccGini10)
print('The Entropy 5 branch tree has a Accuracy of: %4.2f' %AccEnt5)
print('The Entropy 10 branch tree has a Accuracy of: %4.2f' %AccEnt10)
print('\n')

#   Precision
print('The GINI 5 branch tree has a Precision of: %4.2f' %PrecGini5)
print('The GINI 10 branch tree has a Precision of: %4.2f' %PrecGini10)
print('The Entropy 5 branch tree has a Precision of: %4.2f' %PrecEnt5)
print('The Entropy 10 branch tree has a Precision of: %4.2f' %PrecEnt10)
print('\n')

#   Recall
print('The GINI 5 branch tree has a Recall of: %4.2f' %RecallGini5)
print('The GINI 10 branch tree has a Recall of: %4.2f' %RecallGini10)
print('The Entropy 5 branch tree has a Recall of: %4.2f' %RecallEnt5)
print('The Entropy 10 branch tree has a Recall of: %4.2f' %RecallEnt10)
print('\n')

#   F1-Score
print('The GINI 5 branch tree has a F1-Score of: %4.2f' %F1Gini5)
print('The GINI 10 branch tree has a F1-Score of: %4.2f' %F1Gini10)
print('The Entropy 5 branch tree has a F1-Score of: %4.2f' %F1Ent5)
print('The Entropy 10 branch tree has a F1-Score of: %4.2f' %F1Ent10)
print('\n')

#   Specificity
print('The GINI 5 branch tree has a Specificity of: %4.2f' %SpecGini5)
print('The GINI 10 branch tree has a Specificity of: %4.2f' %SpecGini10)
print('The Entropy 5 branch tree has a Specificity of: %4.2f' %SpecEnt5)
print('The Entropy 10 branch tree has a Specificity of: %4.2f' %SpecEnt10)
print('\n')

#   Risk
print('The GINI 5 branch tree has a Risk of: %4.2f' %RiskGini5)
print('The GINI 10 branch tree has a Risk of: %4.2f' %RiskGini10)
print('The Entropy 5 branch tree has a Risk of: %4.2f' %RiskEnt5)
print('The Entropy 10 branch tree has a Risk of: %4.2f' %RiskEnt10)
print('\n')

#   AUC
print('The GINI 5 branch tree has an AUC of: %4.2f' %AUCGini5)
print('The GINI 10 branch tree has an AUC of: %4.2f' %AUCGini10)
print('The Entropy 5 branch tree has an AUC of: %4.2f' %AUCEnt5)
print('The Entropy 10 branch tree has an AUC of: %4.2f' %AUCEnt10)
print('\n')

#   Youden Index
print('The GINI 5 branch tree has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudGini5, YoudThreshGini5))
print('The GINI 10 branch tree has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudGini10, YoudThreshGini10))
print('The Entropy 5 branch tree has a maximum Youden index of %4.2f at the threshold of %4.3f' %(YoudEnt5, YoudThreshEnt5))
print('The Entropy 10 branch tree has amaximum Youden index of %4.2f at the threshold of %4.3f' %(YoudEnt10, YoudThreshEnt10))
print('\n')

#   Distance from leftmost corner of ROC curve
print('The GINI 5 branch tree has a minimum distance of %4.2f at the threshold of %4.3f' %(dGini5, dThreshGini5))
print('The GINI 10 branch tree has a minimum distance of %4.2f at the threshold of %4.3f' %(dGini10, dThreshGini10))
print('The Entropy 5 branch tree has a minimum distance of %4.2f at the threshold of %4.3f' %(dEnt5, dThreshEnt5))
print('The Entropy 10 branch tree has a minimum distance of %4.2f at the threshold of %4.3f' %(dEnt10, dThreshEnt10))
print('\n')


# Plot the ROC curve for the models
fig = plt.figure(figsize = [14,14])
ax = fig.add_subplot(1,1,1)

#   Set the title
ax.set_title('Receiver Operating Characteristic Curve for the Four Decision Trees', fontsize = 24)

#   Create the plots
ax.plot(FPRGini5, TPRGini5, 'r-', linewidth = 2.0, label = 'GINI 5 Branch Model')
ax.plot(FPRGini10, TPRGini10, 'b-', linewidth = 2.0, label = 'GINI 10 Branch Model')
ax.plot(FPREnt5, TPREnt5, 'k-', linewidth = 2.0, label = 'Entropy 5 Branch Model')
ax.plot(FPREnt10, TPREnt10, 'g-', linewidth = 2.0, label = 'Entropy 5 Branch Model')

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
# cell 15


