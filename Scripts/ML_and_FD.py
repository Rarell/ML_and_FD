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
# Load in the index data

path = './Data/Indices/'

sesr  = LoadNC('sesr', 'sesr.NARR.CONUS.pentad.nc', path = path)
sedi  = LoadNC('sedi', 'sedi.NARR.CONUS.pentad.nc', path = path)
spei  = LoadNC('spei', 'spei.NARR.CONUS.pentad.nc', path = path)
sapei = LoadNC('sapei', 'spei.NARR.CONUS.pentad.nc', path = path)
eddi  = LoadNC('eddi', 'eddi.NARR.CONUS.pentad.nc', path = path)
smi   = LoadNC('smi', 'smi.NARR.CONUS.pentad.nc', path = path)
sodi  = LoadNC('sodi', 'sodi.NARR.CONUS.pentad.nc', path = path)
fdii  = LoadNC('fdii', 'fdii.NARR.CONUS.pentad.nc', path = path)

# Load the FD_INT variable as well.
FDInt  = LoadNC('ric', 'fd_int.NARR.CONUS.pentad.nc', path = path)

#%%
# cell 7
# Load the flash drought data

path = './Data/FD_Data/'

ChFD  = LoadNC('chfd', 'ChristianFD.NARR.CONUS.pentad.nc', path = path)
NogFD = LoadNC('nogfd', 'NogueraFD.NARR.CONUS.pentad.nc', path = path)
# LiFD  = LoadNC('lifd', 'LiFD.NARR.CONUS.pentad.nc', path = path)
LiuFD = LoadNC('liufd', 'LiuFD.NARR.CONUS.pentad.nc', path = path)
PeFD  = LoadNC('pefd', 'PendergrassFD.NARR.CONUS.pentad.nc', path = path)
OtFD  = LoadNC('otfd', 'OtkinFD.NARR.CONUS.pentad.nc', path = path)


#%%
# cell 8
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
# cell 9
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
    if t < 5:
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


#%%
# cell 10
# Organize the data into two variables x (training) and y (label)
        
NIndices = 15 # Number of indices. Each index (except FDII [see cell 9]) has 2 values. The index is the drought part, the mean change is the intensification part
NMethods = 6 # Number of flash drought identification methods being investigated

x = np.ones((I*J*T, NIndices)) * np.nan
y = np.ones((I*J*T, NMethods)) * np.nan

# Reorder data into 1D arrays
sesr1d   = sesr['sesr'].reshape(I*J*T, order = 'F')
mdsesr1d = MeanDsesr.reshape(I*J*T, order = 'F')

sedi1d   = sedi['sedi'].reshape(I*J*T, order = 'F')
mdsedi1d = MeanDsedi.reshape(I*J*T, order = 'F')

spei1d   = spei['spei'].reshape(I*J*T, order = 'F')
mdspei1d = MeanDspei.reshape(I*J*T, order = 'F')

sapei1d   = sapei['sapei'].reshape(I*J*T, order = 'F')
mdsapei1d = MeanDsapei.reshape(I*J*T, order = 'F')

eddi1d   = eddi['eddi'].reshape(I*J*T, order = 'F')
mdeddi1d = MeanDeddi.reshape(I*J*T, order = 'F')

smi1d   = smi['smi'].reshape(I*J*T, order = 'F')
mdsmi1d = MeanDsmi.reshape(I*J*T, order = 'F')

sodi1d   = sodi['sodi'].reshape(I*J*T, order = 'F')
mdsodi1d = MeanDsodi.reshape(I*J*T, order = 'F')

fdii1d  = fdii['fdii'].reshape(I*J*T, order = 'F')
fdint1d = FDInt['ric'].reshape(I*J*T, order = 'F')


ChFD1d  = ChFD['chfd'].reshape(I*J*T, order = 'F')
NogFD1d = NogFD['nogfd'].reshape(I*J*T, order = 'F')
# LiFD1d  = LiFD['lifd'].reshape(I*J*T, order = 'F')
LiuFD1d = LiuFD['liufd'].reshape(I*J*T, order = 'F')
PeFD1d  = PeFD['pefd'].reshape(I*J*T, order = 'F')
OtFD1d  = OtFD['otfd'].reshape(I*J*T, order = 'F')



# Place the data in their respective arrays. For posterity, comments will record which columns contains what variable
x[:,0]  = sesr1d # The first column of x contains SESR
x[:,1]  = mdsesr1d # The second column of x contains the mean change in SESR
x[:,2]  = sedi1d # The third column of x contains SEDI
x[:,3]  = mdsedi1d # The fourth column of x contains the mean change in SEDI
x[:,4]  = spei1d # The fifth column of x contains SPEI
x[:,5]  = mdspei1d # The sixth column of x contains the mean change in SPEI
x[:,6]  = sapei1d # The seventh column of x contains SAPEI
x[:,7]  = mdsapei1d # The eighth column of x contains the mean change in SAPEI
x[:,8]  = eddi1d # The nineth column of x contains EDDI
x[:,9]  = mdeddi1d # The tenth column of x contains the mean change in EDDI
x[:,10] = smi1d # The eleventh column of x contains SMI
x[:,11] = mdsmi1d # The twelveth column of x contains the mean change in SMI
x[:,12] = sodi1d # The thirteenth column of x contains SODI
x[:,13] = mdsodi1d # The forteenth column of x contains the mean change in SODI
x[:,14] = fdii1d # The fifteenth column of x contains FDII



y[:,0] = ChFD1d  # The first column in y contains FD identified using the Christian et al. method
y[:,1] = NogFD1d # The second column in y contains FD identified using the Noguera et al. method
# y[:,2] = LiFD1d  # The third column in y contains FD identified using the Li et al. method
y[:,3] = LiuFD1d # The fourth column in y contains FD identified using the Liu et al. method
y[:,4] = PeFD1d  # The fifth column in y contains FD identified using the Pendergrass et al. method
y[:,5] = OtFD1d  # The sixth column in y contains FD identified using the Otkin et al. method


#%%
# cell 11
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
xTrain, xVal, xTest, yTrain, yVal, yTest = Preprocessing(x, y[:,FDInd], DelCols)


#%%
# cell 12
# Begin ML methods. Start with basic decision trees to start iron out the process of refining the model, displaying results, etc.















