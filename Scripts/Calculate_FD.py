#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:25:25 2021

@author: stuartedris

This script is designed to take the indices created in the Calculate_Indices
script and identify flash drought using a number of indices. A number of flash
drought identification methods are used in this script, and each method corresponds
to a scrip (e.g., identifying flash drought with SESR will use and improved version 
of the Christian et al. 2019).

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory



Full citations for the referenced papers can be found at:
- Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1

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
# Create a function to write a variable to a .nc file
  
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', VarSName = 'tmp', description = 'Description', path = './Data/FD_Data/'):
    '''
    This function is deisgned write data, and additional information such as
    latitude and longitude and timestamps to a .nc file.
    
    Inputs:
    - var: The variable being written (lat x lon x time format).
    - lat: The latitude data with the same spatial grid as var.
    - lon: The longitude data with the same spatial grid as var.
    - dates: The timestamp for each pentad in var in a %Y-%m-%d format, same time grid as var.
    - filename: The filename of the .nc file being written.
    - sm: A boolean value to determin if soil moisture is being written. If true, an additional variable containing
          the soil depth information is provided.
    - VarName: The full name of the variable being written (for the nc description).
    - VarSName: The short name of the variable being written. I.e., the name used
                to call the variable in the .nc file.
    - description: A string descriping the data.
    - path: The path to the directory the data will be written in.
                
    Outputs:
    - None. Data is written to a .nc file.
    '''
    
    # Determine the spatial and temporal lengths
    I, J, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = description

        
        # Create the spatial and temporal dimensions
        nc.createDimension('x', size = I)
        nc.createDimension('y', size = J)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('x', 'y'))
        nc.createVariable('lon', lon.dtype, ('x', 'y'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create the main variable
        nc.createVariable(VarSName, var.dtype, ('x', 'y', 'time'))
        nc.variables[str(VarSName)][:,:,:] = var[:,:,:]


#%% 
# cell 4
# Calculate the climatological means and standard deviations
  
def CalculateClimatology(var, pentad = True):
    '''
    The function takes in a 3 dimensional variable (2 dimensional space and time)
    and calculates climatological values (mean and standard deviation) for each
    grid point and day in the year.
    
    Inputs:
    - var: 3 dimensional variable whose mean and standard deviation will be
           calculated.
    - pentad: Boolean (True/False) value giving if the time scale of var is 
              pentad (5 day average) or daily.
              
    Outputs:
    - ClimMean: Calculated mean of var for each day/pentad and grid point. 
                ClimMean as the same spatial dimensions as var and 365 (73)
                temporal dimension for daily (pentad) data.
    - ClimStd: Calculated standard deviation for each day/pentad and grid point.
               ClimStd as the same spatial dimensions as var and 365 (73)
               temporal dimension for daily (pentad) data.
    '''
    
    # Obtain the dimensions of the variable
    if len(var.shape) < 3:
        T = var.size
    else:
        I, J, T = var.shape
    
    # Count the number of years
    if pentad is True:
        yearLen = int(365/5)
    else:
        yearLen = int(365)
        
    NumYear = int(np.ceil(T/yearLen))
    
    # Create a variable for each day, assumed starting at Jan 1 and no
    #   leap years (i.e., each year is only 365 days each)
    day = np.ones((T)) * np.nan
    
    n = 0
    for i in range(1, NumYear+1):
        if i >= NumYear:
            day[n:T+1] = np.arange(1, len(day[n:T+1])+1)
        else:
            day[n:n+yearLen] = np.arange(1, yearLen+1)
        
        n = n + yearLen
    
    # Initialize the climatological mean and standard deviation variables
    if len(var.shape) < 3:
        ClimMean = np.ones((yearLen)) * np.nan
        ClimStd  = np.ones((yearLen)) * np.nan
    else:
        ClimMean = np.ones((I, J, yearLen)) * np.nan
        ClimStd  = np.ones((I, J, yearLen)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, yearLen+1):
        ind = np.where(i == day)[0]
        
        if len(var.shape) < 3:
            ClimMean[i-1] = np.nanmean(var[ind], axis = -1)
            ClimStd[i-1]  = np.nanstd(var[ind], axis = -1)
        else:
            ClimMean[:,:,i-1] = np.nanmean(var[:,:,ind], axis = -1)
            ClimStd[:,:,i-1]  = np.nanstd(var[:,:,ind], axis = -1)
    
    return ClimMean, ClimStd


#%%
# cell 5
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
# cell 6    
# Create a function for polynomial regression
def PolynomialRegress(X, Y, order = 1):
    '''
    A function designed to take in two vectors of equal size (X and Y) and perform a polynomial
    regression of the data. Function outputs the estimated data yhat, and the R^2 coefficient.
    
    Inputs:
    - X: The input x data
    - Y: The input y data that is being estimated
    - order: The order of the polynomial used in the regression
    
    Outputs:
    - yhat: The estimated value of Y using the polynomial regression
    - R2: The R^2 coefficient from the polynomial regression
    '''
    
    # Determine the size of the data
    T = len(X)
    
    # Initialize the model matrix E
    E = np.ones((T, order+1)) * np.nan
    
    # Fill the model matrix with data
    for n in range(order+1):
        if n == order:
            # For the last column, fill with 1, for the coefficient a_0
            E[:,n] = 1
        else:
            E[:,n] = x**(n+1)
            
    # Perform the polynomial regression
    invEtE = np.linalg.inv(np.dot(E.T, E))
    xhat = np.dot(np.dot(invEtE, E), Y)
    
    # Estimate yhat
    yhat = E.dot(xhat)
    
    # Determine the R^2 coefficient
    R2 = np.nanvar(yhat - np.nanmean(Y))/np.nanvar(Y)
    
    return yhat, R2


#%%
# cell 7
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
# cell 8
# Load the indices 
    
path = './Data/Indices/'

sesr = LoadNC('sesr', 'sesr.NARR.CONUS.pentad.nc', path = path)
spei = LoadNC('spei', 'spei.NARR.CONUS.pentad.nc', path = path)
eddi = LoadNC('eddi', 'eddi.NARR.CONUS.pentad.nc', path = path)
fdii = LoadNC('fdii', 'fdii.NARR.CONUS.pentad.nc', path = path)

SM   = LoadNC('soilm', 'soil_moisture.NARR.CONUS.pentad.nc', path = './Data/Processed_Data/')

# In addition, calculate a datetime array that is 1 year in length
OneYearGen = DateRange(datetime(2001, 1, 1), datetime(2001, 12, 31)) # 2001 is a non-leap year
OneYear = np.asarray([date for date in OneYearGen])

OneYearMonth = np.asarray([date.month for date in OneYear])
OneYearDay   = np.asarray([date.day for date in OneYear])

# Determine the path indices will be written to
OutPath = './Data/FD_Data/'

#%%
# cell 9
# Load and subset the land-sea mask data

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
###############################
### Christian et al. Method ###
###############################

# Calcualte flash droughts using an improved version of the FD identification method from Christian et al. 2019
# This method uses SESR to identify FD

# Initialize some variables
I, J, T = sesr['sesr'].shape
sesr_inter = np.ones((I, J, T)) * np.nan
sesr_filt  = np.ones((I, J, T)) * np.nan

sesr2d = sesr['sesr'].reshape(I*J, T, order = 'F')
sesr_inter2d = sesr_inter.reshape(I*J, T, order = 'F')
sesr_filt2d  = sesr_filt.reshape(I*J, T, order = 'F')

mask2d = maskSub.reshape(I*J, 1, order = 'F')

x = np.arange(-6.5, 6.5, (13/T)) # a variable covering the range of SESR with 1 entry for each time step

# Parameters for the filter
WinLength = 21 # Window length of 21 pentads
PolyOrder = 4

# Perform a basic linear interpolation for NaN values and apply a SG filter
print('Applying interpolation and Savitzky-Golay filter to SESR')
for ij in range(I*J):
    if mask2d[ij,0] == 0:
        continue
    else:
        pass
    
    # Perform a linear interpolation to remove NaNs
    ind = np.isfinite(sesr2d[ij,:])
    if np.nansum(ind) == 0:
        continue
    else:
        pass
    
    ind = np.where(ind == True)[0]
    interp_func = interpolate.interp1d(x[ind], sesr2d[ij,ind], kind = 'linear', fill_value = 'extrapolate')
    
    sesr_inter2d[ij,:] = interp_func(x)
    
    # Apply the Savitzky-Golay filter to the interpolated SESR data
    sesr_filt2d[ij,:] = signal.savgol_filter(sesr_inter2d[ij,:], WinLength, PolyOrder)
        
# Reorder SESR back to 3D data
sesr_filt = sesr_filt2d.reshape(I, J, T, order = 'F')

# Determine the change in SESR
print('Calculating the change in SESR')
DeltaSESR  = np.ones((I, J, T)) * np.nan
DeltaSESRz = np.ones((I, J, T)) * np.nan

DeltaSESR[:,:,1:] = sesr_filt[:,:,1:] - sesr_filt[:,:,:-1]

# Standardize the change in SESR
DeltaSESRMean, DeltaSESRstd = CalculateClimatology(DeltaSESR, pentad = True)

for n, date in enumerate(OneYear[::5]):
    ind = np.where( (date.month == sesr['month']) & (date.day == sesr['day']) )[0]
    
    for t in ind:
        DeltaSESRz[:,:,t] = (DeltaSESR[:,:,t] - DeltaSESRMean[:,:,n])/DeltaSESRstd[:,:,n]

# Begin the flash drought calculations
print('Identifying flash drought')
ChFD = np.ones((I, J, T)) * np.nan

ChFD2d  = ChFD.reshape(I*J, T, order = 'F')
dSESR2d = DeltaSESRz.reshape(I*J, T, order = 'F')

dsesrPercentile = 25
sesrPercentile  = 20

MinChange = timedelta(days = 30)
StartDate = sesr['ymd'][-1]

for ij in range(I*J):
    if mask2d[ij,0] == 0:
        continue
    else:
        pass
    
    StartDate = sesr['ymd'][-1]
    for t in range(T):
        ind = np.where( (sesr['ymd'][t].month == sesr['month']) & (sesr['ymd'][t].day == sesr['day']) )[0]
        
        ri_crit = np.nanpercentile(dSESR2d[ij,ind], dsesrPercentile)
        dc_crit = np.nanpercentile(sesr_filt2d[ij,ind], sesrPercentile)
        
        if ( (sesr['ymd'][t] - StartDate) >= MinChange ) & (sesr_filt2d[ij,t] <= dc_crit):
            ChFD2d[ij,t] = 1
        else:
            ChFD2d[ij,t] = 0
        
        if (dSESR2d[ij,t] <= ri_crit) & (StartDate == sesr['ymd'][-1]):
            StartDate = sesr['ymd'][t]
        elif (dSESR2d[ij,t] <= ri_crit) & (StartDate != sesr['ymd'][-1]):
            pass
        else:
            StartDate = sesr['ymd'][-1]
        
ChFD = ChFD2d.reshape(I, J, T, order = 'F')

# Write the data
print('Writing the data')

description = 'This file contains the flash drought identified for all pentads and CONUS grid points ' +\
                  'in the NARR dataset using the flash drought identification method in Christian et al. 2019. ' +\
                  'This method uses SESR as the variable for flash drought identification. ' +\
                  'Details on SESR method to identify flash drought can be found ' +\
                  'in Christian et al. 2019 (https://doi.org/10.1175/JHM-D-18-0198.1). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'chfd: Flash drought identified using the method in Christian et al. 2019. ' +\
                  'Data is either 0 (no flash drought) or 1 (flash drought identified). Data is on the pentad timescale. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(ChFD, sesr['lat'], sesr['lon'], sesr['date'], filename = 'ChristianFD.NARR.CONUS.pentad.nc', 
        VarSName = 'chfd', description = description, path = OutPath)


#%%
# cell 11
# Calculate and plot the climatology the Christian et al. flash drought to ensure the identification is correct


#### Calcualte the climatology ###

# Initialize variables
I, J, T = ChFD.shape
years  = np.unique(sesr['year'])

AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == sesr['year']) & ((sesr['month'] >= 4) & (sesr['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of flash drought for each year    
    AnnFD[:,:,y] = np.nanmean(ChFD[:,:,yInd], axis = -1)
    
    # Turn nonzero values to 1 (each year gets 1 count to the total)    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    

# Calculate the percentage number of years with rapid intensifications and flash droughts
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size

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
ax.set_title('Percent of Years from 1979 - 2019 with Christian et al. Flash Drought', size = 18)

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
cs = ax.pcolormesh(sesr['lon'], sesr['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
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
plt.show(block = False)



#%%
# cell 12
#############################
### Noguera et al. Method ###
#############################

# Calcualte flash droughts using a FD identification method from Noguera et al. 2020
# This method uses SPEI to identify FD

# Determine the change in SPEI across a 1 month (30 day = 6 pentad) period
print('Calculating the change in SPEI')

I, J, T = spei['spei'].shape
DeltaSPEI = np.ones((I, J, T)) * np.nan

DeltaSPEI[:,:,6:] = spei['spei'][:,:,6:] - spei['spei'][:,:,:-6] # Set the indices so that each entry in DeltaSPEI corrsponds to the end date of the difference

# MeanDeltaSPEI = np.ones((I, J, T)) * np.nan
# for t in range(5, T):
#     MeanDeltaSPEI[:,:,t] = np.nanmean(DeltaSPEI[:,:,t-5:t+1], axis = -1)

# Reorder data into 2D arrays for fewer embedded loops
spei2d = spei['spei'].reshape(I*J, T, order = 'F')
DeltaSPEI2d = DeltaSPEI.reshape(I*J, T, order = 'F')

# Calculate the occurrence of flash drought
print('Identifying flash drought')
NogFD = np.ones((I, J, T)) * np.nan

NogFD2d = NogFD.reshape(I*J, T, order = 'F')

ChangeCriterion = -2
DroughtCriterion = -1.28

MinChange = timedelta(days = 30)
StartDate = spei['ymd'][-1]

for ij in range(I*J):
    if mask2d[ij,0] == 0:
        continue
    else:
        pass
    
    #print(np.nanpercentile(spei2d[ij,:], 20))
    #print(np.where(DeltaSPEI2d[ij,:] < -2)[0])
    StartDate = spei['ymd'][-1]
    for t in range(T-1):
        ind = np.where( (spei['ymd'][t].month == spei['month']) & (spei['ymd'][t].day == spei['day']) )[0]
        
        ChangePercent = stats.percentileofscore(DeltaSPEI2d[ij,ind], DeltaSPEI2d[ij,t])
        DroughtPercent = stats.percentileofscore(spei2d[ij,ind], spei2d[ij,t])
        
        if (ChangePercent < 2) & (DroughtPercent < 10):
            NogFD2d[ij,t] = 1
        else:
            NogFD2d[ij,t] = 0
        
        # if (DeltaSPEI2d[ij,t] <= ChangeCriterion) & (spei2d[ij,t] <= DroughtCriterion): # Note, since the changes are calculated over a 1 month period, the first criterion in Noguera et al. is automatically satisified
        #     NogFD2d[ij,t] = 1
        # else:
        #     NogFD2d[ij,t] = 0
            
        # if ( (spei['ymd'][t] - StartDate) >= MinChange ) & (spei2d[ij,t] <= DroughtCriterion): # Note, since the changes are calculated over a 1 month period, the first criterion in Noguera et al. is automatically satisified
        #     NogFD2d[ij,t] = 1
        # else:
        #     NogFD2d[ij,t] = 0
            
        # if (DeltaSPEI2d[ij,t] <= ChangeCriterion) & (StartDate == spei['ymd'][-1]):
        #     StartDate = spei['ymd'][t]
        # elif (DeltaSPEI2d[ij,t] <= ChangeCriterion) & (StartDate != spei['ymd'][-1]):
        #     pass
        # else:
        #     StartDate = spei['ymd'][-1]
            
NogFD = NogFD2d.reshape(I, J, T, order = 'F')

# Write the data
print('Writing the data')

description = 'This file contains the flash drought identified for all pentads and CONUS grid points ' +\
                  'in the NARR dataset using the flash drought identification method in Noguera et al. 2020. ' +\
                  'This method uses SPEI as the variable for flash drought identification. ' +\
                  'Details on SPEI method to identify flash drought can be found ' +\
                  'in Noguera et al. 2020 (https://doi.org/10.1111/nyas.14365). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'nogfd: Flash drought identified using the method in Noguera et al. 2020. ' +\
                  'Data is either 0 (no flash drought) or 1 (flash drought identified). Data is on the pentad timescale. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(NogFD, spei['lat'], spei['lon'], spei['date'], filename = 'NogueraFD.NARR.CONUS.pentad.nc', 
        VarSName = 'nogfd', description = description, path = OutPath)



#%%
# cell 13
# Calculate and plot the climatology the Noguera et al. flash drought to ensure the identification is correct


#### Calcualte the climatology ###

# Initialize variables
I, J, T = NogFD.shape
years  = np.unique(spei['year'])

AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == spei['year']) & ((spei['month'] >= 4) & (spei['month'] <= 10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of rapid intensification and flash drought for each year    
    AnnFD[:,:,y] = np.nanmean(NogFD[:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    

# Calculate the percentage number of years with rapid intensifications and flash droughts
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size

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
ax.set_title('Percent of Years from 1979 - 2019 with Noguera et al. Flash Drought', size = 18)

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
cs = ax.pcolormesh(spei['lon'], spei['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
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
plt.show(block = False)


#%%
# cell 14
#################################
### Pendergrass et al. Method ###
#################################

# Calcualte flash droughts using a FD identification method from Pendergrass et al. 2020
# This method uses EDDI to identify FD

# Initialize some variables
I, J, T = eddi['eddi'].shape

PeFD = np.ones((I, J, T)) * np.nan

eddi2d = eddi['eddi'].reshape(I*J, T, order = 'F')
PeFD2d = PeFD.reshape(I*J, T, order = 'F')

print('Identifying flash drought')
for ij in range(I*J):
    if mask2d[ij,0] == 0:
        continue
    else:
        pass
    
    for t in range(3, T-3): # The criteria are EDDI must be 50% greater than EDDI 2 weeks (3 pentads) ago, or a 50 percentile increase in 2 weeks, and remain that intense for another 2 weeks.
        
        ind = np.where( (eddi['ymd'][t].month == eddi['month']) & (eddi['ymd'][t].day == eddi['day']) )[0]
        
        CurrentPercent = stats.percentileofscore(eddi2d[ij,ind], eddi2d[ij,t])
        PreviousPercent = stats.percentileofscore(eddi2d[ij,ind], eddi2d[ij,t-3])
        
        if ( (CurrentPercent - PreviousPercent) > 50 ) & (eddi2d[ij,t+1] >= eddi2d[ij,t]) & (eddi2d[ij,t+2] >= eddi2d[ij,t]) & (eddi2d[ij,t+3] >= eddi2d[ij,t]): # Note this checks for all pentads in the + 2 week period, so there cannot be moderation
            PeFD2d[ij,t] = 1
        else:
            PeFD2d[ij,t] = 0
            
PeFD = PeFD2d.reshape(I, J, T, order = 'F')

# Write the data
print('Writing the data')

description = 'This file contains the flash drought identified for all pentads and CONUS grid points ' +\
                  'in the NARR dataset using the flash drought identification method in Pendergrass et al. 2020. ' +\
                  'This method uses EDDI as the variable for flash drought identification. ' +\
                  'Details on EDDI method to identify flash drought can be found ' +\
                  'in Pendergrass et al. 2020 (https://doi.org/10.1038/s41558-020-0709-0). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'pegfd: Flash drought identified using the method in Pendergrass et al. 2020. ' +\
                  'Data is either 0 (no flash drought) or 1 (flash drought identified). Data is on the pentad timescale. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(PeFD, eddi['lat'], eddi['lon'], eddi['date'], filename = 'PendergrassFD.NARR.CONUS.pentad.nc', 
        VarSName = 'pegfd', description = description, path = OutPath)
        

#%%
# cell 15
# Calculate and plot the climatology the Pendergrass et al. flash drought to ensure the identification is correct


#### Calcualte the climatology ###

# Initialize variables
I, J, T = PeFD.shape
years  = np.unique(eddi['year'])

AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == eddi['year']) & ((eddi['month'] >= 4) & (eddi['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of rapid intensification and flash drought for each year    
    AnnFD[:,:,y] = np.nanmean(PeFD[:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    

# Calculate the percentage number of years with rapid intensifications and flash droughts
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size

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
ax.set_title('Percent of Years from 1979 - 2019 with Pendergrass et al. Flash Drought', size = 18)

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
cs = ax.pcolormesh(eddi['lon'], eddi['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
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
plt.show(block = False)
#%%
# cell 16
########################
### Li et al. Method ###
########################

# Uses SEDI




#%%
# cell 18
#########################
### Liu et al. Method ###
#########################

# Calcualte flash droughts using a FD identification method from Liu et al. 2020
# This method uses soil moisture to identify FD

# First, determine the soil moisture percentiles
print('Calculating soil moisture percentiles')
I, J, T = SM['soilm'].shape
SMPer = np.ones((I, J, T)) * np.nan

SM2d    = SM['soilm'].reshape(I*J, T, order = 'F')
SMPer2d = SMPer.reshape(I*J, T, order = 'F')

for t in range(T):
    ind = np.where( (SM['ymd'][t].day == SM['day']) & (SM['ymd'][t].month == SM['month']) )[0]
    
    for ij in range(I*J):
        SMPer2d[ij,t] = stats.percentileofscore(SM2d[ij,ind], SM2d[ij,t])
        
        
# Begin drought identification process
print('Identifying flash droughts')
LiuFD = np.ones((I, J, T)) * np.nan

LiuFD2d = LiuFD.reshape(I*J, T, order = 'F')

# Initialize up a variable to look up to 12 pentads ahead (from Otkin et al. 2021, that rapid intensification goes up to 10 pentads ahead); 12 ensures data after intensification is included
FutPentads = np.arange(0, 13)
FP = len(FutPentads)

for ij in range(I*J):
    if mask2d[ij,0] == 0:
        continue
    else:
        pass
    
    for t in range(T-12): # Exclude the last few months in the dataset for simplicity since FD identification involves looking up to 12 pentads ahead
        # First determine if the soil moisture is below the 40 percentile
        if SMPer2d[ij,t] <= 40:
            
            R2 = np.ones((FP)) * np.nan
            RIentries = np.ones((FP)) * np.nan
            
            # To determine when the percentiles level out (when the intensification ends), regress SM percentiles with pentads with increasing polynomial degrees until R^2 > 0.95 or until a 10th order polynomial is used (assumed accuracy is being lost here)
            for p in range(1, 11):
                SMest, R2p = PolynomialRegress(FutPentads, SMPer2d[ij,t:t+FP], order = p)
                
                R2[p-1] = R2
                if (R2 >= 0.95):
                    order = p
                    break
                elif (p >= 10):
                    # Find the maximum R2
                    ind = np.where(R2 == np.nanmax(R2))[0]
                    order = ind[0]+1
                    
                    # Get the SM estimates for the polynomial regression with the highest R2
                    SMest, R2p = PolynomialRegress(FutPentads, SMPer2d[ij,t:t+FP], order = order)
                    break
                else:
                    pass
                
                # Next, determine where the change in SMest is approximately 0 (within 0.01) to find when the rapid intensification ends
                for pent in FutPentads[1:]:
                    RIentries[pent-1] = (SMPer2d[ij,t+pent] - SMPer2d[ij,t])/pent # pent here is the difference between the current pentad and how many pentads ahead one is looking
                    
                    if (SMest[pent] - SMest[pent-1]) < 0.1:
                        RIend = pent
                        break
                    else:
                        pass
                    
                RImean = np.nanmean(RIentries)
                RImax  = np.nanmax(RIentries)
                
                # Lastly, to identify FD, two criteria are required. At the peak of the drought (this is approximately when Delta SMPercentiles = 0 since there is no more intensification), SMPercentiles < 20,
                # and the Rapid Intensification component must be: RImean >= 6.5 percentiles/week (about 5 percentiles/pentad) or RImax >= 10 percentiles/week (about 7.5 percentiles/pentad)
                
                # Note also that the FD is being identified for the end of RI period
                if (SMPer2d[ij,t+RIend] <= 20) & ( (RImean >= 5) | (RImax >= 7.5) ):
                    LiuFD2d[ij,t+RIend] = 1
                else:
                    LiuFD2d[ij,t+RIend] = 0
        else:
            continue


LiuFD = LiuFD2d.reshape(I, J, T, order = 'F')

# Write the data
print('Writing the data')

description = 'This file contains the flash drought identified for all pentads and CONUS grid points ' +\
                  'in the NARR dataset using the flash drought identification method in Liu et al. 2020. ' +\
                  'This method uses soil moisture as the variable for flash drought identification. ' +\
                  'Details on soil moisture method to identify flash drought can be found ' +\
                  'in Liu et al. 2020 (https://doi.org/10.1175/JHM-D-19-0088.1). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'liufd: Flash drought identified using the method in Liu et al. 2020. ' +\
                  'Data is either 0 (no flash drought) or 1 (flash drought identified). Data is on the pentad timescale. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(LiuFD, SM['lat'], SM['lon'], SM['date'], filename = 'LiuFD.NARR.CONUS.pentad.nc', 
        VarSName = 'liufd', description = description, path = OutPath)
        

#%%
# cell 19
# Calculate and plot the climatology the Pendergrass et al. flash drought to ensure the identification is correct


#### Calcualte the climatology ###

# Initialize variables
I, J, T = LiuFD.shape
years  = np.unique(SM['year'])

AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == SM['year']) & ((SM['month'] >= 4) & (SM['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of rapid intensification and flash drought for each year    
    AnnFD[:,:,y] = np.nanmean(LiuFD[:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    

# Calculate the percentage number of years with rapid intensifications and flash droughts
PerAnnFD = np.nansum(AnnFD[:,:,:], axis = -1)/years.size

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
ax.set_title('Percent of Years from 1979 - 2019 with Liu et al. Flash Drought', size = 18)

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
cs = ax.pcolormesh(SM['lon'], SM['lat'], PerAnnFD*100, vmin = cmin, vmax = cmax,
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
plt.show(block = False)


#%%
# cell 20
###########################
### Otkin et al. Method ###
###########################














