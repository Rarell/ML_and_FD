#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:56:05 2021

@author: stuartedris

This script is designed to take a set of raw, 3 hour data from the NARR and
subset and average the data to reduce it to a managable size. Data will be
averaged or summed (depending on variable) to a weekly average/sum to 
correspond with time scales of flash drought indices, which will be calculated
in a later script. In addition, data will be subsetted to the United States
in order to focus on it, where the data is most valid and has the most
observations, and to reduce the size of the data.

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from fnmatch import fnmatch
from scipy import stats
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta

# Define function and library names
FunctionNames = ['os', 'sys', 'np', 'plt', 'mcolors', 'ccrs', 'cfeature', 'cticker', 'fnmatch', 'stats', 'Dataset', 'num2date', 'datetime', 'timedelta', 'LoadNC', 'WriteNC', 'SubsetData', 'DailyMean', 'DateRange', 'this', 'FunctionNames']


#%%
# cell 2
# Read some of the datasets to examine them

#print(Dataset('./Data/Raw/VSM/soilw.197901.nc', 'r'))

print(Dataset('./Data/Raw/Liquid_VSM/soill.197901.nc', 'r'))

print(Dataset('./Data/Raw/Baseflow_Runoff/bgrun.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Soil_Moisture_Content/soilm.1979.nc', 'r'))

print(Dataset('./Data/Raw/Evaporation_accumulation/evap.1979.nc', 'r'))

print(Dataset('./Data/Raw/Potential_Evaporation_accumulation/pevap.1979.nc', 'r'))

print(Dataset('./Data/Raw/Precipitation_accumulation/apcp.1979.nc', 'r'))

print(Dataset('./Data/Raw/Temperature_2m/air.2m.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Temperature_skin/air.sfc.1979.nc', 'r'))

#%%
# cell 3
# Create a function to import the nc files

def LoadNC(SName, filename, sm = False, path = './Data/Raw/'):
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
        time = nc.variables['time'][:]
        X['time'] = time

        # Collect the data itself
        if sm is True:
            X[str(SName)] = nc.variables[str(SName)][:,:,:,:]
            X['level']    = nc.variables['level'][:]
        else:
            X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X


#%%
# cell 4
# Create a function to write a variable to a .nc file
  
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', sm = False, level = 'tmp', VarSName = 'tmp', description = 'Description', path = './Data/Processed/'):
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
        
        if sm is True:
            nc.createVariable('level', str, ())
            nc.variables['level'] = str(level)
        else:
            pass
            
            
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
# cell 7
# A cell to calculate a daily mean.
def DailyMean(X, summation = False):
    '''
    This function is designed to take raw, 3-hourly NARR data and average/sum
    it to daily data.
    
    Inputs:
    - X: the 3 or 4D data that is being averaged/summed to a daily format.
    - summation: A boolean value indicating whether the data is compressed 
                 to a daily mean or daily accumulation.
    
    Outputs:
    - X_daily: The variable X in a daily mean/accumulation format.
    '''
    
    T, I, J = X.shape
    
    T = int(T/8) # The NARR data is 3 hourly, so the daily data will have a temporal size of T/8 (8 = 24/3)
    
    X_daily = np.ones((T, I, J)) * np.nan
    
    hour_stamp = 0
    for t in range(T):
        if summation == True:
            X_daily[t,:,:] = np.nansum(X[hour_stamp:hour_stamp+8,:,:], axis = 0)
        else:
            X_daily[t,:,:] = np.nanmean(X[hour_stamp:hour_stamp+8,:,:], axis = 0)
            
        hour_stamp = hour_stamp + 8
            
    return X_daily
    
#%%
# cell 8
# Load in a sample file to examine and test it.
path = './Data/Raw/Liquid_VSM/'
filename = 'soill.197901.nc'

sm = LoadNC(SName = 'soill', filename = filename, sm = True, path = path)
# Note the main variable, soill, has dimensions time x level x y ("lat") x x ("lon")

# Examine the data
print(sm['level'])
print(sm['time'])


# Turn incorrectly labeled positive longtitude values back to negative
for i in range(len(sm['lon'][:,0])):
    ind = np.where( sm['lon'][i,:] > 0 )[0]
    sm['lon'][i,ind] = -1*sm['lon'][i,ind]

#%%
# cell 9
# To examine the data and grid further, plot some of the data.

# Lonitude and latitude tick information
lat_int = 15
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)


# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Colorbar information
cmin = -0.5; cmax = 0.5; cint = 0.05
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()


cs = ax.contourf(sm['lon'], sm['lat'], sm['soill'][0,0,:,:], levels = clevs, cmap = cmap, 
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([np.nanmin(sm['lon']), np.nanmax(sm['lon']), np.nanmin(sm['lat']), np.nanmax(sm['lat'])], 
                crs = fig_proj)

plt.show(block = False)


#%%
# cell 10
# Process the data to a more managable size (computer normally freezes if working with 2+ variables at their current size)

# First clear any data that might be loaded. Large amounts of data are about to be processed, and anything else should
#   be cleared for more space and better performance
this = sys.modules[__name__]
for n in dir():
    if (n[0] != '_') & (n not in FunctionNames):
        delattr(this, n)
        

# Determine which variable is being processed.
# data = 'temp'
# data = 'evap'
# data = 'pevap'
# data = 'precip'
# data = 'runoff'
# data = 'soilmoist'
# data = 'soilmoist00'
# data = 'soilmoist10'
data = 'soilmoist40'

# Create a range of datetimes
print('Constructing dates')
date_gen = DateRange(datetime(1979, 1, 1), datetime(2020, 12, 31))
dates = np.asarray([date for date in date_gen])

years  = np.asarray([date.year for date in dates])
months = np.asarray([date.month for date in dates])
days   = np.asarray([date.day for date in dates])

NumYears  = len(np.unique(years))
NumMonths = len(np.unique(months))

YearMonths = np.array([datetime(year, month, 1) for year in np.unique(years) for month in np.unique(months)]) # Create a datetime that iterate 1 month per time step



# Determine the location and filenames of the variable
if data == 'temp':
    path = './Data/Raw/Temperature_2m/'
    filenames = ['tmp'] * NumYears
    indvid_fn = 'air.2m.'
    OutFile = 'temperature_2m.NARR.CONUS.pentad.nc'
    
    SName = 'air'
    SNameOut = 'temp'
    
    description = 'This file contains the 2m temperature (K) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and averaged to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'temp: Pentad average temperature (K) at 2 meters above ' +\
                  'ground level. Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for temp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for temp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in temp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
    
elif data == 'evap':
    path = './Data/Raw/Evaporation_accumulation/'
    filenames = ['tmp'] * NumYears
    indvid_fn = 'evap.'
    OutFile = 'evaporation.NARR.CONUS.pentad.nc'
    
    SName = 'evap'
    SNameOut = 'evap'
    
    description = 'This file contains the evaporation (kg m^-2) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and summed to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'evap: Pentad accumulation of evaporation (kg m^-2). ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for evap. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for evap. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in evap. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
    
elif data == 'pevap':
    path = './Data/Raw/Potential_Evaporation_accumulation/'
    filenames = ['tmp'] * NumYears
    indvid_fn = 'pevap.'
    OutFile = 'potential_evaporation.NARR.CONUS.pentad.nc'
    
    SName = 'pevap'
    SNameOut = 'pevap'
    
    description = 'This file contains the potential evaporation (kg m^-2) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and summed to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'pevap: Pentad accumulation of potential evaporation (kg m^-2). ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for pevap. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for pevap. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in pevap. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
    
elif data == 'precip':
    path = './Data/Raw/Precipitation_accumulation/'
    filenames = ['tmp'] * NumYears
    indvid_fn = 'apcp.'
    OutFile = 'accumulated_precipitation.NARR.CONUS.pentad.nc'
    
    SName = 'apcp'
    SNameOut = 'precip'
    
    description = 'This file contains the accumulated precipitation (kg m^-2) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and summed to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'apcp: Pentad accumulation of precipitaiton (kg m^-2). ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
                  
elif data == 'runoff':
    path = './Data/Raw/Baseflow_Runoff/'
    filenames = ['tmp'] * NumYears
    indvid_fn = 'bgrun.'
    OutFile = 'baseflow_runoff.NARR.CONUS.pentad.nc'
    
    SName = 'bgrun'
    SNameOut = 'ro'
    
    description = 'This file contains the accumulated baseflow runoff (kg m^-2) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and summed to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'ro: Pentad accumulation of runoff (kg m^-2). ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
                  
elif data == 'soilmoist00':
    path = './Data/Raw/Liquid_VSM/'
    filenames = ['tmp'] * NumYears * NumMonths # Note the soil moisture data is in monthly files, not yearly
    indvid_fn = 'soill.'
    OutFile = 'soil_moisture.00cm.NARR.CONUS.pentad.nc'
    
    SName = 'soill'
    SNameOut = 'soilm'
    level = '0 cm'
    
    depth = int(0)
    
    description = 'This file contains the 0 cm volumetric soil moisture (fraction) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and averaged to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'soilm: Pentad average volumetric soil moisture (fraction) between 1 - 40 cm depths. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in soilm. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.\n' +\
                  'level: A string giving the soil depth.'

elif data == 'soilmoist10':
    path = './Data/Raw/Liquid_VSM/'
    filenames = ['tmp'] * NumYears * NumMonths # Note the soil moisture data is in monthly files, not yearly
    indvid_fn = 'soill.'
    OutFile = 'soil_moisture.10cm.NARR.CONUS.pentad.nc'
    
    depth = int(1)
    
    SName = 'soill'
    SNameOut = 'soilm'
    level = '10 cm'
    
    description = 'This file contains the 10 cm volumetric soil moisture (fraction) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and averaged to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'soilm: Pentad average volumetric soil moisture (fraction) between 1 - 40 cm depths. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in soilm. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.\n' +\
                  'level: A string giving the soil depth.'
                  
elif data == 'soilmoist40':
    path = './Data/Raw/Liquid_VSM/'
    filenames = ['tmp'] * NumYears * NumMonths # Note the soil moisture data is in monthly files, not yearly
    indvid_fn = 'soill.'
    OutFile = 'soil_moisture.40cm.NARR.CONUS.pentad.nc'
    
    depth = int(2)
    
    SName = 'soill'
    SNameOut = 'soilm'
    level = '40 cm'
    
    description = 'This file contains the 40 cm volumetric soil moisture (fraction) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and averaged to ' +\
                  'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'soilm: Pentad average volumetric soil moisture (fraction) between 1 - 40 cm depths. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in soilm. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.\n' +\
                  'level: A string giving the soil depth.'

else: # The remaining scenario is data = soilmoist
    path = './Data/Raw/Liquid_VSM/'
    filenames = ['tmp'] * NumYears * NumMonths # Note the soil moisture data is in monthly files, not yearly
    indvid_fn = 'soill.'
    OutFile = 'soil_moisture.NARR.CONUS.pentad.nc'
    
    depth = np.arange(0, 3+1, 1)
    
    SName = 'soill'
    SNameOut = 'soilm'
    level = '0 - 40 cm'
    
    description = 'This file contains the 1 - 40 cm average volumetric soil moisture (fraction) data from the ' +\
                  'North American Regional Reanalysis dataset. The data is ' +\
                  'subsetted to focus on the contential U.S., and averaged to ' +\
                  'the weekly timescale. Data ranges form Jan. 1 1979 to ' +\
                  'Dec. 31 2020. Variables are:\n' +\
                  'soil: Weekly average volumetric soil moisture (fraction) between 1 - 40 cm depths. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for soilm. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in soilm. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.\n' +\
                  'level: A string giving the soil depth over which soilm was averaged.'




# Constuct the filenames
print('Constructing filenames')
for n in range(len(filenames)):
    if fnmatch(data, 'soilmoist*'): # Soil moisture data is monthly
        if YearMonths[n].month < 10:
            filenames[n] = indvid_fn + str(YearMonths[n].year) + '0' + str(YearMonths[n].month) + '.nc'
        else:
            filenames[n] = indvid_fn + str(YearMonths[n].year) + str(YearMonths[n].month) + '.nc'
    else: # All other files are yearly
        filenames[n] = indvid_fn + str(np.unique(years)[n]) + '.nc'
        




# Load a sample file to get dimesions
examp = LoadNC(SName = 'soill', filename = 'soill.197901.nc', sm = True, path = './Data/Raw/Liquid_VSM/')
T, l, I, J = examp['soill'].shape
T = dates.size

# Initialize the combined data
RawData = np.ones((T, I, J)) * np.nan




# Load all the data for the chosen variable and reduce it to the daily timescale.
print('Loading files')
t = 0
for fn in filenames:
    if fnmatch(data, 'soilmoist*'): # Soil moisture data files are monthly, and first 3 levels (0 - 40 cm) need to be averaged
        X = LoadNC(SName = SName, filename = fn, sm = True, path = path)
        if fnmatch(data, 'soilmoist?0'):
            DailyX = DailyMean(X[str(SName)][:,depth,:,:], summation = False)
        else:
            DailyX = DailyMean(np.nanmean(X[str(SName)][:,depth,:,:], axis = 1), summation = False)
        
        
    elif (data == 'evap') | (data == 'pevap') | (data == ' precip') | (data == 'runoff'):
        X = LoadNC(SName = SName, filename = fn, sm = False, path = path)
        DailyX = DailyMean(X[str(SName)][:,:,:], summation = True)
        
    else:
        X = LoadNC(SName = SName, filename = fn, sm = False, path = path)
        DailyX = DailyMean(X[str(SName)][:,:,:], summation = False)

    VarTime = DailyX.shape[0]
    RawData[t:t+VarTime,:,:] = DailyX[:,:,:]
    
    t = t + VarTime




# Delete leap days for simplicity
print('Removing leap days')
    
ind = np.where( (months == 2) & (days == 29) )[0]
RawData = np.delete(RawData, ind, axis = 0)
dates = np.delete(dates, ind, axis = 0)
T = dates.size




# Correct longitudes
print('Correcting longitudes')

for i in range(len(X['lon'][:,0])):
    ind = np.where( X['lon'][i,:] > 0 )[0]
    X['lon'][i,ind] = -1*X['lon'][i,ind]
    



    
# Average or sum data to a pentad timescale.
print('Reducing data to pentad timescale')
DataPentad = np.ones((int(T/5), I, J)) * np.nan

n = 0
for t in range(int(T/5)):
    if (data == 'evap') | (data == 'pevap') | (data == 'precip') | (data == 'runoff'):
        DataPentad[t,:,:] = np.nansum(RawData[n:n+5,:,:], axis = 0)
    else:
        DataPentad[t,:,:] = np.nanmean(RawData[n:n+5,:,:], axis = 0)
        
    n = n + 5




# Make the time variable last in the list
DataPentadT = np.ones((I, J, int(T/5))) * np.nan
for t in range(int(T/5)):
    DataPentadT[:,:,t] = DataPentad[t,:,:]




# Subset the data to focus on the U.S.
print('Subsetting data')
LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65

DataProcessed, LatSub, LonSub = SubsetData(DataPentadT, X['lat'], X['lon'], 
                                           LatMin = LatMin, LatMax = LatMax, 
                                           LonMin = LonMin, LonMax = LonMax)





# Write the data to a file.
print('Writing data')
OutPath = './Data/Processed_Data/'
if fnmatch(data, 'soilmoist*'):
    WriteNC(DataProcessed, LatSub, LonSub, dates[::5], filename = OutFile, sm = True, level = level,
            VarSName = SNameOut, description = description, path = OutPath)
else: 
    WriteNC(DataProcessed, LatSub, LonSub, dates[::5], filename = OutFile, sm = False, 
            VarSName = SNameOut, description = description, path = OutPath)

# Repeat this cell for all NARR variables collected.




#%%
# cell 11
# Plot some of the data to test it and check for errors.

# Lonitude and latitude tick information
lat_int = 15
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)


# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Colorbar information
# cmin = 260; cmax = 310; cint = 5.0 # Temperature values
# cmin = 0; cmax = 50; cint = 1 # Evaporation and potential evaporation values
# cmin = 0; cmax = 2; cint = 0.1 # Accumulated precipitation values
cmin = 0; cmax = 0.5; cint = 0.05 # Volumetric soil moisture values
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

ax.coastlines()

ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()


cs = ax.contourf(LonSub, LatSub, DataProcessed[:,:,-1], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

ax.set_extent([LonMin, LonMax, LatMin, LatMax], crs = fig_proj)

plt.show(block = False)