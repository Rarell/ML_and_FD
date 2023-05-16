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


TODO:
- Add other reanalysis model descriptions to compress_raw_data() and load_mask()
- Update test_map() to also display the world
- Update soil moisture calculations to include arbitrary layers between 0 and 40 cm
"""

#%%
##############################################

# Import libraries
import os, sys, warnings
import gc
import argparse
import pickle
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
# Read some of the datasets to examine them

#print(Dataset('./Data/Raw/VSM/soilw.197901.nc', 'r'))

#print(Dataset('./Data/Raw/Liquid_VSM/soill.197901.nc', 'r'))

#print(Dataset('./Data/Raw/Baseflow_Runoff/bgrun.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Soil_Moisture_Content/soilm.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Evaporation_accumulation/evap.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Potential_Evaporation_accumulation/pevap.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Precipitation_accumulation/apcp.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Temperature_2m/air.2m.1979.nc', 'r'))

#print(Dataset('./Data/Raw/Temperature_skin/air.sfc.1979.nc', 'r'))

#%%
##############################################
# Create a function to import the nc files

def load_raw_nc(SName, filename, sm = False, model = 'narr', path = './Data/Raw/', meta_path = './Data/Raw/'):
    '''
    Load a raw, unprocessed .nc files.
    
    Inputs:
    :param SName: The short name of the variable being loaded. I.e., the name used
                  to call the variable in the .nc file.
    :param filename: The name of the .nc file.
    :param sm: Boolean determining if soil moisture is being loaded (an extra variable and dimension, level,
               needs to be loaded).
    :param model: String of the model name the dataset comes from.
    :param path: The path from the current directory to the directory the .nc file is in.
    :param meta_path: The path to the metadata (lon and lat) of the raw data.
    
    Outputs:
    :param X: A dictionary containing all the data loaded from the .nc file. The 
              entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
              (space dimensions), 'time' contains the dates in a string variable
              (time dimension), and 'SName' contains the variable (time x lat x lon).
    '''
    
    # Initialize the directory to contain the data
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S'
    
    # Load lat and lon data
    lat = load2D_nc('lat_%s.nc'%model, 'lat', path = meta_path)
    lon = load2D_nc('lon_%s.nc'%model, 'lon', path = meta_path)
    
    with Dataset(path + filename, 'r') as nc:
        
        # Correct longitude so values?
        if model == 'narr':
            for n in range(len(lon[:,0])):
                ind = np.where(lon[n,:] > 0)[0]
                lon[n,ind] = -1*lon[n,ind]

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
##############################################
# Create a function to import the processed .nc files

def load_nc(SName, filename, sm = False, path = './Data/Processed_Data/'):
    '''
    Load a .nc file.
    
    Inputs:
    :param SName: The short name of the variable being loaded. I.e., the name used
                  to call the variable in the .nc file.
    :param filename: The name of the .nc file.
    :param sm: Boolean determining if soil moisture is being loaded (an extra variable and dimension, level,
               needs to be loaded).
    :param path: The path from the current directory to the directory the .nc file is in.
    :param narr: Boolean indicating whether NARR data is being loaded (longitude values have to be corrected if so)
    
    Outputs:
    :param X: A dictionary containing all the data loaded from the .nc file. The 
              entry 'lat' contains latitude (space dimensions), 'lon' contains longitude
             (space dimensions), 'date' contains the dates in a datetime variable
             (time dimension), 'month' 'day' are the numerical month
             and day value for the given time (time dimension), 'ymd' contains full
             datetime values, and 'SName' contains the variable (time x lat x lon).
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
        if sm is True:
            X[str(SName)] = nc.variables[str(SName)][:,:,:]
            X['level']    = nc.variables['level']
        else:
            X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X


#%%
##############################################
# Create a function to load 2D data

def load2D_nc(filename, sname, path = './Data/'):
    '''
    This function loads 2 dimensional .nc files (e.g., the lat or lon files/
    only spatial files). Function is simple as these files only contain the raw data.
    
    Inputs:
    :param filename: The filename of the .nc file to be loaded.
    :param SName: The short name of the variable in the .nc file (i.e., the name to
                  call when loading the data)
    :param path: The path from the present direction to the directory the file is in.
    
    Outputs:
    :param x: The main variable in the .nc file.
    '''
    
    with Dataset('%s/%s'%(path, filename), 'r') as nc:
        x = nc.variables[sname][:,:]
        
    return x

# Create a function to load in mask data
def load_mask(model):
    '''
    Load in the land-sea mask
    
    Inputs:
    :param model: Name of reanalysis model whose mask is being loaded
    
    Outputs:
    :param mask: The land-sea mask in a 2D grid
    '''
    
    # Determine model specific variables
    if model == 'narr':
        path = '/Users/stuartedris/Desktop/PhD_Research_ML_and_FD/ML_and_FD_in_NARR/Data/narr'
        filename = 'land.nc'
        sname = 'land'
        
        lat_fname = 'lat_narr.nc'
        lon_fname = 'lon_narr.nc'
        
        subset = True
        LatMin = 25
        LatMax = 50
        LonMin = -130
        LonMax = -65
        
    # Load in the data and lat and lon
    mask = load2D_nc(filename, sname, path = path)
    lat = load2D_nc(lat_fname, 'lat', path = path) # Dataset is lat x lon
    lon = load2D_nc(lon_fname, 'lon', path = path) # Dataset is lat x lon
    
    # Subset the mask?
    if subset == True:
        mask, _, _ = subset_data(mask, lat, lon, LatMin = LatMin, LatMax = LatMax,
                                                 LonMin = LonMin, LonMax = LonMax)
        
    return mask[0,:,:]


#%%
##############################################
# Create a function to write a variable to a .nc file
  
def write_nc(var, lat, lon, dates, filename = 'tmp.nc', sm = False, level = 'tmp', VarSName = 'tmp', description = 'Description', path = './Data/Processed/'):
    '''
    Write data, and additional information such as latitude and longitude and timestamps, to a .nc file.
    
    Inputs:
    :param var: The variable being written (time x lat x lon format).
    :param lat: The latitude data with the same spatial grid as var.
    :param lon: The longitude data with the same spatial grid as var.
    :param dates: The timestamp for each pentad in var in a %Y-%m-%d format, same time grid as var.
    :param filename: The filename of the .nc file being written.
    :param sm: A boolean value to determine if soil moisture is being written. If true, an additional variable containing
               the soil depth information is provided.
    :param VarName: The full name of the variable being written (for the nc description).
    :param VarSName: The short name of the variable being written. I.e., the name used
                     to call the variable in the .nc file.
    :param description: A string descriping the data.
    :param path: The path to the directory the data will be written in.

    '''
    
    # Determine the spatial and temporal lengths
    T, I, J = var.shape
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
            nc.variables['date'][n] = str(dates[n])
            
        # Create the main variable
        nc.createVariable(VarSName, var.dtype, ('time', 'x', 'y'))
        nc.variables[str(VarSName)][:,:,:] = var[:,:,:]
        
        # Add soil moisture depth?
        if sm is True:
            nc.createVariable('level', str, ())
            nc.variables['level'] = str(level)
        else:
            pass
            
            
#%%
##############################################
# Function to subset any dataset.
def subset_data(X, lat, lon, LatMin, LatMax, LonMin, LonMax):
    '''
    This function is designed to subset data for any gridded dataset, including
    the non-simple grid used in the NARR dataset, where the size of the subsetted
    data is unknown. Note this function only makes square subsets with a maximum 
    and minimum latitude/longitude.
    
    Inputs:
    :param X: The variable to be subsetted.
    :param lat: The gridded latitude data corresponding to X.
    :param lon: The gridded Longitude data corresponding to X.
    :param LatMax: The maximum latitude of the subsetted data.
    :param LatMin: The minimum latitude of the subsetted data.
    :param LonMax: The maximum longitude of the subsetted data.
    :param LonMin: The minimum longitude of the subsetted data.
    
    Outputs:
    :param XSub: The subsetted data.
    :param LatSub: Gridded, subsetted latitudes.
    :param LonSub: Gridded, subsetted longitudes.
    '''
    
    # Collect the original sizes of the data/lat/lon
    T, I, J = X.shape
    
    # Reshape the data into a 2D array and lat/lon to a 1D array for easier referencing.
    X2D   = X.reshape(T, I*J, order = 'F')
    Lat1D = lat.reshape(I*J, order = 'F')
    Lon1D = lon.reshape(I*J, order = 'F')
    
    # Find the indices in which to make the subset.
    LatInd = np.where( (Lat1D >= LatMin) & (Lat1D <= LatMax) )[0]
    LonInd = np.where( (Lon1D >= LonMin) & (Lon1D <= LonMax) )[0]
    
    # Find the points where the lat and lon subset overlap. This comprises is the subsetted grid.
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
    X2D[:,PlaceHolderInt] = np.nan
    
    # Collect the subset of the data, lat, and lon
    XSub = X2D[:,SubIndTotal]
    LatSub = Lat1D[SubIndTotal]
    LonSub = Lon1D[SubIndTotal]
    
    # Reorder the data back into a 3D array, and lat and lon into gridded 2D arrays
    XSub = XSub.reshape(T, Isub, Jsub, order = 'F')
    LatSub = LatSub.reshape(Isub, Jsub, order = 'F')
    LonSub = LonSub.reshape(Isub, Jsub, order = 'F')
    
    # Return the the subsetted data
    return XSub, LatSub, LonSub


#%%
##############################################
# Create a function to generate a range of datetimes
def date_range(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
    :param StartDate: A datetime. The starting date of the interval.
    :param EndDate: A datetime. The ending date of the interval.
        
    Outputs:
    - A generator of all dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 

#%%
##############################################
# A cell to calculate a daily mean.
def daily_compression(X, summation = False, N_per_day = 8):
    '''
    Take subdaily data and average/sum it to daily data.
    
    Inputs:
    :param X: The 3D or 4D data that is being averaged/summed to a daily format.
    :param summation: A boolean value indicating whether the data is compressed 
                      to a daily mean or daily accumulation.
    :param N_per_day: Number of entries in a given day
    
    Outputs:
    :param X_daily: The variable X in a daily mean/accumulation format.
    '''
    
    # Initialize values
    T, I, J = X.shape
    
    T = int(T/N_per_day)
    
    X_daily = np.ones((T, I, J)) * np.nan
    
    # Average/sum the data to the daily timescale
    hour_stamp = 0
    for t in range(T):
        # Sum the data?
        if summation == True:
            X_daily[t,:,:] = np.nansum(X[hour_stamp:hour_stamp+N_per_day,:,:], axis = 0)
        else:
            X_daily[t,:,:] = np.nanmean(X[hour_stamp:hour_stamp+N_per_day,:,:], axis = 0)
            
        hour_stamp = hour_stamp + N_per_day
            
    return X_daily
    

def pentad_compression(X, summation = False):
    '''
    Reduce daily data to the pentad timescale.
    
    Inputs:
    :param X: The 3D daily data that is being averaged/summed to a pentad format.
    :param summation: A boolean value indicating whether the data is compressed 
                      to a pentad mean or pentad accumulation.
    
    Outputs:
    :param X_pentad: The variable X in a pentad mean/accumulation format.
    '''
    
    # Initialize some variables
    T, I, J = X.shape
    
    X_pentad = np.ones((int(T/5), I, J)) * np.nan
    
    # Reduce the data to the pentad timescale
    n = 0
    for t in range(int(T/5)):
        if summation == True:
            X_pentad[t,:,:] = np.nansum(X[n:n+5,:,:], axis = 0)
        else:
            X_pentad[t,:,:] = np.nanmean(X[n:n+5,:,:], axis = 0)
            
        n = n + 5
        
    return X_pentad

#%%
##############################################
# Function to examine and test loaded datadata.

def test_map(data, lat, lon, dates, data_name):
    '''
    Create a simple plot of the data to examine and test it.
    
    Inputs:
    :param data: Data to be plotted.
    :param lat: Latitude grid of the data.
    :param lon: Longitude grid of the data.
    :param dates: Array of datetimes corresponding to the timestamps in data.
    :param data_name: Full name of the variable being processed.
    '''
    
    # Pick a random date to plot
    rand_int = np.random.randint(dates.size)
    
    # Lonitude and latitude tick information
    lat_int = 15
    lon_int = 10
    
    lat_label = np.arange(-90, 90, lat_int)
    lon_label = np.arange(-180, 180, lon_int)


    # Projection information
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Colorbar information
    #### NOTE: Might need to adjust cmin, cmax, and cint for other variables
    if (np.ceil(np.nanmin(data[rand_int,:,:])) == 1) & (np.floor(np.nanmax(data[rand_int,:,:])) == 0): # Special case if the variable varies from 0 to 1
        cmin = np.round(np.nanmin(data[rand_int,:,:]), 2); cmax = np.round(np.nanmax(data[rand_int,:,:]), 2); cint = (cmax - cmin)/100
    else:
        cmin = np.ceil(np.nanmin(data[rand_int,:,:])); cmax = np.floor(np.nanmax(data[rand_int,:,:])); cint = (cmax - cmin)/100
    
    clevs = np.arange(cmin, cmax+cint, cint)
    nlevs = len(clevs) - 1
    cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)
    
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()

    # Figure
    fig = plt.figure(figsize = [12, 16])
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

    ax.set_title('%s for %s'%(data_name, dates[rand_int].strftime('%Y-%m-%d')), fontsize = 16)    

    # Add coastlines
    ax.coastlines()

    # Set tick information
    ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
    ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
    ax.set_xticklabels(lon_label, fontsize = 16)
    ax.set_yticklabels(lat_label, fontsize = 16)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # Plot the data
    cs = ax.contourf(lon, lat, data[rand_int,:,:], levels = clevs, cmap = cmap, 
                     transform = data_proj, extend = 'both', zorder = 1)

    # Add a colorbar
    cbax = fig.add_axes([0.125, 0.35, 0.80, 0.02])
    cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

    ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], 
                    crs = fig_proj)
    
    plt.savefig('%s_test_map.png'%data_name)

    plt.show(block = False)



#%%
##############################################

# Process the data to a more managable size (computer normally freezes if working with 2+ variables at their current size)

        
def compress_raw_data(data_name, model, fname_base, raw_sname, start_date, end_date, N_per_day, path, plot):
    '''
    Process raw data: Load in raw data and average it to pentad format and perform any subsets if needed.
    
    Data is saved at the end 
    
    Inputs:
    :param data_name: Full name of the variable being processed
    :param model: Name of the reanalysis model the data is being loaded from
    :param fname_base: The base of the filenames of the variable to be compressed
    :param raw_sname: The short name of the variable in the raw .nc file
    :param start_date: Datetime for the first point in the dataset
    :param end_date: Datetime for the last point in the dataset
    :param N_per_day: Number of data entries in a given day
    :param path: Path to the data. Assumed it has a separate "Raw" and "Processed_Data" directories within it
    :param plot: Boolean indicating whether to make a test plot the processed data to examine it

    '''

    # Create a range of datetimes
    print('Constructing dates')
    date_gen = date_range(start_date, end_date)
    dates = np.asarray([date for date in date_gen])

    years  = np.asarray([date.year for date in dates])
    months = np.asarray([date.month for date in dates])
    days   = np.asarray([date.day for date in dates])
    
    # Determine information specific to the variable
    print('Collecting variable specific information')
    
    if data_name == 'temperature':
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Temperature_2m')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.pentad.nc'%(data_name, model)
        
        output_sname = 'temp'
        
        var_descrip = 'temp: Pentad average temperature (K) at 2 meters above ' +\
                      'ground level. Variable format is time x by y\n'
                      
        sm = False
        summation = False
    
    elif data_name == 'evaporation':
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Evaporation_accumulation')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.pentad.nc'%(data_name, model)
        
        output_sname = raw_sname
        
        var_descrip = 'evap: Pentad accumulation of evaporation (kg m^-2). ' +\
                      'Variable format is time by x by y\n'
                      
        sm = False
        summation = True
    
    elif data_name == 'potential_evaporation':
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Potential_Evaporation_accumulation')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.pentad.nc'%(data_name, model)
        
        output_sname = raw_sname
        
        var_descrip = 'pevap: Pentad accumulation of potential evaporation (kg m^-2). ' +\
                      'Variable format is time by x by y\n'
                      
        sm = False
        summation = True

    elif data_name == 'precipitation':
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Precipitation_accumulation')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.pentad.nc'%(data_name, model)
        
        output_sname = 'precip'
        
        var_descrip = 'precip: Pentad accumulation of precipitaiton (kg m^-2). ' +\
                      'Variable format is time by x by y\n' 
                      
        sm = False
        summation = True

    elif data_name == 'runoff':
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Baseflow_Runoff')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.pentad.nc'%(data_name, model)
        
        output_sname = 'ro'
        
        var_descrip = 'ro: Pentad accumulation of runoff (kg m^-2). ' +\
                      'Variable format is time by x by y\n'
                      
        sm = False
        summation = True

    elif data_name == 'soil_moisture':
        level = '0-40cm'
        
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Liquid_VSM')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.%s.pentad.nc'%(data_name, level, model)
        
        output_sname = 'soilm'
        
        var_descrip = 'soilm: Pentad average volumetric soil moisture (fraction) between 1 - 40 cm depths. ' +\
                      'Variable format is time by x by y\n' +\
                      'level: A string giving the soil depth.' 
                      
        depth = np.arange(0, 2+1, 1)
        sm = True
        summation = False

    elif data_name == 'soil_moisture00':
        level = '0cm'
        
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Liquid_VSM')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.%s.pentad.nc'%(data_name, level, model)
        
        output_sname = 'soilm'
        
        var_descrip = 'soilm: Pentad average volumetric soil moisture (fraction) for a 0 cm depth. ' +\
                      'Variable format is time by x by y\n' +\
                      'level: A string giving the soil depth.' 
                      
        depth = int(0)
        sm = True
        summation = False
    
    elif data_name == 'soil_moisture10':
        level = '10cm'
        
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Liquid_VSM')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.%s.pentad.nc'%(data_name, level, model)
        
        output_sname = 'soilm'
        
        var_descrip = 'soilm: Pentad average volumetric soil moisture (fraction) for a  10 cm depth. ' +\
                      'Variable format is time by x by y\n' +\
                      'level: A string giving the soil depth.' 
                      
        depth = int(1)
        sm = True
        summation = False
    
    elif data_name == 'soil_moisture40':
        level = '40cm'
        
        input_path = '%s/%s/%s/'%(path, 'Raw', 'Liquid_VSM')
        output_path = '%s/%s/'%(path, 'Processed_Data')
        output_filename = '%s.%s.%s.pentad.nc'%(data_name, level, model)
        
        output_sname = 'soilm'
        
        var_descrip = 'soilm: Pentad average volumetric soil moisture (fraction) for a 40 cm depth. ' +\
                      'Variable format is time by x by y\n' +\
                      'level: A string giving the soil depth.' 
                      
        depth = int(2)
        sm = True
        summation = False
    
    else:
        # Variable is known and not in one of the examined variables
        print('%s is not one of the variables examined for this experiment'%data_name)
        return
    
    
    # Check if output processed file already exists
    if os.path.exists('%s/%s'%(output_path, output_filename)):
            # Processed file does exist: exit
            print("File %s already exists"%output_filename)
            return
    
    
    # Determine the input filenames and main description of the output data based on the model 
    print('Constructing filenames')
    filenames = []
    if model == 'narr':
        for year in np.unique(years):
            if sm == True:
                # Soil moisture datafiles are monthly in the NARR
                for month in np.unique(months):
                    if month < 10:
                        mon_str = '0%s'%(month)
                    else:
                        mon_str = '%s'%(month)
                        
                    fn = '%s.%s%s.nc'%(fname_base, year, mon_str)
                    filenames.append(fn)
                    
            else: # NARR datafiles other than soil moisture are in yearly files
                fn = '%s.%s.nc'%(fname_base, year)
                filenames.append(fn)

        # NARR data is subsetted to the U.S.
        subset = True
        LatMin = 25
        LatMax = 50
        LonMin = -130
        LonMax = -65
                

        # Main description for the output .nc file
        main_description = 'This file contains the' + data_name + ' data from the ' +\
                           'North American Regional Reanalysis dataset. The data is ' +\
                           'subsetted to focus on the contential U.S., and averaged to ' +\
                           'the pentad timescale. Data ranges form Jan. 1 1979 to ' +\
                           'Dec. 31 2020. Variables are:\n'
                           
                           
    ####### NOTE: ADD other reanalysis models
    else:
        pass
        
    
    
    
    # Put together the output description
    description = main_description + var_descrip +\
                  'lat: 2D latitude corresponding to the grid for temp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for temp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in temp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'
       
                  
       
    
    # Load a sample file to get dimesions
    examp = load_raw_nc(SName = 'soill', filename = 'soill.197901.nc', sm = True, path = './Data/%s/Raw/Liquid_VSM/'%model, meta_path = path)
    T, l, I, J = examp['soill'].shape
    T = dates.size

    # Initialize the combined data
    raw_data = np.ones((T, I, J)) * np.nan

    print(T)
    # Load all the data for the chosen variable and reduce it to the daily timescale.
    print('Loading raw data')
    t = 0
    for fn in filenames:
        
        # Load the raw data
        x_raw = load_raw_nc(SName = raw_sname, filename = fn, sm = sm, model = model, path = input_path, meta_path = path)
            
        # Determine whether soil moisture at a specific depth or all depths are desired
        if fnmatch(data_name, 'soil_moisture?0'):
            x = x_raw[str(raw_sname)][:,depth,:,:]
        elif fnmatch(data_name, 'soil_moisture'):
            x = np.nanmean(x_raw[str(raw_sname)][:,depth,:,:], axis = 1)
        else:
            x = x_raw[str(raw_sname)][:,:,:]
            
        daily_x = daily_compression(x, summation = summation, N_per_day = N_per_day)

        var_T = daily_x.shape[0]
        print(fn)
        raw_data[t:t+var_T,:,:] = daily_x[:,:,:]
        
        t = t + var_T
        
        
        
    # Delete leap days for simplicity
    print('Removing leap days')
        
    ind = np.where( (months == 2) & (days == 29) )[0]
    raw_data = np.delete(raw_data, ind, axis = 0)
    dates = np.delete(dates, ind, axis = 0)
    T = dates.size


    # Average or sum data to a pentad timescale.
    print('Reducing data to pentad timescale')
    data_pentad = pentad_compression(raw_data, summation = summation)
    
    
    
    # Subset the data?
    if subset == True:
        print('Subsetting data')
        data_processed, lat, lon = subset_data(data_pentad, x_raw['lat'], x_raw['lon'], 
                                                       LatMin = LatMin, LatMax = LatMax, 
                                                       LonMin = LonMin, LonMax = LonMax)
        
    else:
        data_processed = data_pentad
        lat = x_raw['lat']; lon = x_raw['lon']


    # Write the data to a file.
    print('Writing data')
    write_nc(data_processed, lat, lon, dates[::5], filename = output_filename, sm = sm, 
             VarSName = output_sname, description = description, path = output_path)
    
    
    # Plot the data?
    if plot:
        test_map(data_processed, lat, lon, dates[::5], data_name)
    
    
    # Since some of these files are large, remove them from the namespace to ensure conserve memory
    del examp, x_raw, x, raw_data, daily_x, data_pentad, data_processed
    gc.collect() # Clears deleted variables from memory 


#%%
##############################################

# data parsing function(s)

def parse_data(data, dates, path, fname, years = None, months = None, days = None):
    '''
    Parse a list 3D time x lat x space datasets into time x space x folds datasets, with each
    fold being 1 growing season. The data is then saved to a cdf (.nc) file for later use.
    
    Inputs:
    :param data: List of 3D datasets in time x lat x lon formats. Must be a list.
    :param dates: Array of datetimes corresponding to the timestamps in data.
    :param path: Path leadind to where the  parsed dataset will be saved.
    :param fname: The filename the parsed data will be saved to
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    
    # Stack the list into an axis
    data_stacked = np.stack(data, axis = 0)
    Nf, T, I, J = data_stacked.shape
    
    data_stacked = data_stacked.reshape(Nf, T, I*J, order = 'F')
    
    Nyears = np.unique(years)
    
    data_parsed = []
    for ny in Nyears:
        ind = np.where( (months >= 4) & (months <= 10) & (years == ny))[0]
        
        data_parsed.append(data_stacked[:,ind,:])
        
    # Stack the folds into a new axis
    data_parsed = np.stack(data_parsed, axis = -1)
    
    # Write the data
    with open("%s/%s"%(path, fname), "wb") as fp:
        pickle.dump(data_parsed, fp)
    
    

#%%
##############################################

# argument parser
def create_data_process_parser():
    '''
    Create argument parser
    '''
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Data Processing', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    parser.add_argument('--variables', nargs='+', type=str, default=['temperature','precipitation'], help='Name of the variables being loaded and processed')
    parser.add_argument('--snames', nargs='+', type=str, default=['air','apcp'], help='Short names of the variable(s) used in the raw .nc files')
    parser.add_argument('--fname_bases', nargs='+', type=str, default=['air.2m.','apcp.'], help='Base name of the filenames for the variable(s)')
    
    parser.add_argument('--make_features', action='store_true', help='Make the features dataset and split it into folds for future ML')
    
    parser.add_argument('--dataset', type=str, default='/Users/stuartedris/desktop/PhD_Research_ML_and_FD/ML_and_FD_in_NARR/Data', help='Data set directory')
    parser.add_argument('--model', type=str, default='narr', help='Reanalysis model the dataset(s) came from')
    
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--plot', action='store_true', help="Make a test plot of the variable(s) being processed to examine it")
    
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--N_per_day', type=int, default=8, help="Number of entries in each day in the raw data files")
    
    parser.add_argument('--start_date', type=str, default='1990-01-01', help='Start date for the dataset in %Y-%m-%d format')
    parser.add_argument('--end_date', type=str, default='2020-12-31', help='End date for the dataset in %Y-%m-%d format')
    
    
    return parser


#%%
##############################################

# Main function
if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_data_process_parser()
    args = parser.parse_args()
    
    # Get the directory of the dataset
    dataset_dir = '%s/%s'%(args.dataset, args.model)
    
    # Turn the start and end dates into datetimes
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Process each of the variables specified, if they have not been done
    for var, fbase, sname in zip(args.variables, args.fname_bases, args.snames):
        print('Processing %s'%var)
        
        # data_name, model, fname_base, raw_sname, start_date, end_date, N_per_day, path
        compress_raw_data(var, args.model, fbase, sname, start_date, end_date, args.N_per_day, dataset_dir, args.plot)
        
    # Create a set of ML input data?
    if args.make_features:
        print('Creating ML input data')
        
        feature_fname = 'fd_input_features.pkl'
        
        # Check if output processed file already exists
        if os.path.exists('%s/%s'%(dataset_dir, feature_fname)):
            # Processed file does exist: exit
            print("File %s already exists"%feature_fname)
        
        else:
            # Load the data that will make up the input data
            temp = load_nc('temp', 'temperature.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            evap = load_nc('evap', 'evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            pevap = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            precip = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            soilm = load_nc('soilm', 'soil_moisture.0-40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
            
            # Initialize the change in ET/PET/SM variables
            T, I, J = evap['evap'].shape
            
            del_et = np.ones((T, I, J)) * np.nan
            del_pet = np.ones((T, I, J)) * np.nan
            del_sm = np.ones((T, I, J)) * np.nan
            
            # Perform a running mean of 1 month (6 pentads) long so the changes simulate something like a 1 month change
            runmean = 6
            
            for i in range(I):
                for j in range(J):
                    et_mean = np.convolve(evap['evap'][:,i,j], np.ones((runmean,))/runmean)[(runmean-1):]
                    pet_mean = np.convolve(pevap['pevap'][:,i,j], np.ones((runmean,))/runmean)[(runmean-1):]
                    sm_mean = np.convolve(soilm['soilm'][:,i,j], np.ones((runmean,))/runmean)[(runmean-1):]

                    # Calculate the change
                    del_et[:-1,i,j] = et_mean[1:] - et_mean[:-1]
                    del_pet[:-1,i,j] = pet_mean[1:] - pet_mean[:-1]
                    del_sm[:-1,i,j] = sm_mean[1:] - sm_mean[:-1]
            
            # Parse and save the input data into a pickle file
            parse_data([temp['temp'], evap['evap'], del_et, pevap['pevap'], del_pet, precip['precip'], soilm['soilm'], del_sm], 
                       temp['ymd'], dataset_dir, feature_fname)
        
    
    

                  