#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:27:37 2021

@author: stuartedris

This script is designed to take the processed data created in the Raw_Data_Processing
script create various indices designed to examine flash drought. The indices calculated
here include SESR, EDDI, FDII, SPEI, SAPEI, SEDI, and RI. The indices will written
to new files in the ./Data/Indices/ directory for future use.

This script assumes it is being running in the 'ML_and_FD_in_NARR' directory

Flash drought indices emitted:
- ESI: In Anderson et al. 2013 (https://doi.org/10.1175/2010JCLI3812.1) "Standardized
       anomalies in ET and f_PET [= ET/PET] will be referred to as the evapotranspiration
       index (ETI) and ESI, respectively." (Section 2. a. 1), paragraph 3) This means the 
       ESI is identical to SESR and is thus omitted from calculations here.
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
from scipy import stats
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta


#%%
# cell 2
# Create a function to import the nc files

def LoadNC(SName, filename, sm = False, path = './Data/Processed_Data/'):
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
        if sm is True:
            X[str(SName)] = nc.variables[str(SName)][:,:,:]
            X['level']    = nc.variables['level']
        else:
            X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X


#%%
# cell 3
# Create a function to write a variable to a .nc file
  
def WriteNC(var, lat, lon, dates, filename = 'tmp.nc', VarSName = 'tmp', description = 'Description', path = './Data/Indices/'):
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
  
def CalculateClimatology(var, week = True):
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
    I, J, T = var.shape
    
    # Count the number of years
    if week is True:
        yearLen = int(365/7)
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
    ClimMean = np.ones((I, J, yearLen)) * np.nan
    ClimStd  = np.ones((I, J, yearLen)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, yearLen+1):
        ind = np.where(i == day)[0]
        ClimMean[:,:,i-1] = np.nanmean(var[:,:,ind], axis = -1)
        ClimStd[:,:,i-1]  = np.nanstd(var[:,:,ind], axis = -1)
    
    return ClimMean, ClimStd


#%%
# cell 5
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
# cell 6
# Load all the data files

path = './Data/Processed_Data/'

T   = LoadNC('temp', 'temperature_2m.NARR.CONUS.weekly.nc', sm = False, path = path)
ET  = LoadNC('evap', 'evaporation.NARR.CONUS.weekly.nc', sm = False, path = path)
PET = LoadNC('pevap', 'potential_evaporation.NARR.CONUS.weekly.nc', sm = False, path = path)
P   = LoadNC('precip', 'accumulated_precipitation.NARR.CONUS.weekly.nc', sm = False, path = path)
SM  = LoadNC('soilm', 'soil_moisture.NARR.CONUS.weekly.nc', sm = True, path = path)


# In addition, calculate a datetime array that is 1 year in length
OneYearGen = DateRange(datetime(2001, 1, 1), datetime(2001, 12, 31)) # 2001 is a non-leap year
OneYear = np.asarray([date for date in OneYearGen])

OneYearMonth = np.asarray([date.month for date in OneYear])
OneYearDay   = np.asarray([date.day for date in OneYear])

# Determine the path indices will be written to
OutPath = './Data/Indices/'


#%%
# cell 7

######################
### Calculate SESR ###
######################

# Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
ESR = ET['evap']/PET['pevap']

# Determine the climatological mean and standard deviations of ESR
ESRMean, ESRstd = CalculateClimatology(ESR, week = True)

# Calculate SESR; it is the standardized ESR
I, J, T = ESR.shape

SESR = np.ones((I, J, T)) * np.nan

for n, date in enumerate(OneYear[::7]):
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0]
    
    for t in ind:
        SESR[:,:,t] = (ESR[:,:,t] - ESRMean[:,:,n])/ESRstd[:,:,n]
        
        
# Write the SESR data
description = 'This file contains the standardized evaporative stress ratio ' +\
                  '(SESR; unitless), calculated from evaporation and potential ' +\
                  'evaporation from the North American Regional Reanalysis ' +\
                  'dataset. Details on SESR and its calculations can be found ' +\
                  'in Christian et al. 2019 (https://doi.org/10.1175/JHM-D-18-0198.1). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'sesr: Weekly SESR (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SESR, ET['lat'], ET['lon'], ET['date'], filename = 'sesr.NARR.CONUS.weekly.nc', 
        VarSName = 'sesr', description = description, path = OutPath)


#%%
# cell 8
# Create a plot of SESR to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(ET['ymd'] == ExamineDate)[0]



# Lonitude and latitude tick information
lat_int = 10
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Colorbar information
cmin = -3; cmax = 3; cint = 0.5
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SESR for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

# Set borders
ax.coastlines()
ax.add_feature(cfeature.STATES, edgecolor = 'black')

# Set tick information
ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# Plot the data
cs = ax.contourf(ET['lon'], ET['lat'], SESR[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)


#%%
# cell 9

######################
### Calculate EDDI ###
######################


# Initialize the set of probabilities of getting a certain PET.
I, J, T = PET['pevap'].shape

prob = np.ones((I, J, T)) * np.nan
EDDI = np.ones((I, J, T)) * np.nan

N = np.unique(PET['year']) # Number of observations per time series

# Define the constants given in Hobbins et al. 2016
C0 = 2.515517
C1 = 0.802853
C2 = 0.010328

d1 = 1.432788
d2 = 0.189269
d3 = 0.001308

# Determine the probabilities of getting PET at time t.
for date in OneYear:
    ind = np.where( (PET['month'] == date.month) & (PET['day'] == date.day) )[0]
    
    # Collect the rank of the time series. Note in Hobbins et al. 2016, maximum PET is assigned rank 1 (so min PET has the highest rank)
    # This is opposite the order output by rankdata. (N+1) - rankdata puts the rank order to what is specificied in Hobbins et al. 2016.
    rank = (N+1) - stats.mstats.rankdata(PET['pevap'][:,:,ind], axis = -1)
    
    # Calculate the probabilities based on Tukey plotting in Hobbins et al. (Sec. 3a)
    prob[:,:,ind] = (rank - 0.33)/(N + 0.33)
    
    
# Reorder data to reduce number of embedded loops
prob2d = prob.reshape(I*J, T, order = 'F')
EDDI2d = EDDI.reshape(I*J, T, order = 'F')


# Calculate EDDI based on the inverse normal approximation given in Hobbins et al. 2016, Sec. 3a
EDDISign = 1
for ij in range(I*J):
    for t in range(T):
        if prob2d[ij,t] <= 0.5:
            prob2d[ij,t] = prob2d[ij,t]
            EDDISign = 1
        else:
            prob2d[ij,t] = 1 - prob2d[ij,t]
            EDDISign = -1
            
        W = np.sqrt(-2 * np.log(prob2d[ij,t]))
        
        EDDI2d[ij,t] = EDDISign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
        
# Reorder the data back to 3D
EDDI = EDDI2d.reshape(I, J, T, order = 'F')

# Write the EDDI data
description = 'This file contains the evaporative demand drought index ' +\
                  '(EDDI; unitless), calculated from potential ' +\
                  'evaporation from the North American Regional Reanalysis ' +\
                  'dataset. Details on EDDI and its calculations can be found ' +\
                  'in Hobbins et al. 2016 (https://journals.ametsoc.org/view/journals/hydr/17/6/jhm-d-15-0121_1.xml). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'eddi: Weekly EDDI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(EDDI, PET['lat'], PET['lon'], PET['date'], filename = 'eddi.NARR.CONUS.weekly.nc', 
        VarSName = 'eddi', description = description, path = OutPath)



#%%
# cell 10
# Create a plot of EDDI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(PET['ymd'] == ExamineDate)[0]



# Lonitude and latitude tick information
lat_int = 10
lon_int = 10

lat_label = np.arange(-90, 90, lat_int)
lon_label = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Projection information
data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Colorbar information
cmin = -3; cmax = 3; cint = 0.5
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('EDDI for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

# Set borders
ax.coastlines()
ax.add_feature(cfeature.STATES, edgecolor = 'black')

# Set tick information
ax.set_xticks(lon_label, crs = ccrs.PlateCarree())
ax.set_yticks(lat_label, crs = ccrs.PlateCarree())
ax.set_xticklabels(lon_label, fontsize = 16)
ax.set_yticklabels(lat_label, fontsize = 16)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# Plot the data
cs = ax.contourf(PET['lon'], PET['lat'], EDDI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)




#%%
# cell 11

#######################
### Calculate SAPEI ###
#######################
















