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



Full citations for the referenced papers can be found at:
- Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1
- Hobbins et al. 2016 (for EDDI): https://doi.org/10.1175/JHM-D-15-0121.1
- Li et al. 2020a (for SEDI): https://doi.org/10.1016/j.catena.2020.104763
- Li et al. 2020b (for SAPEI): https://doi.org/10.1175/JHM-D-19-0298.1
- Hunt et al. 2009 (for SMI): https://doi.org/10.1002/joc.1749
- Otkin et al. 2021 (for FDII): https://doi.org/10.3390/atmos12060741
- Vicente-Serrano et al. 2010 (for SPEI): https://doi.org/10.1175/2009JCLI2909.1
- Anderson et al. 2013 (for ESI): https://doi.org/10.1175/2010JCLI3812.1
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
from scipy.special import gamma
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

T    = LoadNC('temp', 'temperature_2m.NARR.CONUS.pentad.nc', sm = False, path = path)
ET   = LoadNC('evap', 'evaporation.NARR.CONUS.pentad.nc', sm = False, path = path)
PET  = LoadNC('pevap', 'potential_evaporation.NARR.CONUS.pentad.nc', sm = False, path = path)
P    = LoadNC('precip', 'accumulated_precipitation.NARR.CONUS.pentad.nc', sm = False, path = path)
SM   = LoadNC('soilm', 'soil_moisture.NARR.CONUS.pentad.nc', sm = True, path = path)
SM00 = LoadNC('soilm', 'soil_moisture.00cm.NARR.CONUS.pentad.nc', sm = True, path = path)
SM10 = LoadNC('soilm', 'soil_moisture.10cm.NARR.CONUS.pentad.nc', sm = True, path = path)
SM40 = LoadNC('soilm', 'soil_moisture.40cm.NARR.CONUS.pentad.nc', sm = True, path = path)



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

# Details in SESR can be found in the Christian et al. 2019 paper.

# Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
ESR = ET['evap']/PET['pevap']

# Determine the climatological mean and standard deviations of ESR
ESRMean, ESRstd = CalculateClimatology(ESR, pentad = True)

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
                  'sesr: Pentad SESR (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SESR, ET['lat'], ET['lon'], ET['date'], filename = 'sesr.NARR.CONUS.pentad.nc', 
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

# Details are found in the Hobbins et al. 2016 paper.

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
                  'eddi: Pentad EDDI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(EDDI, PET['lat'], PET['lon'], PET['date'], filename = 'eddi.NARR.CONUS.pentad.nc', 
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

######################
### Calculate SEDI ###
######################

# Details on SEDI can be found in the Li et al. 2020a paper.

# Where SESR is the standardized ratio of ET to PET, SEDI is the standardardized difference of ET to PET
ED = ET['evap'] - PET['pevap']

EDMean, EDstd = CalculateClimatology(ED, pentad = True)

# Calculate SEDI; it is the standardized ED
I, J, T = ED.shape

SEDI = np.ones((I, J, T)) * np.nan

for n, date in enumerate(OneYear[::7]):
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0]
    
    for t in ind:
        SEDI[:,:,t] = (ED[:,:,t] - EDMean[:,:,n])/EDstd[:,:,n]
        
        
# Write the SESR data
description = 'This file contains the standardized evapotranspiration deficit index ' +\
                  '(SEDI; unitless), calculated from evaporation and potential ' +\
                  'evaporation from the North American Regional Reanalysis ' +\
                  'dataset. Details on SEDI and its calculations can be found ' +\
                  'in Li et al. 2020 (https://doi.org/10.1016/j.catena.2020.104763). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'sedi: Pentad SEDI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SEDI, ET['lat'], ET['lon'], ET['date'], filename = 'sedi.NARR.CONUS.pentad.nc', 
        VarSName = 'sedi', description = description, path = OutPath)

#%%
# cell 12
# Create a plot of SEDI to check calculations

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
ax.set_title('SEDI for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(ET['lon'], ET['lat'], SEDI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)




#%%
# cell 13

#######################
### Calculate SAPEI ###
#######################

# Details for SAPEI can be found in the Li et al. 2020b paper.

a = 0.903 # Note this decay rate is defined by keeping the total decay (13%) after 100 days or 20 pentads.
          # These values may be adjusted, as SAPEI with this decay/memory is like unto a 3-month SPEI
          # (see Li et al. 2020b sections 3a and 4a).

# Initialize the moisture deficit D
I, J, T = P['precip'].size
D = np.zeros((I, J, T))

NDays = 100 # Number of days in the decay/memory
counters = np.arange(1, (NDays/5)+1)

for t in range(T):
    for i in counters:
        if i > t:
            break
        
        moistDeficit = (a**i) * (P['precip'][:,:,t-i] - PET['pevap'][:,:,t-i])
        
        D[:,:,t] = D[:,:,t] + moistDeficit

# From here, continue to perform the transformation of D from a log-logistic distribution to normal as detailed in Vicente-Serrano et al. 2010
N = np.unique(P['year']) # Number of observations per time series

frequencies = np.ones((I, J, T)) * np.nan
PWM0 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 0
PWM1 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 1
PWM2 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 2

# Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
for t, date in enumerate(OneYear):
    ind = np.where( (P['month'] == date.month) & (P['day'] == date.day) )[0]
    
    # Get the frequency estimator
    frequencies[:,:,ind] = (stats.mstats.rankdata(D[:,:,ind]) - 0.35)/N
    
    # Get the moments
    PWM0[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**0)*D[:,:,ind], axis = -1)/N
    PWM1[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**1)*D[:,:,ind], axis = -1)/N
    PWM2[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**2)*D[:,:,ind], axis = -1)/N



# Calculate the parameters of log-logistic distribution, using the equations in the Vicente-Serrano et al. 2010 paper
alpha = np.ones((I, J, OneYear.size)) * np.nan # Scale parameter
beta  = np.ones((I, J, OneYear.size)) * np.nan # Shape parameter
gamm  = np.ones((I, J, OneYear.size)) * np.nan # Origin parameter; not gamm refers to the gamma parameter, but preserves the name gamma for the imported gamam function.

beta  = (2*PWM1 - PWM0)/(6*PWM1 - PWM0 - 6*PWM2)
alpha = (PWM0 - 2*PWM1)*beta/(gamma(1 + 1/beta)*gamma(1-1/beta))
gamm  = PWM0 - (PWM0 - 2*PWM1)*beta

# Obtain the cumulative distribution of the deficit.
F = (1 + (alpha/(D - gamm))**beta)**-1


# Finally, use this to obtain the probabilities and convert the data to a standardized normal distribution
prob = 1 - F
SAPEI = np.ones((I, J, T)) * np.nan

# Define the constants given in Vicente-Serrano et al. 2010
C0 = 2.515517
C1 = 0.802853
C2 = 0.010328

d1 = 1.432788
d2 = 0.189269
d3 = 0.001308    
    
# Reorder data to reduce number of embedded loops
prob2d = prob.reshape(I*J, T, order = 'F')
SAPEI2d = SAPEI.reshape(I*J, T, order = 'F')


# Calculate SAPEI based on the inverse normal approximation given in Vicente-Serrano et al. 2010, Sec. 3
SAPEISign = 1
for ij in range(I*J):
    for t in range(T):
        if prob2d[ij,t] <= 0.5:
            prob2d[ij,t] = prob2d[ij,t]
            SAPEISign = 1
        else:
            prob2d[ij,t] = 1 - prob2d[ij,t]
            SAPEISign = -1
            
        W = np.sqrt(-2 * np.log(prob2d[ij,t]))
        
        SAPEI2d[ij,t] = SAPEISign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
        
# Reorder the data back to 3D
SAPEI = SAPEI2d.reshape(I, J, T, order = 'F')


# Write the SAPEI data
description = 'This file contains the standardized antecedent precipitation evapotranspiration index ' +\
                  '(SAPEI; unitless), calculated from precipitation and potential ' +\
                  'evaporation from the North American Regional Reanalysis ' +\
                  'dataset. Details on SAPEI and its calculations can be found ' +\
                  'in LI et al. 2020 (https://doi.org/10.1175/JHM-D-19-0298.1). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'sapei: Pentad SAPEI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SAPEI, P['lat'], P['lon'], P['date'], filename = 'sapei.NARR.CONUS.pentad.nc', 
        VarSName = 'sapei', description = description, path = OutPath)


#%%
# cell 14
# Create a plot of SAPEI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(P['ymd'] == ExamineDate)[0]



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
ax.set_title('SAPEI for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(P['lon'], P['lat'], SAPEI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 15

######################
### Calculate SPEI ###
######################

# Details for SPEI can be found in the Vicente-Serrano et al. 2010 paper.

# Determine the moisture deficit
D = P['precip'] - PET['pevap']

# Initialize some needed variables.
I, J, T = P['precip'].size
N = np.unique(P['year']) # Number of observations per time series

frequencies = np.ones((I, J, T)) * np.nan
PWM0 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 0
PWM1 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 1
PWM2 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 2

# Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
for t, date in enumerate(OneYear):
    ind = np.where( (P['month'] == date.month) & (P['day'] == date.day) )[0]
    
    # Get the frequency estimator
    frequencies[:,:,ind] = (stats.mstats.rankdata(D[:,:,ind]) - 0.35)/N
    
    # Get the moments
    PWM0[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**0)*D[:,:,ind], axis = -1)/N
    PWM1[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**1)*D[:,:,ind], axis = -1)/N
    PWM2[:,:,t] = np.nansum(((1 - frequencies[:,:,ind])**2)*D[:,:,ind], axis = -1)/N



# Calculate the parameters of log-logistic distribution, using the equations in the Vicente-Serrano et al. 2010 paper
alpha = np.ones((I, J, OneYear.size)) * np.nan # Scale parameter
beta  = np.ones((I, J, OneYear.size)) * np.nan # Shape parameter
gamm  = np.ones((I, J, OneYear.size)) * np.nan # Origin parameter; not gamm refers to the gamma parameter, but preserves the name gamma for the imported gamam function.

beta  = (2*PWM1 - PWM0)/(6*PWM1 - PWM0 - 6*PWM2)
alpha = (PWM0 - 2*PWM1)*beta/(gamma(1 + 1/beta)*gamma(1-1/beta))
gamm  = PWM0 - (PWM0 - 2*PWM1)*beta

# Obtain the cumulative distribution of the deficit.
F = (1 + (alpha/(D - gamm))**beta)**-1


# Finally, use this to obtain the probabilities and convert the data to a standardized normal distribution
prob = 1 - F
SPEI = np.ones((I, J, T)) * np.nan

# Define the constants given in Vicente-Serrano et al. 2010
C0 = 2.515517
C1 = 0.802853
C2 = 0.010328

d1 = 1.432788
d2 = 0.189269
d3 = 0.001308    
    
# Reorder data to reduce number of embedded loops
prob2d = prob.reshape(I*J, T, order = 'F')
SPEI2d = SPEI.reshape(I*J, T, order = 'F')


# Calculate SPEI based on the inverse normal approximation given in Vicente-Serrano et al. 2010, Sec. 3
SPEISign = 1
for ij in range(I*J):
    for t in range(T):
        if prob2d[ij,t] <= 0.5:
            prob2d[ij,t] = prob2d[ij,t]
            SPEISign = 1
        else:
            prob2d[ij,t] = 1 - prob2d[ij,t]
            SPEISign = -1
            
        W = np.sqrt(-2 * np.log(prob2d[ij,t]))
        
        SPEI2d[ij,t] = SPEISign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
        
# Reorder the data back to 3D
SPEI = SPEI2d.reshape(I, J, T, order = 'F')


# Write the SPEI data
description = 'This file contains the standardized precipitation evapotranspiration index ' +\
                  '(SPEI; unitless), calculated from precipitation and potential ' +\
                  'evaporation from the North American Regional Reanalysis ' +\
                  'dataset. Details on SPEI and its calculations can be found ' +\
                  'in Vicente-Serrano et al. 2010 (https://doi.org/10.1175/2009JCLI2909.1). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'spei: Pentad SPEI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SPEI, P['lat'], P['lon'], P['date'], filename = 'spei.NARR.CONUS.pentad.nc', 
        VarSName = 'spei', description = description, path = OutPath)




#%%
# cell 16
# Create a plot of SPEI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(P['ymd'] == ExamineDate)[0]



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
ax.set_title('SPEI for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(P['lon'], P['lat'], SPEI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 17

#####################
### Calculate SMI ###
#####################

# Details for SMI can be found in the Hunt et al. 2009 paper.

# In a similar vain to the Hunt et al. paper, SMI will be determined for 10, 25, and 40 cm averages of VSM
# These are then averaged together.


# Initialize some variables
WPPercentile = 5
FCPercentile = 95

I, J, T = SM['soilm'].shape
GrowInd = np.where( (SM['month'] >= 4) & (SM['month'] <= 10) )[0] # Percentiles are determined from growing season values.

SMI = np.ones((I, J, T)) * np.nan

# Reshape data into a 2D size.
VSM_00_2d = SM00['soilm'].reshape(I*J, T, order = 'F')
VSM_10_2d = SM10['soilm'].reshape(I*J, T, order = 'F')
VSM_40_2d = SM40['soilm'].reshape(I*J, T, order = 'F')

SMI2d = SMI.reshape(I*J, T, order = 'F')

for ij in range(I*J):
    # First determine the wilting point and field capacity. This is done by examining 5th and 95th percentiles.
    VSM_WP_00 = stats.percentileofscore(VSM_00_2d[ij,GrowInd], WPPercentile)
    VSM_WP_10 = stats.percentileofscore(VSM_10_2d[ij,GrowInd], WPPercentile)
    VSM_WP_40 = stats.percentileofscore(VSM_40_2d[ij,GrowInd], WPPercentile)
    
    VSM_FC_00 = stats.percentileofscore(VSM_00_2d[ij,GrowInd], FCPercentile)
    VSM_FC_10 = stats.percentileofscore(VSM_10_2d[ij,GrowInd], FCPercentile)
    VSM_FC_40 = stats.percentileofscore(VSM_40_2d[ij,GrowInd], FCPercentile)
    
    # Determine the SMI at each level based on equation in section 1 of Hunt et al. 2009
    SMI00 = -5 + 10*(VSM_00_2d[ij,:] - VSM_WP_00)/(VSM_FC_00 - VSM_WP_00)
    SMI10 = -5 + 10*(VSM_10_2d[ij,:] - VSM_WP_10)/(VSM_FC_10 - VSM_WP_10)
    SMI40 = -5 + 10*(VSM_40_2d[ij,:] - VSM_WP_40)/(VSM_FC_40 - VSM_WP_40)
    
    # Average these values together to get the full SMI
    SMI2d[ij,:] = np.nanmean(np.concatenate((SMI00, SMI10, SMI40), axis = 1), axis = 1)
    
    
# Reshape data back to a 3D array.
SMI = SMI2d.reshape(I, J, T, order = 'F')


# Write the SMI data
description = 'This file contains the soil moisture index ' +\
                  '(SMI; unitless), calculated from volumetric soil moisture ' +\
                  'at depths of 0, 10, and 40 cm from the North American Regional Reanalysis ' +\
                  'dataset. Details on SMI and its calculations can be found ' +\
                  'in Hunt et al. 2009 (https://doi.org/10.1002/joc.1749). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'smi: Pentad SMI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SMI, SM['lat'], SM['lon'], SM['date'], filename = 'smi.NARR.CONUS.pentad.nc', 
        VarSName = 'smi', description = description, path = OutPath)


#%%
# cell 18
# Create a plot of SMI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(P['ymd'] == ExamineDate)[0]



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
ax.set_title('SMI for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(SM['lon'], SM['lat'], SMI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 19

######################
### Calculate FDII ###
######################

# Details for FDII can be found in the Otkin et al. 2021 paper.

# Define some base constants
PER_BASE = 15 # Minimum percentile drop in 4 pentads
T_BASE   = 4
DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought

# Next, FDII can be calculated with the standardized soil moisture, or percentiles.
# Use percentiles for consistancy with Otkin et al. 2021
I, J, T = SM['soilm'].shape
SMPer = np.ones((I, J, T)) * np.nan

SM2d    = SM['soilm'].reshape(I*J, T, order = 'F')
SMPer2d = SMPer.reshape(I*J, T, order = 'F')

for t in range(T):
    ind = np.where( (SM['ymd'][t].day == SM['day']) & (SM['ymd'][t].month == SM['month']) )[0]
    
    for ij in range(I*J):
        SMPer2d[ij,t] = stats.percentileofscore(SM2d[ij,ind], SM2d[ij,t])
        
# Determine the rapid intensification based on percentile changes based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
FD_INT = np.ones((I, J, T)) * np.nan
FD_INT2d = FD_INT.reshape(I*J, T, order = 'F')

for ij in range(I*J):
    for t in range(T-2): # Note the last two day is excluded as there is no change to examine
    
        obs = np.ones((9)) * np.nan # Note, the method detail in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
        for npend in np.arange(2, 10+1, 1):
            npend = int(npend)
            if (t+npend) > T: # If t + npend is in the future (beyond the dataset), break the loop and use NaNs for obs instead
                break         # This should not effect results as this will only occur in November to December, outside of the growing season.
            else:
                obs[npend-2] = (SMPer2d[ij,t+npend] - SMPer2d[ij,t])/npend # Note m is the number of pentads the system is corrently looking ahead to.
        
        # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
        #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
        if np.nanmax(obs) < (PER_BASE/T_BASE):
            FD_INT2d[ij,t] = 0
        else:
            FD_INT2d[ij,t] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
            

# Next determine the drought severity component using equation 2 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
DRO_SEV = np.ones((I, J ,T)) * np.nan
DRO_SEV2d = DRO_SEV.reshape(I*J, T, order = 'F')

DRO_SEV2d[:,0] = 0 # Initialize the first entry to 0, since there is no rapid intensification before it

for ij in range(I*J):
    for t in range(1, T-1):
        if (FD_INT2d[ij,t] > 0) & (FD_INT2d[ij, t+1] == 0):
            obs = np.ones((18)) * np.nan # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
            
            Dro_Sum = 0
            for npent in np.arange(0, 18, 1):
                
                if (t+npend) > T:       # For simplicity, set DRO_SEV to 0 when near the end of the dataset (this should not impact anything as it is not in
                    DRO_SEV2d[ij,t] = 0 # the growing season)
                    break
                else:
                    if SMPer2d[ij,t+npent] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                        if npent < 4:
                            # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                            DRO_SEV2d[ij,t] = 0
                            break
                        else:
                            DRO_SEV2d[ij,t] = Dro_Sum/npent
                            break
                            
                    Dro_Sum = Dro_Sum + (DRO_BASE - SMPer2d[ij,t+npent])
        
        # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
        else:
            DRO_SEV2d[ij,t] = 0
            continue
    
# Reorder the data back into 3D data
FD_INT  = FD_INT2d.reshape(I, J, T, order = 'F')
DRO_SEV = DRO_SEV2d.reshape(I, J, T, order = 'F')

# Finally, FDII is the product of the components
FDII = FD_INT * DRO_SEV

# Since FDII has its own rapid intensification and drought components pre-defined, it is worth saving these as well.

# Save the FD_INT variable
description = 'This file contains the rapid intensification component of FDII, ' +\
                  'calculated from volumetric soil moisture averaged from ' +\
                  'depths of 0 to 40 cm from the North American Regional Reanalysis ' +\
                  'dataset. Details on FDII, its components, and their calculations can be found ' +\
                  'in Otkin et al. 2021 (https://doi.org/10.3390/atmos12060741). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'ric: Pentad FDII rapid intensification component (ric; unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(FD_INT, SM['lat'], SM['lon'], SM['date'], filename = 'fd_int.NARR.CONUS.pentad.nc', 
        VarSName = 'ric', description = description, path = OutPath)


# Save the DOR_SEV variable
description = 'This file contains the drought component of FDII, ' +\
                  'calculated from volumetric soil moisture averaged from ' +\
                  'depths of 0 to 40 cm from the North American Regional Reanalysis ' +\
                  'dataset. Details on FDII, its components, and their calculations can be found ' +\
                  'in Otkin et al. 2021 (https://doi.org/10.3390/atmos12060741). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'dc: Pentad FDII drought component (dc; unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(DRO_SEV, SM['lat'], SM['lon'], SM['date'], filename = 'dro_sev.NARR.CONUS.pentad.nc', 
        VarSName = 'dc', description = description, path = OutPath)


# Finally, save the FDII data
description = 'This file contains the flash drought intensity index (FDII; unitless), ' +\
                  'calculated from volumetric soil moisture averaged from ' +\
                  'depths of 0 to 40 cm from the North American Regional Reanalysis ' +\
                  'dataset. Details on FDII, its components, and their calculations can be found ' +\
                  'in Otkin et al. 2021 (https://doi.org/10.3390/atmos12060741). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'fdii: Pentad FDII (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(FDII, SM['lat'], SM['lon'], SM['date'], filename = 'fdii.NARR.CONUS.pentad.nc', 
        VarSName = 'fdii', description = description, path = OutPath)



#%%
# cell 20
# Create a plot of FDII to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 8, 1)

ind = np.where(P['ymd'] == ExamineDate)[0]



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
ax.set_title('FDII for the week of' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(SM['lon'], SM['lat'], FDII[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.325, 0.02, 0.35])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)



