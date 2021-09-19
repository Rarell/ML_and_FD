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
# Load the indices 
    
path = './Data/Indices/'

sesr = LoadNC('sesr', 'sesr.NARR.CONUS.pentad.nc', path = path)
spei = LoadNC('spei', 'spei.NARR.CONUS.pentad.nc', path = path)
fdii = LoadNC('fdii', 'fdii.NARR.CONUS.pentad.nc', path = path)


# In addition, calculate a datetime array that is 1 year in length
OneYearGen = DateRange(datetime(2001, 1, 1), datetime(2001, 12, 31)) # 2001 is a non-leap year
OneYear = np.asarray([date for date in OneYearGen])

OneYearMonth = np.asarray([date.month for date in OneYear])
OneYearDay   = np.asarray([date.day for date in OneYear])

# Determine the path indices will be written to
OutPath = './Data/FD_Data/'


#%%
# cell 7
###############################
### Christian et al. Method ###
###############################

# Calcualte flash droughts using an improved version of the FD identification method from Christian et al. 2019
# This method uses SESR to identify FD

I, J, T = sesr['sesr'].shape

ChFD = np.ones((I, J, T)) * np.nan

#%%
# cell 8
# Calculate and plot the climatology the flash drought to ensure the identification is correct


#### Calcualte the climatology ###

# Initialize variables
I, J, T = ChFD.shape
years  = np.unique(sesr['year'])

AnnFD = np.ones((I, J, years.size)) * np.nan

# Calculate the average number of rapid intensifications and flash droughts in a year
for y in range(years.size):
    yInd = np.where( (years[y] == sesr['year']) & ((sesr['month'] >= 4) & (sesr['month'] <=10)) )[0] # Second set of conditions ensures only growing season values
    
    # Calculate the mean number of rapid intensification and flash drought for each year    
    AnnFD[:,:,y] = np.nanmean(ChFD[:,:,yInd], axis = -1)
    
    # Turn nonzero values to nan (each year gets 1 count to the total)    
    AnnFD[:,:,y] = np.where(( (AnnFD[:,:,y] == 0) | (np.isnan(AnnFD[:,:,y])) ), 
                            AnnFD[:,:,y], 1) # This changes nonzero  and nan (sea) values to 1.
    
    #### This part needs to be commented out if this code is run without additional help, as the land-sea mask was read in seperately from this
    #AnnC4[:,:,y] = np.where(np.isnan(maskSub[:,:,0]), AnnC4[:,:,y], np.nan)

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
ax = fig.add_subplot(2, 1, 1, projection = fig_proj)

# Set the flash drought title
ax.set_title('Percent of Years from 1979 - 2019 with Flash Drought', size = 18)

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
cbax = fig.add_axes([0.85, 0.13, 0.025, 0.75])

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
# cell 9










