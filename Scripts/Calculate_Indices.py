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

Flash drought indices omitted:
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
- Sohrabi et al. 2015 (for SODI): https://doi.org/10.1061/(ASCE)HE.1943-5584.0001213
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
# Load all the data files

path = './Data/Processed_Data/'

T    = LoadNC('temp', 'temperature_2m.NARR.CONUS.pentad.nc', sm = False, path = path)
ET   = LoadNC('evap', 'evaporation.NARR.CONUS.pentad.nc', sm = False, path = path)
PET  = LoadNC('pevap', 'potential_evaporation.NARR.CONUS.pentad.nc', sm = False, path = path)
P    = LoadNC('precip', 'accumulated_precipitation.NARR.CONUS.pentad.nc', sm = False, path = path)
RO   = LoadNC('ro', 'baseflow_runoff.NARR.CONUS.pentad.nc', sm = False, path = path)
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
# cell 8
# Load and subset the land-sea mask

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

######################
### Calculate SESR ###
######################

# Details in SESR can be found in the Christian et al. 2019 paper.

# Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
ESR = ET['evap']/PET['pevap']

# Remove values exceed a certain limit as they are likely an error
ESR[ESR < 0] = np.nan
ESR[ESR > 3] = np.nan

# Determine the climatological mean and standard deviations of ESR
ESRMean, ESRstd = CalculateClimatology(ESR, pentad = True)

# Calculate SESR; it is the standardized ESR
I, J, T = ESR.shape

SESR = np.ones((I, J, T)) * np.nan

for n, date in enumerate(OneYear[::5]):
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0]
    
    for t in ind:
        SESR[:,:,t] = (ESR[:,:,t] - ESRMean[:,:,n])/ESRstd[:,:,n]
        

# Remove any sea data points
SESR[maskSub[:,:,0] == 0] = np.nan

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
# cell 10
# Create a plot of SESR to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(ET['ymd'] == ExamineDate)[0][0]



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
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SESR for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)


#%%
# cell 11

######################
### Calculate EDDI ###
######################

# Details are found in the Hobbins et al. 2016 paper.

# Initialize the set of probabilities of getting a certain PET.
I, J, T = PET['pevap'].shape

prob = np.ones((I, J, T)) * np.nan
EDDI = np.ones((I, J, T)) * np.nan

N = np.unique(PET['year']).size # Number of observations per time series

# Define the constants given in Hobbins et al. 2016
C0 = 2.515517
C1 = 0.802853
C2 = 0.010328

d1 = 1.432788
d2 = 0.189269
d3 = 0.001308

# Determine the probabilities of getting PET at time t.
for date in OneYear[::5]:
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

# Remove any sea data points
EDDI[maskSub[:,:,0] == 0] = np.nan

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
# cell 12
# Create a plot of EDDI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(PET['ymd'] == ExamineDate)[0][0]



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
cmin = -2; cmax = 2; cint = 0.5
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('EDDI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)




#%%
# cell 13

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

for n, date in enumerate(OneYear[::5]):
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0]
    
    for t in ind:
        SEDI[:,:,t] = (ED[:,:,t] - EDMean[:,:,n])/EDstd[:,:,n]
        

# Remove any sea data points
SEDI[maskSub[:,:,0] == 0] = np.nan
        
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
# cell 14
# Create a plot of SEDI to check calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(PET['ymd'] == ExamineDate)[0][0]



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
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SEDI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)




#%%
# cell 15

#######################
### Calculate SAPEI ###
#######################

# Details for SAPEI can be found in the Li et al. 2020b paper.

a = 0.903 # Note this decay rate is defined by keeping the total decay (13%) after 100 days or 20 pentads.
          # These values may be adjusted, as SAPEI with this decay/memory is like unto a 3-month SPEI
          # (see Li et al. 2020b sections 3a and 4a).

# Initialize the moisture deficit D
I, J, T = P['precip'].shape
D = np.zeros((I, J, T))

NDays = 100 # Number of days in the decay/memory
counters = np.arange(1, (NDays/5)+1)

for t in range(T):
    for i in counters:
        i = int(i)
        if i > t:
            break
        
        moistDeficit = (a**i) * (P['precip'][:,:,t-i] - PET['pevap'][:,:,t-i])
        
        D[:,:,t] = D[:,:,t] + moistDeficit

# From here, continue to perform the transformation of D from a log-logistic distribution to normal as detailed in Vicente-Serrano et al. 2010
N = np.unique(P['year']).size # Number of observations per time series

frequencies = np.ones((I, J, T)) * np.nan
PWM0 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 0
PWM1 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 1
PWM2 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 2

# Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
for t, date in enumerate(OneYear[::5]):
    ind = np.where( (P['month'] == date.month) & (P['day'] == date.day) )[0]
    
    # Get the frequency estimator
    frequencies[:,:,ind] = (stats.mstats.rankdata(D[:,:,ind], axis = -1) - 0.35)/N
    
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
F = np.ones((I, J, T)) * np.nan

for n, date in enumerate(OneYear[::5]):
    ind = np.where( (date.month == P['month']) & (date.day == P['day']) )[0]
    
    for t in ind:
        F[:,:,t] = (1 + (alpha[:,:,n]/(D[:,:,t] - gamm[:,:,n]))**beta[:,:,n])**-1



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

# Remove any sea data points
SAPEI[maskSub[:,:,0] == 0] = np.nan

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
# cell 16
# Create a plot of SAPEI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(P['ymd'] == ExamineDate)[0][0]



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
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SAPEI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 17

######################
### Calculate SPEI ###
######################

# Details for SPEI can be found in the Vicente-Serrano et al. 2010 paper.

# Determine the moisture deficit
D = P['precip'] - PET['pevap']

# Initialize some needed variables.
I, J, T = P['precip'].shape
N = np.unique(P['year']).size # Number of observations per time series

frequencies = np.ones((I, J, T)) * np.nan
PWM0 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 0
PWM1 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 1
PWM2 = np.ones((I, J, OneYear.size)) * np.nan # Probability weighted moment of 2

# Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
for t, date in enumerate(OneYear[::5]):
    ind = np.where( (P['month'] == date.month) & (P['day'] == date.day) )[0]
    
    # Get the frequency estimator
    frequencies[:,:,ind] = (stats.mstats.rankdata(D[:,:,ind], axis = -1) - 0.35)/N
    
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
F = np.ones((I, J, T)) * np.nan

for n, date in enumerate(OneYear[::5]):
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0]
    
    for t in ind:
        F[:,:,t] = (1 + (alpha[:,:,n]/(D[:,:,t] - gamm[:,:,n]))**beta[:,:,n])**-1


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

# Remove any sea data points
SPEI[maskSub[:,:,0] == 0] = np.nan

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
# cell 18
# Create a plot of SPEI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(P['ymd'] == ExamineDate)[0][0]



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
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SPEI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 19

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
    VSM_WP_00 = stats.scoreatpercentile(VSM_00_2d[ij,GrowInd], WPPercentile)
    VSM_WP_10 = stats.scoreatpercentile(VSM_10_2d[ij,GrowInd], WPPercentile)
    VSM_WP_40 = stats.scoreatpercentile(VSM_40_2d[ij,GrowInd], WPPercentile)
    
    VSM_FC_00 = stats.scoreatpercentile(VSM_00_2d[ij,GrowInd], FCPercentile)
    VSM_FC_10 = stats.scoreatpercentile(VSM_10_2d[ij,GrowInd], FCPercentile)
    VSM_FC_40 = stats.scoreatpercentile(VSM_40_2d[ij,GrowInd], FCPercentile)
    
    # Determine the SMI at each level based on equation in section 1 of Hunt et al. 2009
    SMI00 = -5 + 10*(VSM_00_2d[ij,:] - VSM_WP_00)/(VSM_FC_00 - VSM_WP_00)
    SMI10 = -5 + 10*(VSM_10_2d[ij,:] - VSM_WP_10)/(VSM_FC_10 - VSM_WP_10)
    SMI40 = -5 + 10*(VSM_40_2d[ij,:] - VSM_WP_40)/(VSM_FC_40 - VSM_WP_40)
    
    # Average these values together to get the full SMI
    SMI2d[ij,:] = np.nanmean(np.stack((SMI00, SMI10, SMI40), axis = 1), axis = 1)
    
    
# Reshape data back to a 3D array.
SMI = SMI2d.reshape(I, J, T, order = 'F')

# Remove any sea data points
SMI[maskSub[:,:,0] == 0] = np.nan

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
# cell 20
# Create a plot of SMI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(P['ymd'] == ExamineDate)[0][0]



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
cmin = -5; cmax = 5; cint = 0.5
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SMI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)

#%%
# cell 21

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
    
        obs = np.ones((9)) * np.nan # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
        for npend in np.arange(2, 10+1, 1):
            npend = int(npend)
            if (t+npend) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use NaNs for obs instead
                break          # This should not effect results as this will only occur in November to December, outside of the growing season.
            else:
                obs[npend-2] = (SMPer2d[ij,t+npend] - SMPer2d[ij,t])/npend # Note npend is the number of pentads the system is corrently looking ahead to.
        
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
        if (FD_INT2d[ij,t] > 0):
            
            Dro_Sum = 0
            for npent in np.arange(0, 18+1, 1): # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                
                if (t+npent) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset (this should not impact anything as it is not in
                    DRO_SEV2d[ij,t] = 0 # the growing season)
                    break
                else:
                    Dro_Sum = Dro_Sum + (DRO_BASE - SMPer2d[ij,t+npent])
                    
                    if SMPer2d[ij,t+npent] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                        if npent < 4:
                            # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                            DRO_SEV2d[ij,t] = 0
                            break
                        else:
                            DRO_SEV2d[ij,t] = Dro_Sum/npent # Terminate the loop and determine the drought severity if the drought condition is broken
                            break
                        
                    elif (npent >= 18): # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                        DRO_SEV2d[ij,t] = Dro_Sum/npent
                        break
                    else:
                        pass
        
        # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
        else:
            DRO_SEV2d[ij,t] = 0
            continue
    
# Reorder the data back into 3D data
FD_INT  = FD_INT2d.reshape(I, J, T, order = 'F')
DRO_SEV = DRO_SEV2d.reshape(I, J, T, order = 'F')

# Finally, FDII is the product of the components
FDII = FD_INT * DRO_SEV

# Remove any sea data points
FD_INT[maskSub[:,:,0] == 0] = np.nan
DRO_SEV[maskSub[:,:,0] == 0] = np.nan
FDII[maskSub[:,:,0] == 0] = np.nan


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
# cell 22
# Create a plot of FDII to check the calculations

# Determine the date to be examined
StartExamineDate = datetime(2012, 5, 1)
EndExamineDate   = datetime(2012, 8, 1)

ind = np.where( (SM['year'] == StartExamineDate.year) & (SM['month'] >= StartExamineDate.month) & (SM['month'] <= EndExamineDate.month) )[0]



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
cmin = 0; cmax = 60; cint = 1.0
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'gist_rainbow_r', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('FDII for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(SM['lon'], SM['lat'], np.nanmax(FDII[:,:,ind], axis = -1), levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)



#%%
# cell 23

######################
### Calculate SODI ###
######################

# Details for SODI can be found in the Sohrabi et al. 2015 paper.

# In order to get SODI, moisture loss from the soil column is needed. This is assumed to be the ET - P
L = ET['evap'] - P['precip']

# If P > ET, there is no moisture loss.
L[P['precip'] > ET['evap']] = 0

# Next, determine the available water content in the soil using the FC and WP estimates from Hunt et al. 2009.
I, J, T = SM['soilm'].shape
GrowInd = np.where( (SM['month'] >= 4) & (SM['month'] <= 10) )[0] # Percentiles are determined from growing season values.

WPPercentile = 5
FCPercentile = 95

AWC = np.ones((I, J)) * np.nan

# Reshape data into a 2D size.
VSM2d = SM['soilm'].reshape(I*J, T, order = 'F') 

AWC2d = AWC.reshape(I*J, order = 'F')

for ij in range(I*J):
    # First determine the wilting point and field capacity. This is done by examining 5th and 95th percentiles.
    VSM_WP = stats.scoreatpercentile(VSM2d[ij,GrowInd], WPPercentile)
    
    VSM_FC = stats.scoreatpercentile(VSM2d[ij,GrowInd], FCPercentile)
    
    # The available water content is simply the difference between field capacity and wilting point
    AWC2d[ij] = VSM_FC - VSM_WP
    
# Convert AWC back to 2D data
AWC = AWC2d.reshape(I, J, order = 'F')

# The soil moisture deficiency then becomes the difference between AWC and soil moisutre
SMD = np.ones((I, J, T)) * np.nan
for t in range(T):
    SMD[:,:,t] = AWC[:,:] - SM['soilm'][:,:,t]
    
# Note to get the volumetric water content in fractional form, it is the mass of water lost divided by rho_l (to convert to volume), divided by sample volume.
# To invert this, multiply this by rho_l to get the mass of water in a volume of soil, then multiply by soil depth to get the mass of water in an area of soil.
### Note: This is primarily done to bring the soil moisture variable (SMD) to the same units as the other variables (kg m^-2). I.e., it is done for unit consistency.
SoilDepth = 0.4  # m
rho_l     = 1000 # kg m^-3

SMD = SMD * SoilDepth * rho_l

# Next, calculate the moisture deficit given in equation 1 of Sohrabi et al. 2015.
D = np.ones((I, J, T)) * np.nan
# D[:,:,6:] = (P['precip'][:,:,6:] + L[:,:,6:] + RO['ro'][:,:,:-6]) - (PET['pevap'][:,:,6:] + SMD[:,:,:-6]) # Note D[:,:,:-1] means each D is at the start of the respective delta t, consistent with other indices calculated thus far.

for t in range(12, T): # Use a monthly average for variables in the previous month
    D[:,:,t] = (P['precip'][:,:,t] + L[:,:,t] + np.nanmean(RO['ro'][:,:,t-12:t-6], axis = -1)) - (PET['pevap'][:,:,t] + np.nanmean(SMD[:,:,t-12:t-6], axis = -1))

# Next, perform the Box-Car transformation and standardize the data to create SODI, according to equations 5 and 6 in Sohrabi et al. 2015
SODI = np.ones((I, J, T)) * np.nan

SODI2d = SODI.reshape(I*J, T, order = 'F')
D2d    = D.reshape(I*J, T, order = 'F')

for ij in range(I*J):
    
    # From looking around at various features, it seems as if lambda2 in the Box-Car transformation is the minimum value of the data, so that all values are > 0.
    if np.nanmin(D2d[ij,:]) < 0: # Ensure the shift is positive
        lambda2 = -1 * np.nanmin(D2d[ij,:])
    else:
        lambda2 = np.nanmin(D2d[ij,:])
    
    # Perform the Box-Car transformation. Note boxcar only accepts a vector, so this has to be done for 1 grid point at a time
    y, lambda1 = stats.boxcox(D2d[ij,:] + lambda2 + 0.001) # 0.001 should have a small impact on values, but ensure D + lambda2 is not 0 at any point
    
    # Determine the climatology of the transformed data
    yMean, yStd = CalculateClimatology(y, pentad = True)
    
    # Standardize the transformed data to calculate SODI for the grid point
    for n, date in enumerate(OneYear[::5]):
        ind = np.where( (date.month == SM['month']) & (date.day == SM['day']) )[0]
        
        for i in ind:
            SODI2d[ij,i] = (y[i] - yMean[n])/yStd[n]

# Transform the data back into a 3D array
SODI = SODI2d.reshape(I, J, T, order = 'F')

# Remove any sea data points
SODI[maskSub[:,:,0] == 0] = np.nan

# Write the data
description = 'This file contains the soil moisture drought index (SODI; unitless), ' +\
                  'calculated from volumetric soil moisture averaged from ' +\
                  'depths of 0 to 40 cm, precipitation, ET, PET, and precipitation from the North American Regional Reanalysis ' +\
                  'dataset. Details on SODI, its components, and their calculations can be found ' +\
                  'in Sohrabi et al. 2015 (https://doi.org/10.1061/(ASCE)HE.1943-5584.0001213). ' +\
                  'The data is subsetted to focus on the contential ' +\
                  'U.S., and it is on the weekly timescale. Data ranges form ' +\
                  'Jan. 1 1979 to Dec. 31 2020. Variables are:\n' +\
                  'sodi: Pentad SODI (unitless) data. ' +\
                  'Variable format is x by y by time\n' +\
                  'lat: 2D latitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'lon: 2D longitude corresponding to the grid for apcp. ' +\
                  'Variable format is x by y.\n' +\
                  'date: List of strings containing dates corresponding to the ' +\
                  'start of the week for the corresponding time point in apcp. Dates ' +\
                  'are in %Y-%m-%d format. Leap days were excluded for ' +\
                  'simplicity. Variable format is time.'


WriteNC(SODI, SM['lat'], SM['lon'], SM['date'], filename = 'sodi.NARR.CONUS.pentad.nc', 
        VarSName = 'sodi', description = description, path = OutPath)


#%%
# cell 24
# Create a plot of SODI to check the calculations

# Determine the date to be examined
ExamineDate = datetime(2012, 7, 30)

ind = np.where(P['ymd'] == ExamineDate)[0][0]



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
cmin = -2; cmax = 2; cint = 0.5
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs) - 1
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

data_proj = ccrs.PlateCarree()
fig_proj  = ccrs.PlateCarree()

# Create the figure
fig = plt.figure(figsize = [12, 16])
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)

# Set title
ax.set_title('SODI for ' + ExamineDate.strftime('%Y-%m-%d'), fontsize = 16)

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
cs = ax.contourf(SM['lon'], SM['lat'], SODI[:,:,ind], levels = clevs, cmap = cmap,
                  transform = data_proj, extend = 'both', zorder = 1)

# Create and set the colorbar
cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
cbar = fig.colorbar(cs, cax = cbax)

# Set the extent
ax.set_extent([-130, -65, 25, 50], crs = fig_proj)

plt.show(block = False)