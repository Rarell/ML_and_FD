# Preprocesses 1 year of the raw NLDAS netcdf files
# Parses data into individual variables, and averages the data down to daily timescale

import os
import re
import argparse
import time
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ERA Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--dataset', type=str, default='./', help='Path to the raw NLDAS data')
    parser.add_argument('--test', action='store_true', help='Perform a test run')
    parser.add_argument('--year', type=int, default=1979, help='The year of data being processed')
    parser.add_argument('--years', type=int, nargs='+', default=[1979, 1980], help='All years of data to process')
    
    return parser
  
  
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
    
    
# A cell to calculate a daily mean.
def daily_compression(X, dates, summation = False):
    '''
    Take subdaily data and average/sum it to daily data.
    
    Inputs:
    :param X: The 3D or 4D data that is being averaged/summed to a daily format.
    :param dates: Array of datetimes corresponding to the timestamps in data.
    :param summation: A boolean value indicating whether the data is compressed 
                      to a daily mean or daily accumulation.
    
    Outputs:
    :param X_daily: The variable X in a daily mean/accumulation format.
    :param timestamps: Array of datetimes corresponding to each day.
    '''
    
    # Initialize values
    years = np.array([date.year for date in dates])
    months = np.array([date.month for date in dates])
    days = np.array([date.day for date in dates])
    
    time_gen = date_range(dates[0], dates[-1])
    
    timestamps = np.array([date for date in time_gen])
    
    T, I, J = X.shape
    
    T = len(timestamps)
    
    X_daily = np.ones((T, I, J)) * np.nan
    
    # Average/sum the data to the daily timescale
    for t, time in enumerate(timestamps):
        ind = np.where( (time.day == days) & (time.month == months) )[0]
        # Sum the data?
        if summation == True:
            X_daily[t,:,:] = np.nansum(X[ind,:,:], axis = 0)
        else:
            X_daily[t,:,:] = np.nanmean(X[ind,:,:], axis = 0)

            
    return X_daily, timestamps
    
    
# Function to write netcdf files  
def write_nc(var, lat, lon, dates, filename = 'tmp.nc', var_sname = 'tmp', description = 'Description', path = './'):
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
        nc.createVariable(var_sname, var.dtype, ('time', 'x', 'y'))
        nc.variables[str(var_sname)][:,:,:] = var[:,:,:]


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    start = time.time()
    
    # Collect the files to be processed
    print('Collecting NLDAS files...')
    files = ['%s/%s'%(args.dataset,f) for f in os.listdir(args.dataset) if re.match(r'HTTP_*', f)]
    files.sort()
    
    #T = int(len(files)/len(args.years))
    
    # Load a file for the lat and lon
    print('Loading lat/lon...')
    with Dataset(files[0], 'r') as nc:
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        depth = nc.variables['depth_2'][:]
        
    depth = depth/100 # Depth in meters
    
    # Get the datetimes
    print('Obtaining time stamps...')
    #if args.year == 1979:
    #    start_time = datetime(args.year, 1, 2, 1, 0) # 1979 starts a little later
    #else:
    #    start_time = datetime(args.year, 1, 1, 0, 0)
    #    
    #end_time = datetime(args.year, 12, 31, 23, 0)
    
    #dates = []
    
    #tmp_time = start_time
    #while tmp_time <= end_time:
    #    dates.append(tmp_time)
    #    tmp_time = tmp_time + timedelta(hours = 1)
        
    #dates = np.array(dates)
    
    # Initialize the full variables
    print('Initializing variables...')
    I = lat.size; J = lon.size
    
    # Mesh the lat and lon grid
    lon, lat = np.meshgrid(lon, lat)
    
    for year in args.years:
    
        # The exact length of the time axis depends on whether the current year is a leap year
        if np.mod(year, 4) == 0:
            T = 366*24
        else:
            T = 365*24
    
        tair = np.ones((T, I, J)) * np.nan
        rain = np.ones((T, I, J)) * np.nan
        evap = np.ones((T, I, J)) * np.nan
        pevap = np.ones((T, I, J)) * np.nan
        sm = np.ones((T, I ,J)) * np.nan
        runoff = np.ones((T, I, J)) * np.nan
    
        dates = []
    
        # Collect the data
        print('Collecting data for %d...'%year)
        t = 0
        for file in files:
        
            # Ignore the file if it isn't in the current year
            #print(int(file[102:106]))
            if np.invert(int(file[102:106]) == year):
                continue
    
            # This is a very hardcoded solution to the dates problem, but:
            # .nc filenames sent by NLDAS are long nad convoluted, with no way (known to the author)
            # to isolate particular parts, and the .nc files all contain forecast time (which is 0)
            # and not time stamps, and some files may be randomly missing.
            dates.append(datetime(int(file[102:106]), int(file[106:108]), int(file[108:110]), int(file[111:113]), 0))
            with Dataset(file, 'r') as nc:
                tair[t,:,:] = nc.variables['AVSFT'][0,:,:] # Surface skin temperature in K
                rain[t,:,:] = nc.variables['ARAIN'][0,:,:] # Accumulated rain in kg m^-2
                evap[t,:,:] = nc.variables['EVP'][0,:,:] # Evaporation in kg m^-2
                pevap[t,:,:] = nc.variables['PEVPR'][0,:,:]*1e3/(2.5e6) # Potential evaporation in kg/m^-2 (raw value in w m^-2 times the density of water and divided by the latent heat of vaporization)
                sm[t,:,:] = nc.variables['RZSM'][0,0,:,:]/(1e3*depth) # Root zone (50 cm) soil moisture in faction/unitless (raw value in kg m^-2, divided by the density of water and soil depth); note soil moisture at other depths are available
                runoff[t,:,:] = nc.variables['SSRUN'][0,:,:] # Surface runoff in kg m^-2
                
            t = t + 1
            
        print(dates[0], dates[-1])
        dates = np.array(dates)
            
        # Average/sum the data down to the daily timescale
        print('Compressing data to daily timescale...')
        tair, timestamps = daily_compression(tair, dates, summation = False)
        rain, timestamps = daily_compression(rain, dates, summation = True)
        evap, timestamps = daily_compression(evap, dates, summation = True)
        pevap, timestamps = daily_compression(pevap, dates, summation = True)
        sm, timestamps = daily_compression(sm, dates, summation = False)
        runoff, timestamps = daily_compression(runoff, dates, summation = True)
    
        print('Shape of daily data:', tair.shape)
    
        # Write the data
        print('Writing data...')
        desc_tair = "Daily NLDAS2 reanalysis data for temperature in K"
        desc_rain = "Daily NLDAS2 reanalysis data for accumulated rainfall in kg m^-2"
        desc_evap = "Daily NLDAS2 reanalysis data for evaporation in kg m^-2"
        desc_pevap = "Daily NLDAS2 reanalysis data for potential evaporation in kg m^-2"
        desc_sm = "Daily NLDAS2 reanalysis data for root zone (50cm) soil moisture in fraction/unitless"
        desc_runoff = "Daily NLDAS2 reanalysis data for surface runoff in kg m^-2"
    
        write_nc(tair, lat, lon, timestamps, filename = 'temperature_%d.nc'%year, var_sname = 'temp', description = desc_tair, path = './')
        write_nc(rain, lat, lon, timestamps, filename = 'accumulated_rainfall_%d.nc'%year, var_sname = 'precip', description = desc_rain, path = './')
        write_nc(evap, lat, lon, timestamps, filename = 'evaporation_%d.nc'%year, var_sname = 'evap', description = desc_evap, path = './')
        write_nc(pevap, lat, lon, timestamps, filename = 'potential_evaporation_%d.nc'%year, var_sname = 'pevap', description = desc_pevap, path = './')
        write_nc(sm, lat, lon, timestamps, filename = 'root_zone_soil_moisture_%d.nc'%year, var_sname = 'soilm', description = desc_sm, path = './')
        write_nc(runoff, lat, lon, timestamps, filename = 'surface_runoff_%d.nc'%year, var_sname = 'ro', description = desc_runoff, path = './')
    
    # Loop would end here
    
    # Remove numerous, un-needed files
    print('Removing numerous, large files...')
    [os.remove(file) for file in files]
    
    print('Finished. The program took %4.2f minutes to process the data.'%((time.time()-start)/60))
    
    