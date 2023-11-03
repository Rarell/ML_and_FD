"""
Script to download ERA5 data and reduce it to a more managable daily timescale (1 year of data at the daily times scale is ~ 3 GB per variable)

NOTE: Though the script is set up to download multiple variables at a time, in practice 2 variables
violates some of the constraints in the grib_to_cdf code in cdsapi, and makes the dataset too large
for the classic netCDF format. In short, 2+ variables will return an error.

Acceptable names for --variable and --var_sname_era (i.e., variable names in the downloaded ERA5 dataset):
    --variable:
    - Temperature: 2m_temperature
    - Precipitation: total_precipitation
    - Evaporation: evaporation
    - Potential Evaporation: potential_evaporation
    - Runoff (total): runoff
    - Soil Moisture (0 - 7 cm): volumetric_soil_water_layer_1
    - Soil Moisture (7 - 28 cm): volumetric_soil_water_layer_2
    - Soil Moisture (28 - 100 cm): volumetric_soil_water_layer_3
    - Soil Moisture (100 - 289 cm): volumetric_soil_water_layer_4
    
    --var_sname_era:
    - Temperature: t2m
    - Precipitation: tp
    - Evaporation: e
    - Potential Evaporation: pev
    - Runoff (total): ro
    - Soil Moisture (0 - 7 cm): swvl1
    - Soil Moisture (7 - 28 cm): swvl2
    - Soil Moisture (28 - 100 cm): swvl3
    - Soil Moisture (100 - 289 cm): swvl4
    
Variable units in ERA5 dataset:
    - Temperature: K
    - Precipitation: m
    - Evaporation: m
    - Potential Evaporation: m
    - Runoff: m
    - Soil Moisture: Unitless (m^3 m^-3)
    
Variable units in daily format files:
    - Temperature: K
    - Precipitation: kg m^-2
    - Evaporation: kg m^-2
    - Potential Evaporation: kg m^-2
    - Runoff: kg m^-2
    - Soil Moisture: Unitless (m^3 m^-3)
"""

import os, warnings
import cdsapi
import numpy as np
import argparse
from netCDF4 import Dataset
from datetime import datetime, timedelta

from era5_downloader import downloader

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
    # To add: args.time_series, args.climatology_plot, args.case_studies, args.case_study_years
              
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ERA Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--variable', type=str, nargs='+', default=['2m_temperature'], help='Variable to download (Note only 1 variable should be downloaded at a time)')
    parser.add_argument('--var_sname_era', type=str, nargs='+', default=['t2m'], help='The short name for --variable used in raw datafile')
    parser.add_argument('--var_sname', type=str, nargs='+', default=['tair'], help='What to call the short name for --variable in the processed nc file')
    parser.add_argument('--test', action='store_true', help='Perform a test download (only retrieves 1 year of data)?')
    parser.add_argument('--years', type=int, nargs=2, default=[1979,2021], help='Beginning and ending years to download data for.')
    parser.add_argument('--process', action='store_true', help='Process the data (downscales data to daily scale, which reduces the size by 1/6th)')
    
    return parser
    

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
    
    # Turn off warnings
    warnings.simplefilter('ignore')
    
    # Do a test run?
    if args.test:
        downloader(args.variable, 2021)
    else:
        years = np.arange(args.years[0], args.years[1]+1)
        for year in years:
            print('Downloading data for %d...'%year)
            
            if os.path.exists('era5_raw_%d.nc'%(year)):
            # Processed file does exist: exit
                print("Data is already downloaded.")
            else:
            	downloader(args.variable, year)
            
            # Process the data so it isn't so large?
            if args.process:
                for v, variable in enumerate(args.variable):
                    print('Reducing %s to daily scale for %d...'%(variable, year))
                    # First, load in the data
                    with Dataset('era5_raw_%d.nc'%(year), 'r') as nc:
                        # Load in lat and lon
                        lat = nc.variables['latitude'][:]
                        lon = nc.variables['longitude'][:]
                    
                        # Collect the time + dates
                        time = nc.variables['time'][:]
                        dates = np.asarray([datetime(1900,1,1) + timedelta(hours = int(t)) for t in time])
                    
                        # Initialize the main variable
                        T = time.size; I = lat.size; J = lon.size
                        var = np.ones((int(T/24), I, J)) * np.nan
                        n = 0
                    
                        # Mesh the lat/lon grid
                        lon, lat = np.meshgrid(lon, lat)
                    
                        # Rather than load in the entire (18 GB) dataset, load in a small portion and immediately reduce it to daily size to ease computation
                        for t in range(int(T/24)):
                            var[t,:,:] = np.nanmean(nc.variables[args.var_sname_era[v]][n:n+24,:,:], axis = 0)
                            n = n + 24
                
                    # Several variables need unit conversions for consistency with other datasets
                    if (variable == 'total_precipitation') | (variable == 'evaporation') | (variable == 'potential_evaporation') | (variable == 'runoff'):
                        print('Converting units...')
                        # These variables are in units of meters; multiply by the density of water to put them in kg m^-2 (how much mass of water in a given area)
                        var = var * 1000
                    
                    # Check the data
                    print('Date start and end points:', dates[0], dates[-1])
                    print('Data shape:', var.shape)
                
                    # Write the reduced data as a netcdf file
                    description = "Daily ERA5 reanalysis data for %s"%variable
                    write_nc(var, lat, lon, dates[::24], filename = '%s%d.nc'%(variable, year), var_sname = args.var_sname[v], description = description)
                
                # Delete the extra large file
                print('Removing large datafile...')
                os.remove('era5_raw_%d.nc'%(year))
                
    print('Finished')
                    