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
- Christian et al. 2019 (for SESR method): https://doi.org/10.1175/JHM-D-18-0198.1
- Noguera et al. 2020 (for SPEI method): https://doi.org/10.1111/nyas.14365
- Pendergrass et al. 2020 (for EDDI method): https://doi.org/10.1038/s41558-020-0709-0
- Li et al. 2020 (for SEDI method): https://doi.org/10.1016/j.catena.2020.104763
- Liu et al. 2020 (for soil moisture method): https://doi.org/10.1175/JHM-D-19-0088.1
- Otkin et al. 2020 (for FDII method): https://doi.org/10.3390/atmos12060741


TODO:
- Modify map creation functions to include the whole world
- Add Li et al. 2020 Method
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

# Import a custom script
from Raw_Data_Processing import *
from Calculate_Indices import *


#%%
##############################################

# Create a function to make climatology maps for FD
def display_fd_climatology(fd, lat, lon, dates, method, model = 'narr', path = './Figures', grow_season = False, years = None, months = None):
    '''
    Display the climatology of flash drought (percentage of years with flash drought)
    
    Inputs:
    :param fd: Input flash drought (FD) data to be plotted, time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param method: String describing the method used to calculate the flash drought
    :param model: String describing what reanalysis model the data comes from. Used to name the figure
    :param path: Path the figure will be saved to
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons.
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])

        
    ### Calcualte the climatology ###

    # Initialize variables
    T, I, J = fd.shape
    all_years = np.unique(years)
    
    ann_fd = np.ones((all_years.size, I, J)) * np.nan
    
    # Calculate the average number of rapid intensifications and flash droughts in a year
    for y in range(all_years.size):
        if grow_season:
            y_ind = np.where( (all_years[y] == years) )[0]
        else:
            y_ind = np.where( (all_years[y] == years) & ((months >= 4) & (months <= 10)) )[0] # Second set of conditions ensures only growing season values
        
        # Calculate the mean number of flash drought for each year    
        ann_fd[y,:,:] = np.nanmean(fd[y_ind,:,:], axis = 0)
        
        # Turn nonzero values to 1 (each year gets 1 count to the total)    
        ann_fd[y,:,:] = np.where(( (ann_fd[y,:,:] == 0) | (np.isnan(ann_fd[y,:,:])) ), 
                                 ann_fd[y,:,:], 1) # This changes nonzero  and nan (sea) values to 1.
    

    

    # Calculate the percentage number of years with rapid intensifications and flash droughts
    per_ann_fd = np.nansum(ann_fd[:,:,:], axis = 0)/all_years.size
    
    # Turn 0 values into nan
    per_ann_fd = np.where(per_ann_fd != 0, per_ann_fd, np.nan)
    
    

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
    ax.set_title('Percent of Years from %s - %s with %s Flash Drought'%(all_years[0], all_years[-1], method), size = 18)
    
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
    cs = ax.pcolormesh(lon, lat, per_ann_fd*100, vmin = cmin, vmax = cmax,
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
    
    # Show the figure
    plt.show(block = False)
    
    # Save the figure
    filename = '%s_%s_flash_drought_climatology.png'%(model, method)
    plt.savefig('%s/%s'%(path, filename), bbox_inches = 'tight')
    plt.show(block = False)
    

#%%
##############################################

# Create a function to calcualte flash droughts using an improved version of the FD identification method from Christian et al. 2019
# This method uses SESR to identify FD

def christian_fd(sesr, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using an updated version of the method described in Christian et al. 2019
    (https://doi.org/10.1175/JHM-D-18-0198.1). This method uses the evaporative stress ratio (SESR) to 
    identify flash drought.
    
    Inputs:
    :param sesr: Input SESR values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in sesr
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
        
        

    # Initialize some variables
    T, I, J = sesr.shape
    sesr_inter = np.ones((T, I, J)) * np.nan
    sesr_filt  = np.ones((T, I, J)) * np.nan
    
    sesr2d = sesr.reshape(T, I*J, order = 'F')
    sesr_inter2d = sesr_inter.reshape(T, I*J, order = 'F')
    sesr_filt2d  = sesr_filt.reshape(T, I*J, order = 'F')
    
    mask2d = mask.reshape(I*J, order = 'F')
    
    x = np.arange(-6.5, 6.5, (13/T))[:-1] # a variable covering the range of SESR with 1 entry for each time step
    print(x.size, T)
    
    # Parameters for the filter
    WinLength = 21 # Window length of 21 pentads
    PolyOrder = 4

    # Perform a basic linear interpolation for NaN values and apply a SG filter
    print('Applying interpolation and Savitzky-Golay filter to SESR')
    for ij in range(I*J):
        if mask2d[ij] == 0:
            continue
        else:
            pass
        
        # Perform a linear interpolation to remove NaNs
        ind = np.isfinite(sesr2d[:,ij])
        if np.nansum(ind) == 0:
            continue
        else:
            pass
        
        ind = np.where(ind == True)[0]
        interp_func = interpolate.interp1d(x[ind], sesr2d[ind,ij], kind = 'linear', fill_value = 'extrapolate')
        
        sesr_inter2d[:,ij] = interp_func(x)
        
        # Apply the Savitzky-Golay filter to the interpolated SESR data
        sesr_filt2d[:,ij] = signal.savgol_filter(sesr_inter2d[:,ij], WinLength, PolyOrder)
        
    # Reorder SESR back to 3D data
    sesr_filt = sesr_filt2d.reshape(T, I, J, order = 'F')



    # Determine the change in SESR
    print('Calculating the change in SESR')
    delta_sesr  = np.ones((T, I, J)) * np.nan
    delta_sesr_z = np.ones((T, I, J)) * np.nan
    
    delta_sesr[1:,:,:] = sesr_filt[1:,:,:] - sesr_filt[:-1,:,:]
    
    # Standardize the change in SESR
    delta_sesr_climo = collect_climatology(delta_sesr, dates, start_year = start_year, end_year = end_year)
    
    delta_sesr_mean, delta_sesr_std = calculate_climatology(delta_sesr_climo, pentad = True)
    
    # Find the time stamps for a singular year
    ind = np.where(years == 1999)[0] # Note, any non-leap year will do
    one_year = dates[ind]

    for n, date in enumerate(one_year):
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        for t in ind:
            delta_sesr_z[t,:,:] = (delta_sesr[t,:,:] - delta_sesr_mean[n,:,:])/delta_sesr_std[n,:,:]



    # Begin the flash drought calculations
    print('Identifying flash drought')
    fd = np.ones((T, I, J)) * np.nan

    fd2d = fd.reshape(T, I*J, order = 'F')
    dsesr2d = delta_sesr_z.reshape(T, I*J, order = 'F')

    dsesr_percentile = 25
    sesr_percentile  = 20
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]
    
    for ij in range(I*J):
        if mask2d[ij] == 0:
            continue
        else:
            pass
        
        start_date = dates[-1]
        for t in range(T):
            ind = np.where( (dates[t].month == months) & (dates[t].day == days) )[0]
            
            # Determine the percentiles of dSESR and SESR
            ri_crit = np.nanpercentile(dsesr2d[ind,ij], dsesr_percentile)
            dc_crit = np.nanpercentile(sesr_filt2d[ind,ij], sesr_percentile)
            
            # If start_date != dates[-1], the rapid intensification criteria is satisified
            # If the rapid intensification and drought component criteria are satisified (and FD period is 30+ days)
            # then FD occurs
            if ( (dates[t] - start_date) >= min_change) & (sesr_filt2d[t,ij] <= dc_crit):
                fd2d[t,ij] = 1
            else:
                fd2d[t,ij] = 0
            
            # # If the change in SESR is below the criteria, change the start date of the flash drought
            if (dsesr2d[t,ij] <= ri_crit) & (start_date == dates[-1]):
                start_date = dates[t]
            elif (dsesr2d[t,ij] <= ri_crit) & (start_date != dates[-1]):
                pass
            else:
                start_date = dates[-1]
            
    # Re-order the flash drought back into a 3D array
    fd = fd2d.reshape(T, I, J, order = 'F')
    print('Done')
    
    return fd


#%%
##############################################

# Calcualte flash droughts using a FD identification method from Noguera et al. 2020
# This method uses SPEI to identify FD

def nogeura_fd(spei, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using the method described in Nogeura et al. 2020 (https://doi.org/10.1111/nyas.14365). 
    This method uses the standardized precipitation evaporation index (SPEI) to 
    identify flash drought.
    
    Inputs:
    :param spei: Input SPEI values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in spei
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])

    # Determine the change in SPEI across a 1 month (30 day = 6 pentad) period
    print('Calculating the change in SPEI')
    
    T, I, J = spei.shape
    delta_spei = np.ones((T, I, J)) * np.nan
    
    delta_spei[6:,:,:] = spei[6:,:,:] - spei[:-6,:,:] # Set the indices so that each entry in delta_spei corrsponds to the end date of the difference

    # Reorder data into 2D arrays for fewer embedded loops
    spei2d = spei.reshape(T, I*J, order = 'F')
    delta_spei2d = delta_spei.reshape(T, I*J, order = 'F')
    
    mask2d = mask.reshape(I*J, order = 'F')

    # Calculate the occurrence of flash drought
    print('Identifying flash drought')
    fd = np.ones((T, I, J)) * np.nan
    
    fd2d = fd.reshape(T, I*J, order = 'F')
    
    change_criterion = -2
    drought_criterion = -1.28
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]

    for ij in range(I*J):
        if mask2d[ij] == 0:
            continue
        else:
            pass
        
        start_date = dates[-1]
        for t in range(T-1):
            
            # If the monthly change in SPEI is below the required change, and SPEI is below the drought threshold, FD occurs
            # Note, since the changes are calculated over a 1 month period, the first criterion in Noguera et al. is automatically satisified
            if (delta_spei2d[t,ij] <= change_criterion) & (spei2d[t,ij] <= drought_criterion): 
                fd2d[t,ij] = 1
            else:
                fd2d[t,ij] = 0
                
    # Restore the FD back into a 3D array
    fd = fd2d.reshape(T, I, J, order = 'F')
    print('Done')
    
    return fd



#%%
##############################################

# Calcualte flash droughts using a FD identification method from Pendergrass et al. 2020
# This method uses EDDI to identify FD

def pendergrass_fd(eddi, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using the method described in Pendergrass et al. 2020 (https://doi.org/10.1038/s41558-020-0709-0). 
    This method uses the evaporative demand drought index (EDDI) to 
    identify flash drought.
    
    Inputs:
    :param eddi: Input EDDI values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in eddi
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    # Initialize some variables
    T, I, J = eddi.shape
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    
    fd = np.ones((T, I, J)) * np.nan

    eddi2d = eddi.reshape(T, I*J, order = 'F')
    fd2d = fd.reshape(T, I*J, order = 'F')
    mask2d = mask.reshape(I*J, order = 'F')

    print('Identifying flash drought')
    for ij in range(I*J):
        if mask2d[ij] == 0:
            continue
        else:
            pass
        
        # The criteria are EDDI must be 50% greater than EDDI 2 weeks (3 pentads) ago, or a 50 percentile increase in 2 weeks, and remain that intense for another 2 weeks.
        for t in range(3, T-3): 
            
            ind = np.where( (dates[t].month == months[climo_index]) & (dates[t].day == days[climo_index]) )[0]
            
            current_percent = stats.percentileofscore(eddi2d[ind,ij], eddi2d[t,ij])
            previous_percent = stats.percentileofscore(eddi2d[ind,ij], eddi2d[t-3,ij])
            
            # Note this checks for all pentads in the + 2 week period, so there cannot be moderation
            if ( (current_percent - previous_percent) > 50 ) & (eddi2d[t+1,ij] >= eddi2d[t,ij]) & (eddi2d[t+2,ij] >= eddi2d[t,ij]) & (eddi2d[t+3,ij] >= eddi2d[t,ij]): 
                fd2d[t,ij] = 1
            else:
                fd2d[t,ij] = 0
                
    fd = fd2d.reshape(T, I, J, order = 'F')
    print('Done')
    
    # print(np.nanmin(fd), np.nanmax(fd), np.nanmean(fd))
    
    return fd


#%%
##############################################

# Calcualte flash droughts using a FD identification method from Li et al. 2020
# This method uses SEDI to identify FD

def li_fd(sedi, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using the method described in Li et al. 2020 (https://doi.org/10.1016/j.catena.2020.104763). 
    This method uses the standardized evaporative demand index (SEDI) to 
    identify flash drought.
    
    Inputs:
    :param sedi: Input SEDI values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in sedi
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    ##### FILL THIS AND ADD THE METHOD
    
    T, I, J = sedi.shape
    
    fd = np.ones((T, I, J))
    
    return fd


#%%
##############################################

# Calcualte flash droughts using a FD identification method from Liu et al. 2020
# This method uses soil moisture to identify FD

def liu_fd(vsm, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using the method described in Liu et al. 2020 (https://doi.org/10.1175/JHM-D-19-0088.1). 
    This method uses the volumetric soil moisture (VSM; 0 - 40 cm average) to 
    identify flash drought.
    
    Inputs:
    :param vsm: Input VSM values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in vsm
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
        
        

    # First, determine the soil moisture percentiles
    print('Calculating soil moisture percentiles')
    T, I, J = vsm.shape
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    
    sm_percentiles = np.ones((T, I, J)) * np.nan
    
    sm2d = vsm.reshape(T, I*J, order = 'F')
    mask2d = mask.reshape(I*J, order = 'F')
    sm_percentiles2d = sm_percentiles.reshape(T, I*J, order = 'F')
    
    for t in range(T):
        ind = np.where( (dates[t].day == days[climo_index]) & (dates[t].month == months[climo_index]) )[0]
        
        for ij in range(I*J):
            sm_percentiles2d[t,ij] = stats.percentileofscore(sm2d[ind,ij], sm2d[t,ij])
        
        
    # Begin drought identification process
    print('Identifying flash droughts')
    fd = np.ones((T, I, J)) * np.nan
    
    fd2d = fd.reshape(T, I*J, order = 'F')
    
    # Initialize up a variable to look up to 12 pentads ahead (from Otkin et al. 2021, that rapid intensification goes up to 10 pentads ahead); 12 ensures data after intensification is included
    fut_pentads = np.arange(0, 13)
    fp = len(fut_pentads)

    
    for ij in range(I*J):
        if (ij%1000) == 0:
            print('%d/%d'%(int(ij/1000), int(I*J/1000)))
        
        if mask2d[ij] == 0:
            continue
        else:
            pass
        
        for t in range(T-12): # Exclude the last few months in the dataset for simplicity since FD identification involves looking up to 12 pentads ahead
            # First determine if the soil moisture is below the 40 percentile
            # print(ij)
            # print(sm_percentiles2d[t,ij] <= 40)
            if sm_percentiles2d[t,ij] <= 40:
                
                
                R2 = np.ones((fp)) * np.nan
                ri_entries = np.ones((fp)) * np.nan
                
                # To determine when the percentiles level out (when the intensification ends), regress SM percentiles with pentads with increasing polynomial degrees until R^2 > 0.95 or until a 10th order polynomial is used (assumed accuracy is being lost here)
                for p in range(1, 11):
                    sm_est, R2p = polynomial_regress(fut_pentads, sm_percentiles2d[t:t+fp,ij], order = p)
                    
                    R2[p-1] = R2p
                    if (R2[p-1] >= 0.95):
                        order = p
                        break
                    elif (p >= 10):
                        # Find the maximum R2
                        ind = np.where(R2 == np.nanmax(R2))[0]
                        if len(ind) < 1: # If no maximum is found, the calculations are all NaNs and nothing can be determined
                            fd2d[t,ij] = 0
                            break
                        
                        order = ind[0]+1
                        
                        # Get the SM estimates for the polynomial regression with the highest R2
                        sm_est, R2p = polynomial_regress(fut_pentads, sm_percentiles2d[t:t+fp,ij], order = order)
                        break
                    else:
                        pass
                    
                # Next, determine where the change in sm_est is approximately 0 (within 0.01) to find when the rapid intensification ends
                for pent in fut_pentads[1:]:
                    ri_entries[pent-1] = (sm_percentiles2d[t+pent,ij] - sm_percentiles2d[t,ij])/pent # pent here is the difference between the current pentad and how many pentads ahead one is looking
                    
                    if (sm_est[pent] - sm_est[pent-1]) < 0.1:
                        ri_end = pent
                        break
                    elif pent == 12:
                        ri_end = pent
                        break
                    else:
                        pass
                    
                ri_mean = np.nanmean(ri_entries)
                ri_max  = np.nanmax(ri_entries)
                
                # Lastly, to identify FD, two criteria are required. At the peak of the drought (this is approximately when Delta sm_percentiles = 0 since there is no more intensification), sm_percentiles < 20,
                # and the Rapid Intensification component must be: ri_mean >= 6.5 percentiles/week (about 5 percentiles/pentad) or ri_max >= 10 percentiles/week (about 7.5 percentiles/pentad)
                
                # Note also that the FD is being identified for the end of RI period
                if (sm_percentiles2d[t+ri_end,ij] <= 20) & ( (ri_mean >= 5) | (ri_max >= 7.5) ):
                    fd2d[t+ri_end,ij] = 1
                else:
                    fd2d[t+ri_end,ij] = 0
                    
                # Increment t to the end of the intensification period
                t = t + ri_end
                
            else:
                fd2d[t,ij] = 0
            #     continue
    
    
    fd = fd2d.reshape(T, I, J, order = 'F')    
    
    print('Done')
    
    return fd



# Create a function for polynomial regression
def polynomial_regress(x, y, order = 1):
    '''
    A function designed to take in two vectors of equal size (X and Y) and perform a polynomial
    regression of the data. Function outputs the estimated data yhat, and the R^2 coefficient.
    
    Inputs:
    :param x: The input x data
    :param y: The input y data that is being estimated
    :param order: The order of the polynomial used in the regression
    
    Outputs:
    :param yhat: The estimated value of y using the polynomial regression
    :param R2: The R^2 coefficient from the polynomial regression
    '''
    
    # Determine the size of the data
    T = len(x)
    
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
    xhat = np.dot(np.dot(invEtE, E.T), y)
    
    # Estimate yhat
    yhat = E.dot(xhat)
    
    # Determine the R^2 coefficient
    R2 = np.nanvar(yhat - np.nanmean(y))/np.nanvar(y)
    
    return yhat, R2



#%%
##############################################

# Calcualte flash droughts using a FD identification method from Otkin et al. 2021
# This method uses FDII to identify FD

def otkin_fd(fdii):
    '''
    Calculate the flash drought using the method described in Otkin et al. 2021 (https://doi.org/10.3390/atmos12060741). 
    This method uses the flash drought intensity index (FDII) to 
    identify flash drought.
    
    Inputs:
    :param fdii: Input FDII values, time x lat x lon format
    
    Outputs:
    :param fd: The identified flash drought for all grid points and time steps in fdii
    '''
    
    # This is straightforward as the method is contained in the calculation of FDII. That is, if FDII = 0, no FD, and FDII > 0, there is FD.
    print('Identifying flash droughts')
    fd = fdii
    
    # Turn values > 0 to 1
    fd[fd > 0] = 1
    fd[fd <= 0] = 0
    print('Done')
    
    return fd


#%%
##############################################

# argument parser
def create_fd_calculator_parser():
    '''
    Create argument parser
    '''
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='FD Calculations', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    
    parser.add_argument('--dataset', type=str, default='/Users/stuartedris/desktop/PhD_Research_ML_and_FD/ML_and_FD_in_NARR/Data', help='Data set directory')

    # Flash Drought Indices
    parser.add_argument('--check_index', action='store_true', help='Check the some of the FD indices if they exist and calculate them if not')
    parser.add_argument('--index', nargs='+', type=str, default=['sesr','spei'], help='Name of the indices being checked')
    parser.add_argument('--hist', action='store_true', help='Plot a histogram of the checked indices')
    parser.add_argument('--test_map', action='store_true', help='Plot a test map of the checked indices')
    parser.add_argument('--max_map', action='store_true', help='Make a map of the maximum value of the checked indices')
    
    # Flash Drought Calculations
    parser.add_argument('--methods', nargs='+', type=str, default=['christian','otkin'], help='Name of the FD idenification methods to use')
    parser.add_argument('--fd_climatology', action='store_true', help='Create a climatology map of FD with the methods used')
    
    parser.add_argument('--make_labels', action='store_true', help='Make the label dataset and split it into folds for future ML')
    
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--model', type=str, default='narr', help='Reanalysis model the dataset(s) came from')
    parser.add_argument('--mask', action='store_true', help='Load land-sea mask data')
    
    parser.add_argument('--start_date', type=str, default='1990-01-01', help='Start date for the climatology period in %Y-%m-%d format')
    parser.add_argument('--end_date', type=str, default='2020-12-31', help='End date for the climatology period in %Y-%m-%d format')

    
    return parser


#%%
##############################################

if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_fd_calculator_parser()
    args = parser.parse_args()
    
    # Get the directory of the dataset
    dataset_dir = '%s/%s'%(args.dataset, args.model)
    
    # Turn the start and end dates into datetimes
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Load the land-sea mask?
    if args.mask:
        mask = load_mask(model = args.model)
    
    #####################################
    ##### Calculate indices if they don't exist
    #####
    print('Checking indices')
    if args.check_index:
        for index in args.index:
            if os.path.exists('%s/Indices/%s.%s.pentad.nc'%(dataset_dir, index, args.model)):
                # Processed file does exist: load it
                print("File %s.%s.pentad.nc already exists"%(index, args.model))
                
                index_data = load_nc(index, '%s.%s.pentad.nc'%(index,args.model), path = '%s/Indices/'%dataset_dir)
                lat = index_data['lat']; lon = index_data['lon']; dates = index_data['ymd']
                
                
            else:
                # If the index file does not exist, calculate it
                
                # Load the required variable
                if index == 'sesr':
                    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = et['lat']; lon = et['lon']; dates = et['ymd']
                    
                    # Calculate the index
                    index_data = calculate_sesr(et['evap'], pet['pevap'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del et, pet
                    
                elif index == 'sedi':
                    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = et['lat']; lon = et['lon']; dates = et['ymd']
                    
                    # Calculate the index
                    index_data = calculate_sedi(et['evap'], pet['pevap'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del et, pet
                
                elif index == 'spei':
                    p = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = p['lat']; lon = p['lon']; dates = p['ymd']
                    
                    # Calculate the index
                    index_data = calculate_spei(p['precip'], pet['pevap'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del p, pet
                
                elif index == 'sapei':
                    p = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = p['lat']; lon = p['lon']; dates = p['ymd']
                    
                    # Calculate the index
                    index_data = calculate_sapei(p['precip'], pet['pevap'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del p, pet
                
                elif index == 'eddi':
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = pet['lat']; lon = pet['lon']; dates = pet['ymd']
                    
                    # Calculate the index
                    index_data = calculate_eddi(pet['pevap'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del pet
                
                elif index == 'smi':
                    sm0 = load_nc('soilm', 'soil_moisture.0cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    sm10 = load_nc('soilm', 'soil_moisture.10cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    sm40 = load_nc('soilm', 'soil_moisture.40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    lat = sm0['lat']; lon = sm0['lon']; dates = sm0['ymd']
                    
                    # Calculate the index
                    index_data = caculate_smi([sm0['soilm'], sm10['soilm'], sm40['soilm']], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del sm0, sm10, sm40
                
                elif index == 'sodi':
                    p = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    sm = load_nc('soilm', 'soil_moisture.0-40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    ro = load_nc('ro', 'runoff.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = p['lat']; lon = p['lon']; dates = p['ymd']
                    
                    # Calculate the index
                    index_data = calculate_sodi(p['precip'], et['evap'], pet['pevap'], sm['soilm'], ro['ro'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del p, et, pet, sm, ro
                
                elif index == 'fdii':
                    sm = load_nc('soilm', 'soil_moisture.0-40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    lat = sm['lat']; lon = sm['lon']; dates = sm['ymd']
                    
                    # Calcualte the index
                    fdii, fd_int, dro_sev = calculate_fdii(sm['soilm'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    index_data = [fdii, fd_int, dro_sev]
                    index_names = ['fdii', 'fd_int', 'dro_sev']
                    
                    # Remove the no longer needed variables
                    del sm
                    
                # Write the index data so that it is available for future use
                if index != 'fdii':
                    write_nc(index_data, lat, lon, dates, filename = '%s.%s.pentad.nc'%(index,args.model), VarSName = index, path = '%s/Indices/'%dataset_dir)
                else: # Special case for FDII, which gives 3 variabels to save
                    for datum, name in zip(index_data, index_names):
                        write_nc(datum, lat, lon, dates, filename = '%s.%s.pentad.nc'%(name,args.model), VarSName = index, path = '%s/Indices/'%dataset_dir)
                
                
                gc.collect() # Clears deleted variables from memory
                
                
                # Display a histogram of the index?
                if args.hist:
                    if index != 'fdii':
                        display_histogram(index_data, index, path = dataset_dir)
                    else:
                        for datum, name in zip(index_data, index_names):
                            display_histogram(datum, name, path = dataset_dir)
                    
                # Display a test map of the index?
                if args.test_map:
                    if index != 'fdii':
                        test_map(index_data, lat, lon, dates, index)
                    else:
                        for datum, name in zip(index_data, index_names):
                            test_map(datum, lat, lon, dates, name)
                    
                # Display a map of maximum values?
                if args.max_map:
                    if index != 'fdii':
                        display_maximum_map(index_data, lat, lon, dates, datetime(2012, 5, 1), datetime(2012, 8, 1), index, path = dataset_dir)
                    else:
                        for datum, name in zip(index_data, index_names):
                            display_maximum_map(datum, lat, lon, dates, datetime(2012, 5, 1), datetime(2012, 8, 1), name, path = dataset_dir)
            
            
        
    #####################################
    ##### Calculate flash drought if file don't exist
    #####
    print('Calculating flash drought with designated methods')
    for method in args.methods:
        if os.path.exists('%s/FD_Data/%s.%s.pentad.nc'%(dataset_dir, method, args.model)):
            # Processed file does exist: skip it
            print("File %s.%s.pentad.nc already exists"%(method, args.model))
            
            fd = load_nc('fd', '%s.%s.pentad.nc'%(method, args.model), path = '%s/FD_Data/'%dataset_dir)
            lat = fd['lat']; lon = fd['lon']; dates = fd['ymd']
            fd = fd['fd']
            
        # If the FD file does not exist, calculate FD
        else:
            if method == 'christian':
                # Collect the required index
                index = load_nc('sesr', 'sesr.%s.pentad.nc'%(args.model), path = '%s/Indices/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = christian_fd(index['sesr'], mask, dates, start_year = start_date.year, end_year = end_date.year)
                
            elif method == 'nogeura':
                # Collect the required index
                index = load_nc('spei', 'spei.%s.pentad.nc'%(args.model), path = '%s/Indices/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = nogeura_fd(index['spei'], mask, dates, start_year = start_date.year, end_year = end_date.year)
            
            elif method == 'pendergrass':
                # Collect the required index
                index = load_nc('eddi', 'eddi.%s.pentad.nc'%(args.model), path = '%s/Indices/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = pendergrass_fd(index['eddi'], mask, dates, start_year = start_date.year, end_year = end_date.year)
            
            elif method == 'liu':
                # Collect the required index
                index = load_nc('soilm', 'soil_moisture.0-40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = liu_fd(index['soilm'], mask, dates, start_year = start_date.year, end_year = end_date.year)
            
            elif method == 'li':
                # Collect the required index
                index = load_nc('sedi', 'sedi.%s.pentad.nc'%(args.model), path = '%s/Indices/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = li_fd(index['sedi'], mask, dates, start_year = start_date.year, end_year = end_date.year)
            
            elif method == 'otkin':
                # Collect the required index
                index = load_nc('fdii', 'fdii.%s.pentad.nc'%(args.model), path = '%s/Indices/'%dataset_dir)
                lat = index['lat']; lon = index['lon']; dates = index['ymd']
                
                # Calculate the FD
                fd = otkin_fd(index['fdii'])
                
            # Remove the unnecessary data
            del index
            gc.collect() # Clears deleted variables from memory
            
            # Write the FD data so that it is available for future use
            write_nc(fd, lat, lon, dates, filename = '%s.%s.pentad.nc'%(method,args.model), VarSName = 'fd', path = '%s/FD_Data/'%dataset_dir)
            
        # Make a FD climatology map?
        if args.fd_climatology:
            display_fd_climatology(fd, lat, lon, dates, method, model = args.model, path = dataset_dir)
            
    
    # Create label data?
    if args.make_labels:
        print('Constructing label data')
        
        label_fname = 'fd_output_labels.pkl'
        
        # Check if output processed file already exists
        if os.path.exists('%s/%s'%(dataset_dir, label_fname)):
            # Processed file does exist: exit
            print("File %s already exists"%label_fname)
        
        # Load the data that will make up the label data
        ch_fd = load_nc('fd', 'christian.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        nog_fd = load_nc('fd', 'nogeura.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        pen_fd = load_nc('fd', 'pendergrass.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        liu_fd = load_nc('fd', 'liu.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        # li_fd = load_nc('fd', 'li.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        ot_fd = load_nc('fd', 'otkin.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        
        print(np.nanmin(ch_fd['fd']), np.nanmax(ch_fd['fd']))
        print(np.nanmin(nog_fd['fd']), np.nanmax(nog_fd['fd']))
        print(np.nanmin(pen_fd['fd']), np.nanmax(pen_fd['fd']))
        print(np.nanmin(liu_fd['fd']), np.nanmax(liu_fd['fd']))
        print(np.nanmin(ot_fd['fd']), np.nanmax(ot_fd['fd']))
        
        # Parse and save the label data into a pickle file
        parse_data([ch_fd['fd'], nog_fd['fd'], pen_fd['fd'], liu_fd['fd'], ot_fd['fd']], dates, dataset_dir, label_fname)
        
            
            


