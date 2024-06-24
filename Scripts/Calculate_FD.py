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
- Otkin et al. 2021 (for FDII method): https://doi.org/10.3390/atmos12060741


TODO:
- Modify map creation functions to include the whole world
    - display_fd_climatology
- remove sea points for era5
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
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
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
def display_fd_climatology(fd, lat, lon, dates, mask, methods, model = 'narr', globe = False, grow_season = False, path = './'):
    '''
    Display different types of FD climatology including frequency (percentage of years with flash drought), number of flash droughts in the dataset,
    Average percentage of time spent in flash drought, average duration of flash drought, most common onset month, 
    and average flash drought coveraged/onset month time series and barplot (all this done for multiple flash drought identification methods).
    
    Inputs:
    :param fd: List of input flash drought (FD) data to be plotted. Each entry in the list is for an FD identification method, and data inside each entry is time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param mask: Land-sea mask for the dataset (1 = land, 0 = sea)
    :param dates: Array of datetimes corresponding to the timestamps in FD
    :param methods: List of FD identification methods used to calculate the FD
    :param model: String describing what reanalysis model the data comes from. Used to name the figure
    :param globe: Boolean indicating whether the data is global
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param path: Path the figures will be saved to

    Outputs:
    A set of a figures for each type of climatology and for each method will be made and saved
    '''

    for m, method in enumerate(methods):
        # Calculate and plot FD frequency climatology
        filename = '%s_%s_flash_drought_frequency.png'%(model, method)
        cbar_label = '% of years with Flash Drought'
        
        fd_climo = calculate_climatology_frequency(fd[m], lat, dates, grow_season = grow_season)
        display_climatology_map(fd_climo*100, lat, lon, title = method, cbar_label = cbar_label, globe = globe, 
                                cmin = -20, cmax = 80, cint = 1, cticks = np.arange(0, 90, 10), new_colorbar = True, path = path, savename = filename)

        # Calculate and plot total number of FDs climatology
        filename = '%s_%s_flash_drought_number.png'%(model, method)
        cbar_label = '# of Flash Droughts'
        
        fd_climo = calculate_climatology_number(fd[m], lat, dates, grow_season = grow_season)
        display_climatology_map(fd_climo, lat, lon, title = method, cbar_label = cbar_label, globe = globe, 
                                cmin = -10, cmax = 50, cint = 1, cticks = np.arange(0, 50+1, 10), new_colorbar = True, path = path, savename = filename)

        # Calculate and plot the average time in FD climatology
        filename = '%s_%s_flash_drought_time_in_drought.png'%(model, method)
        cbar_label = '% Time Spent in Flash Drought'
        
        fd_climo = calculate_time_in_fd_climatology(fd[m], lat, dates, grow_season = grow_season)
        display_climatology_map(fd_climo*100, lat, lon, title = method, cbar_label = cbar_label, globe = globe, 
                                cmin = -0.5, cmax = 5, cint = 0.5, cticks = np.arange(0, 5+1, 1), new_colorbar = False, path = path, savename = filename)

        # Calculate and plot the average FD duration climatology
        filename = '%s_%s_flash_drought_duration.png'%(model, method)
        cbar_label = 'Average Duration of Flash Drought (days)'
        
        fd_climo = calculate_duration_climatology(fd[m], lat, dates, grow_season = grow_season)
        display_climatology_map(fd_climo, lat, lon, title = method, cbar_label = cbar_label, globe = globe, 
                                cmin = 30, cmax = 40, cint = 1, cticks = np.arange(30, 40+1, 1), new_colorbar = False, path = path, savename = filename)

        # Calculate and plot the average onset time climatology
        filename = '%s_%s_flash_drought_onset_month.png'%(model, method)
        cbar_label = 'Most Common Onset Month for Flash Drought'
        
        fd_climo = calculate_average_onset_time_climatology(fd[m], lat, dates, grow_season = grow_season)
        display_climatology_map(fd_climo, lat, lon, title = method, cbar_label = cbar_label, globe = globe, 
                                cmin = 0, cmax = 12, cint = 1, cticks = np.arange(0, 12, 1), new_colorbar = False, cbar_months = True, 
                                path = path, savename = filename)

        # Give a time series plot of the climatology
        fd_coverage_time_series(fd[m], dates, mask, lat, grow_season = grow_season, title = method,
                                path = path, savename_ts = '%s_%s_time_series_climatology.png'%(model, method), 
                                savename_bar = '%s_%s_barplot_climatology.png'%(model, method))


# Function for the FD climatology time series
def fd_coverage_time_series(fd, dates, mask, lat, grow_season = False, title = 'tmp', years = None, months = None, days = None, 
                            path = './', savename_ts = 'tmp.png', savename_bar = 'tmp.png'):
    '''
    Create a time series showing the average FD coverage in a year, and a bar plot showing the average the how many grid points experience FD onset in each month

    Inputs:
    :param fd: FD data to be plotted. time x lat x lon format
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param mask: Land-sea mask for the dataset (1 = land, 0 = sea)
    :param lat: Gridded latitude values corresponding to data
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param path: Path the figures will be saved to
    :param savename_ts: Filename the time series plot will be saved to
    :param savename_bar: Filename the barplot will be saved to

    Outputs:
    Two plots of a time series and bar plot will be created and saved
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])


    # Calculate the average number of rapid intensifications and flash droughts in a year
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    # Get the data size
    T, I, J = fd_grow.shape
    
    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    dates_grow = dates[ind]
    years_grow = years[ind]
    months_grow = months[ind]
    days_grow = days[ind]
    
    # Isolate datetimes for a single year
    ind = np.where(years_grow == 2001)[0]
    one_year = dates_grow[ind]
    year_months = np.array([date.month for date in one_year])
    
    # Determine the time series
    fd_ts = np.nansum(fd_grow.reshape(T, I*J), axis = -1)
    I_m, J_m = mask.shape # The mask shape may differ from the actual data for the global dataset
    mask_ts = np.nansum(mask.reshape(I_m*J_m))
    
    # Calculate the average and standard deviation of FD coverage for each pentad in a year
    fd_mean = []
    fd_std = []
    fd_sum = []
    for date in one_year:
        y_ind = np.where((date.month == months_grow) & (date.day == days_grow))[0]
        
        fd_mean.append(np.nanmean(fd_ts[y_ind]))
        fd_std.append(np.nanstd(fd_ts[y_ind]))
        fd_sum.append(np.nansum(fd_ts[y_ind]))
    
    fd_mean = np.array(fd_mean)
    fd_std = np.array(fd_std)
    fd_sum = np.array(fd_sum)
    
    # Date format for the tick labels
    DateFMT = DateFormatter('%b')
    
    # Plot the FD coverage
    fig = plt.figure(figsize = [18, 10])
    ax = fig.add_subplot(1,1,1)
    
    # Plot the data in a time series
    ax.plot(dates_grow[ind], fd_mean/mask_ts*100, 'b')
    #ax.fill_between(dates[ind], 100*(fd_mean/mask_ts-fd_std/mask_ts), 100*(fd_mean/mask_ts+fd_std/mask_ts), alpha = 0.5, edgecolor = 'b', facecolor = 'b')
    
    # Set the title
    ax.set_title(title, fontsize = 18)
    
    # Set the axis labels
    ax.set_xlabel('Time', size = 18)
    ax.set_ylabel('Annual Average Areal Coverage (%)', size = 18)
    
    # Set the ticks
    ax.set_ylim([0, 4.5])
    ax.xaxis.set_major_formatter(DateFMT)

    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename_ts), bbox_inches = 'tight')
    plt.show(block = False)
    
    # Determine the average number of grids for each month
    months_unique = np.unique(months_grow)
    fd_month_mean = []
    fd_month_std = []
    for month in months_unique:
        y_ind = np.where(month == year_months)[0]
        fd_month_mean.append(np.nansum(fd_sum[y_ind])/np.nansum(fd_sum)*100)
        fd_month_std.append(np.nanmean(fd_std[y_ind]))

    # Obtain the labels for each month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names_grow = []
    for m, month in enumerate(months_unique):
        month_names_grow.append(month_names[int(month-1)])
    
    
    # Bar plot of average number of FD in each month
    fig = plt.figure(figsize = [18, 10])
    ax = fig.add_subplot(1,1,1)
    
    # Make the bar plot
    ax.bar(months_unique, fd_month_mean, width = 0.8, edgecolor = 'k')#, yerr = fd_month_std)
    
    # Set the title
    ax.set_title(title, fontsize = 18)
    
    # Set the axis labels
    ax.set_xlabel('Time', size = 18)
    ax.set_ylabel('Percentage of FD Occurance', size = 18)
    
    # Set the ticks
    ax.set_xticks(months_unique, month_names_grow)
    #ax.xaxis.set_major_formatter(DateFMT)

    for i in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        i.set_size(18)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename_bar), bbox_inches = 'tight')
    plt.show(block = False)

# Function for the FD frequency climatology
def calculate_climatology_frequency(fd, lat, dates,  grow_season = False, years = None, months = None):
    '''
    Calculate the frequency climatology of flash droughts (percentage of years that experienced flash drought)
    
    Inputs:
    :param fd: flash drought data to be plotted. time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    :param per_ann_fd: Map containing the number of years that experienced flash drought
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
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    years_grow = years[ind]
        
    for y in range(all_years.size):
        y_ind = np.where( (all_years[y] == years_grow) )[0]
        
        # Calculate the mean number of flash drought for each year    
        ann_fd[y,:,:] = np.nanmean(fd_grow[y_ind,:,:], axis = 0)
        
        # Turn nonzero values to 1 (each year gets 1 count to the total)    
        ann_fd[y,:,:] = np.where(( (ann_fd[y,:,:] == 0) | (np.isnan(ann_fd[y,:,:])) ), 
                                 ann_fd[y,:,:], 1) # This changes nonzero  and nan (sea) values to 1.
    

    

    # Calculate the percentage number of years with rapid intensifications and flash droughts
    per_ann_fd = np.nansum(ann_fd[:,:,:], axis = 0)/all_years.size
    
    # Turn 0 values into nan
    per_ann_fd = np.where(per_ann_fd != 0, per_ann_fd, np.nan)

    return per_ann_fd

# Function for the FD number climatology
def calculate_climatology_number(fd, lat, dates, grow_season = False, years = None, months = None):
    '''
    Calculate the number of flash droughts experienced in a dataset
    
    Inputs:
    :param fd: flash drought data to be plotted. time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    :param fd_number: Map containing the number of flash drought for each grid point
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

    # Calculate the average number of rapid intensifications and flash droughts in a year
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    dates_grow = dates[ind]
    
    # Calculate the number of flash droughts in each year
    fd_number = np.zeros((I, J)) * np.nan

    for i in range(I):
        for j in range(J):
            # Identify the duration of each FD (works to count all unique FDs as well)
            start_times, end_times = identify_drought_by_time_series(fd_grow[:,i,j], dates, min_time = 0)
            # Count the FDs
            n_fd = 0
            if np.invert(len(start_times) < 1):
                for start, end in zip(start_times, end_times):
                    n_fd = n_fd + 1

                # Total number of FD (goes to 50) or average number of droughts per year (goes up to 3)
                #per_ann_fd[i,j] = n_fd/len(all_years)
                fd_number[i,j] = n_fd
            else:
               fd_number[i,j] = 0
    
    
    # Turn 0 values into nan
    fd_number = np.where(fd_number != 0, fd_number, np.nan)

    return fd_number

# Function for the timei n FD climatology
def calculate_time_in_fd_climatology(fd, lat, dates, grow_season = False, years = None, months = None):
    '''
    Calculate the average percentage of time spend in flash drought

    Inputs:
    :param fd: flash drought data to be plotted. time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    :param fd_time: Map containing the percentage of time spend in flash drought for each grid point
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

    # Calculate the average number of rapid intensifications and flash droughts in a year
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    years_grow = years[ind]
    
    # Calculate the percentage of time in FD (or the average number of pentads that experienced FD)
    fd_time = np.nansum(fd_grow, axis = 0)/T
    
    # Turn 0 values into nan
    fd_time = np.where(fd_time != 0, fd_time, np.nan)

    return fd_time

# Function for the FD duration climatology
def calculate_duration_climatology(fd, lat, dates, grow_season = False, years = None, months = None):
    '''
    Calculate the average duration of flash droughts

    Inputs:
    :param fd: flash drought data to be plotted. time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    :param fd_duration: Map containing the average duration of flash drought for each grid point
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

    # Calculate the average number of rapid intensifications and flash droughts in a year
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    dates_grow = dates[ind]
    
    # Calculate the average duration of flash droughts
    fd_duration = np.zeros((I, J)) * np.nan
    
    for i in range(I):
        for j in range(J):
            # Determine the start and end time of each flash drought for a grid point
            start_times, end_times = identify_drought_by_time_series(fd_grow[:,i,j], dates_grow, min_time = 0)
            duration = []
            # Determine the duration of the flash droughts (include the intensification period - add 30 days)
            if np.invert(len(start_times) < 1):
                for start, end in zip(start_times, end_times):
                    duration.append((end - start).days + 30)

                # Determine the mean of the durations in the time series
                fd_duration[i,j] = np.nanmean(duration)
            else:
                fd_duration[i,j] = 0
    
    # Turn 0 values into nan
    fd_duration = np.where(fd_duration != 0, fd_duration, np.nan)

    return fd_duration

# Function for the average onset time climatology
def calculate_average_onset_time_climatology(fd, lat, dates, grow_season = False, years = None, months = None):
    '''
    Calculate the average onset month of flash droughts

    Inputs:
    :param fd: flash drought data to be plotted. time x lat x lon format
    :param lat: Gridded latitude values corresponding to data
    :param dates: Array of datetimes corresponding to the timestamps in fd
    :param grow_season: Boolean indicating whether fd has already been set into growing seasons
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates

    Outputs:
    :param onset_map: Map containing the average onset month (1 = Jan, 2 = Feb, ...) flash drought for each grid point
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])


    # Initialize variables
    all_years = np.unique(years)

    # Calculate the average number of rapid intensifications and flash droughts in a year
    if np.invert(grow_season):
        fd_grow = collect_grow_seasons(fd, dates, lat[:,0])
    else:
        fd_grow = fd

    T, I, J = fd_grow.shape
    
    # Reduce years to the size of the growing season (should be the same for both hemispheres)
    ind = np.where( (months >= 4) & (months <= 10) )[0]
    months_grow = months[ind]

    fd_months = np.zeros((T, I, J))
    # Set each flash drought identifier to the month it occurs in
    for month in np.unique(months_grow):
        mon_ind = np.where(month == months_grow)[0]
        fd_months[mon_ind,:,:] = np.where(fd_grow[mon_ind,:,:] == 1, month, fd_months[mon_ind,:,:])
    
    # Set 0 values to NaN so they are not counted
    fd_months[fd_months == 0] = np.nan

    # Calculate the most common onset month
    onset_map = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            ind = []
            for month in np.unique(months_grow):
                # Determine the length (number) of pentads with FD onset in a given month
                ind_month = np.where(fd_months[:,i,j] == month)[0]
                ind.append(len(ind_month))

            # The longest length is the pentad with the most common onset month
            most_common = np.where(ind == np.nanmax(ind))[0]
            onset_map[i,j] = np.unique(months_grow)[most_common[0]]
            # Mode determines the most common value (month) in a vector (time series)
            #onset_map[i,j] = stats.mode(fd_months[:,i,j], nan_policy = 'omit')[0][0]
    
    # Turn 0 values into nan
    onset_map = np.where(onset_map == 0, np.nan, onset_map)

    return onset_map

# Function to identify unique droughts in a time series and deliver the start and end dates
def identify_drought_by_time_series(index, dates, min_time = 1):
    '''
    Identify drought by a patch of contiuous grid points that falls below a threshold.
    Returns two lists of datetimes indicating when droughts started in the grid, and when they ended
    
    NOTE: This method is NOT perfect; it assumes droughts only last for 1 summer (due to time stamps skipping Nov. - Mar.)
    and droughts that start in different regions of the grid may get skipped over for the earlier drought in the year.
    
    Inputs:
    :param index: The index used to find the droughts; time x lat x lon shape
    :param dates: An array of datetimes corresponding to each time stamp in index
    :param min_time: The minimum number of days that need to be below the threshold to be recorded as drought
    
    Outputs:
    :param start_dates: List of datetimes corresponding to the beginning times of droughts
    :param end_dates: List of datetimes corresponding to the ending times of droughts
    '''
    
    # Identify the droughts
    droughts = np.where(index == 1, 1, 0)

    start_time = []
    end_time = []

    # drought_patch is the minimum area/# of grids that must have drought to register an event
    # So the sum of drought grid points must be the total area/number of grid points (drought_patch x drought_patch)
    times = np.where(droughts == 1)[0]

    # Start looking for times when droughts start/end
    if len(times) >= 2:
        start = times[0]
        end = times[-1]
        start_time.append(start)
        for t in range(len(times)):
            ## Exclude start and end points
            if np.invert(times[t] == end) & np.invert(times[t] == start):
                # If condition for middle of drought
                if (times[t-1] == (times[t] - 1)) & (times[t+1] == (times[t] + 1)):
                    pass
                # If condition for only 1 point with drought
                elif np.invert(times[t-1] == (times[t] - 1)) & np.invert(times[t+1] == (times[t] + 1)):
                    start_time.append(times[t])
                    end_time.append(times[t])
                # If conditions for end of drought
                elif (times[t-1] == (times[t] - 1)) & np.invert(times[t+1] == (times[t] + 1)):
                    end_time.append(times[t])
                # If condition for start of drought
                elif np.invert(times[t-1] == (times[t] - 1)) & (times[t+1] == (times[t] + 1)):
                    start_time.append(times[t])
            
            # If the start or end point is considered
            else:
                # Embedded blocks are to avoid indexing errors (using t+1 or t-1 causes an indexing error at t=0 and t=-1
                
                # Start point
                if (times[t] == start):
                    if np.invert(times[t+1] == (times[t]+1)):
                        end_time.append(times[t])
                        
                # End time
                if (times[t] == end):
                    if np.invert(times[t-1] == (times[t] - 1)):
                        start_time.append(times[t])

            end_time.append(end)

    # If there is only 1 drought found
    elif len(times) == 1:
        start_dates = dates[times]
        end_dates = dates[times]
        
        return start_dates, end_dates
        

    # Loops may find identical start/end times; focus only on unique values
    start_time = np.unique(start_time)
    end_time = np.unique(end_time)
        
    # Start and end times should now have the same length, and start and end times should align        
    
    # Get the datetimes if there are any
    if len(start_time) < 1:
        start_dates = np.array([])
        end_dates = np.array([])
    else:
        if start_time[-1] == dates.size:
            start_time = start_time[:-1]
            end_time = end_time[:-1]
        
        start_days = dates[start_time[:]]
        end_days = dates[end_time[:]]

        # If the difference between start and end times does not pass the minimum time, remove them
        min_time = timedelta(days = min_time)
        start_dates = []
        end_dates = []
        for n in range(len(start_days)):
            difference = end_days[n] - start_days[n]
            if difference >= min_time:
                start_dates.append(start_days[n])
                end_dates.append(end_days[n])

        # Turn the start and end times into numpy arrays
        start_dates = np.array(start_dates)
        end_dates = np.array(end_dates)
    
    return start_dates, end_dates

# Function to display the FD climatology maps
def display_climatology_map(data, lat, lon, title = 'tmp', cbar_label = 'tmp', globe = False, 
                            cmin = -20, cmax = 80, cint = 1, cticks = np.arange(0, 90, 10), new_colorbar = True, cbar_months = False,
                            path = './', savename = 'tmp.png'):
    '''
    Create a map plot of FD climatology

    Inputs:
    :param data: FD map to be plotted
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param title: Title of the plot
    :param globe: Boolean indicating whether the data is global
    :param cmin, cmax, cint: The minimum, maximum, and interval of the values in the colorbar
    :param cticks: List or 1D array of the values to make the ticks in the colorbar
    :param new_colorbar: Boolean indicating whether to make/use a new, adjusted colorbar (separate from the raw one)
    :param cbar_months: Boolean indicating whether to label the colorbar with months instead of values
    :param path: Path the figures will be saved to
    :param savename: Filename of the figure to be saved to

    Outputs:
    Map of FD climatology will be made and saved
    '''
    #### Create the Plot ####
    
    # Set colorbar information
    clevs = np.arange(cmin, cmax + cint, cint)
    nlevs = len(clevs)
    cmap  = plt.get_cmap(name = 'hot_r', lut = nlevs)

    # Get the normalized color values
    vmin = 0 if cmin < 0 else cmin
    v = cmin if cmin < 0 else 0
    norm = mcolors.Normalize(vmin = vmin, vmax = cmax)

    # Create a new/adjust the colorbar?
    if np.invert(cbar_months):
        
        # Generate the colors from the orginal color map in range from [0, cmax]
        colors = cmap(np.linspace(1 - (cmax - vmin)/(cmax - v), 1, cmap.N))  ### Note, in the event cmin and cmax share the same sign, 1 - (cmax - cmin)/cmax should be used
        if new_colorbar:
            colors[:4,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
        else:
            colors[:1,:] = np.array([1., 1., 1., 1.]) # Change the value of 0 to white
        
        # Create a new colorbar cut from the colors in range [0, cmax.]
        ColorMap = mcolors.LinearSegmentedColormap.from_list('cut_hot_r', colors)
        
        colorsNew = cmap(np.linspace(0, 1, cmap.N))
        if new_colorbar:
            colorsNew[abs(cmin)-1:abs(cmin)+1, :] = np.array([1., 1., 1., 1.]) # Change the value of 0 in the plotted colormap to white
        cmap = mcolors.LinearSegmentedColormap.from_list('hot_r', colorsNew)
    
    # Shapefile information
    # ShapeName = 'Admin_1_states_provinces_lakes_shp'
    if np.invert(globe):
        ShapeName = 'admin_0_countries'
        CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)
    
        CountriesReader = shpreader.Reader(CountriesSHP)
    
        USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
        NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']
    
    # Lonitude and latitude tick information
    if globe:
        lat_int = 15
        lon_int = 40
    else:
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
    ax.set_title(title, size = 14)
    
    # Ocean and non-U.S. countries covers and "masks" data outside the U.S.
    ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'white', zorder = 2)
    if np.invert(globe):
        ax.add_feature(cfeature.STATES)
        ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 3)
        ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'white', zorder = 2)
    else:
        ax.coastlines(edgecolor = 'black', zorder = 3)
    
    # Adjust the ticks
    ax.set_xticks(LonLabel, crs = ccrs.PlateCarree())
    ax.set_yticks(LatLabel, crs = ccrs.PlateCarree())
    
    ax.set_yticklabels(LatLabel, fontsize = 14)
    ax.set_xticklabels(LonLabel, fontsize = 14)
    
    ax.xaxis.set_major_formatter(LonFormatter)
    ax.yaxis.set_major_formatter(LatFormatter)
    
    # Plot the flash drought data
    if globe:
        cs = ax.pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                           cmap = cmap, transform = data_proj, zorder = 1)
    else:
        cs = ax.pcolormesh(lon, lat, data, vmin = cmin, vmax = cmax,
                           cmap = cmap, transform = data_proj, zorder = 1)
    
    # Set the map extent to the U.S.
    if globe:
        ax.set_extent([-179, 179, -60, 75])
    else:
        ax.set_extent([-130, -65, 23.5, 48.5])
    
    
    # Set the colorbar size and location
    if globe:
        cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
    else:
        cbax = fig.add_axes([0.915, 0.29, 0.025, 0.425])
    
    # Create the colorbar
    if new_colorbar:
        cbar = mcolorbar.ColorbarBase(cbax, cmap = ColorMap, norm = norm, orientation = 'vertical')
    else:
        cbar = mcolorbar.ColorbarBase(cbax, cmap = cmap, norm = norm, orientation = 'vertical')
    
    # Set the colorbar label
    cbar.ax.set_ylabel(cbar_label, fontsize = 14)
    
    # Set the colorbar ticks
    if cbar_months:
        cbar.set_ticks([0.5, 1.3, 2.3, 3.15, 4.2, 5.05, 6.0, 6.9, 7.8, 8.75, 9.7, 10.7, 11.6])
        cbar.ax.set_yticklabels(['No FD', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], fontsize = 14)
    else:
        cbar.set_ticks(cticks)
        cbar.ax.set_yticklabels(cticks, fontsize = 14)
    
    # Save the figure
    plt.savefig('%s/%s'%(path, savename), bbox_inches = 'tight')
    plt.show(block = False)


#%%
##############################################

# Create a function to create an Aridity Index (AI) mask as defined in Christian et a. 2023
def create_ai_mask(precip, pet, lat, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Create a mask from the aridity index (AI) specified in Christian et al. 2023
    
    Inputs:
    :param precip: Precipitation data. Time x lat x lon format
    :param pet: Potential evaporation data. Time x lat x lon format. Should be in the same units as precip
    :param lat: Vector latitude values corresponding to data
    :param mask: Land-sea mask for the et and pet variables. A a value of none can be provided to not use a mask
    :param dates: Array of datetimes corresponding to the timestamps in precip and pet
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param ai_mask:
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    # Determine the annual precipitation and PET
    T, I, J = pet.shape
    
    pet = np.abs(pet)
    
    climo_years = np.arange(start_year, end_year+1)
    
    print('Calculating the Aridity Index...')
    p_annual = np.ones((climo_years.size, I, J), dtype = np.float32) * np.nan
    pet_annual = np.ones((climo_years.size, I, J), dtype = np.float32) * np.nan
    
    for t, year in enumerate(climo_years):
        ind = np.where(year == years)[0]
        p_annual[t,:,:] = np.nansum(precip[ind,:,:], axis = 0)
        pet_annual[t,:,:] = np.nansum(pet[ind,:,:], axis = 0)
    
    # Calculate the aridity index (ratio of the average annual precip and PET)
    ai = np.nanmean(p_annual, axis = 0)/np.nanmean(pet_annual, axis = 0)
    
    # Note moisture units are kg m^-2, but one of the mask requirements is PET < 1 mm/day, so some unit conversion is necessary
    # For unit conversion, divide by the density of water (m), convert m to mm (multiply by 1000; mm)
    print('Calculating daily mean PET...')
    pet = pet * 1000 / (1000)
    pet = pet.astype(np.float32)
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    months = months[climo_index]

    # Collect the growing season for the northern and southern hemispheres
    ind_north = np.where(lat >= 0)[0]
    ind_south = np.where(lat < 0)[0]
    
    pet_north = pet[:,ind_north,:]
    pet_south = pet[:,ind_south,:]
    
    ind_north = np.where( (months >= 4) & (months <= 10) )[0]
    ind_south = np.where( (months >= 9) | (months <= 4) )[0]
    ind_south = ind_south[:len(ind_north)] # Select up to the same number of pentads as ind_north (for concatenating); should be to about April 5
    
    # Determine the average daily PET in mm day^-1 (215 days in a growing season)
    pet_north = np.nansum(pet_north[ind_north,:,:], axis = 0)/(len(ind_north)*5) # len(ind_north)*5 is the number of days in all the growing seasons
    pet_south = np.nansum(pet_south[ind_south,:,:], axis = 0)/(len(ind_south)*5)
    
    pet_daily_mean = np.concatenate([pet_north, pet_south], axis = 0)
    
    print('Collecting the mask...')
    # Masked grids are highly arid locations (AI < 0.2, and average daily PET < 1 mm day^-1)
    ai_mask = np.where( (ai < 0.2) | (mask == 0) | (pet_daily_mean < 1), 0, 1)

    # Make sure tropics are not masked
    ind = np.where((lat >= -13) & (lat <= 13))[0]
    ai_mask[ind,:] = 1
    ai_mask[mask == 0] = 0
    
    
    print('Done')
    return ai_mask

    

#%%
##############################################

# Create a function to calcualte flash droughts using an improved version of the FD identification method from Christian et al. 2019
# This method uses SESR to identify FD

def christian_fd(sesr, mask, dates, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought using an updated version of the method described in Christian et al. 2019
    (https://doi.org/10.1175/JHM-D-18-0198.1). This method uses the evaporative stress ratio (SESR) to 
    identify flash drought. Updates to the method are details (for LSWI) in Christian et al. 2022
    (https://doi.org/10.1016%2Fj.rsase.2022.100770).
    
    Inputs:
    :param sesr: Input SESR values, time x lat x lon format
    :param mask: Land-sea mask for the et and pet variables. A a value of none can be provided to not use a mask
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
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
        
    # Is a mask provided?
    mask_provided = mask is not None
        

    # Initialize some variables
    T, I, J = sesr.shape
    sesr_inter = np.ones((T, I, J)) * np.nan
    sesr_filt  = np.ones((T, I, J)) * np.nan
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    
    sesr = sesr.reshape(T, I*J, order = 'F')
    sesr_inter = sesr_inter.reshape(T, I*J, order = 'F')
    sesr_filt  = sesr_filt.reshape(T, I*J, order = 'F')
    
    if mask_provided:
        mask = mask.reshape(I*J, order = 'F')
    
    x = np.arange(-6.5, 6.5, (13/T))[:-1] # a variable covering the range of all SESR values with 1 entry for each time step
    print(x.size, T)
    
    # Parameters for the filter
    WinLength = 21 # Window length of 21 pentads
    PolyOrder = 4

    # Perform a basic linear interpolation for NaN values and apply a SG filter
    print('Applying interpolation and Savitzky-Golay filter to SESR')
    for ij in range(I*J):
        if mask_provided:
            if mask[ij] == 0:
                continue
            else:
                pass
        
        # Perform a linear interpolation to remove NaNs
        ind = np.isfinite(sesr[:,ij])
        if np.nansum(ind) == 0:
            continue
        else:
            pass
        
        ind = np.where(ind == True)[0]
        interp_func = interpolate.interp1d(x[ind], sesr[ind,ij], kind = 'linear', fill_value = 'extrapolate')
        
        sesr_inter[:,ij] = interp_func(x)
        
        # Apply the Savitzky-Golay filter to the interpolated SESR data
        sesr_filt[:,ij] = signal.savgol_filter(sesr_inter[:,ij], WinLength, PolyOrder)
        
    # Reorder SESR back to 3D data
    #sesr_filt = sesr_filt.reshape(T, I, J, order = 'F')



    # Determine the change in SESR
    print('Calculating the change in SESR')
    delta_sesr  = np.ones((T, I*J)) * np.nan
    
    delta_sesr[1:,:] = sesr_filt[1:,:] - sesr_filt[:-1,:]

    
    # Begin the flash drought calculations
    print('Identifying flash drought')
    fd = np.ones((T, I*J)) * np.nan

    #fd = fd.reshape(T, I*J, order = 'F')
    #sesr_filt = sesr_filt.reshape(T, I*J, order = 'F')
    #delta_sesr = delta_sesr.reshape(T, I*J, order = 'F')

    dsesr_percentile = 25
    sesr_percentile  = 20
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]
    
    for ij in range(I*J):
        if mask_provided:
            if mask[ij] == 0:
                continue
            else:
                pass
        
        start_date = dates[-1]
        for t in range(T):
            ind = np.where( (dates[t].month == months[climo_index]) & (dates[t].day == days[climo_index]) )[0]
            
            # Determine the percentiles of dSESR and SESR
            ri_crit = np.nanpercentile(delta_sesr[ind,ij], dsesr_percentile)
            dc_crit = np.nanpercentile(sesr_filt[ind,ij], sesr_percentile)

            
            # If start_date != dates[-1], the rapid intensification criteria is satisified
            # If the rapid intensification and drought component criteria are satisified (and FD period is 30+ days)
            # then FD occurs
            if ( (dates[t] - start_date) >= min_change) & (sesr_filt[t,ij] <= dc_crit):
                fd[t,ij] = 1
            else:
                fd[t,ij] = 0
            
            # # If the change in SESR is below the criteria, change the start date of the flash drought
            if (delta_sesr[t,ij] <= ri_crit) & (start_date == dates[-1]):
                start_date = dates[t]
            elif (delta_sesr[t,ij] <= ri_crit) & (start_date != dates[-1]):
                pass
            else:
                start_date = dates[-1]
            
    # Re-order the flash drought back into a 3D array
    fd = fd.reshape(T, I, J, order = 'F')
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
    spei = spei.reshape(T, I*J, order = 'F')
    delta_spei = delta_spei.reshape(T, I*J, order = 'F')
    
    mask = mask.reshape(I*J, order = 'F')

    # Calculate the occurrence of flash drought
    print('Identifying flash drought')
    fd = np.ones((T, I, J)) * np.nan
    
    fd = fd.reshape(T, I*J, order = 'F')
    
    change_criterion = -2
    drought_criterion = -1.28
    
    min_change = timedelta(days = 30)
    start_date = dates[-1]

    for ij in range(I*J):
        if mask[ij] == 0:
            continue
        else:
            pass
        
        start_date = dates[-1]
        for t in range(T-1):
            
            # If the monthly change in SPEI is below the required change, and SPEI is below the drought threshold, FD occurs
            # Note, since the changes are calculated over a 1 month period, the first criterion in Noguera et al. is automatically satisified
            if (delta_spei[t,ij] <= change_criterion) & (spei[t,ij] <= drought_criterion): 
                fd[t,ij] = 1
            else:
                fd[t,ij] = 0
                
    # Restore the FD back into a 3D array
    fd = fd.reshape(T, I, J, order = 'F')
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

    eddi = eddi.reshape(T, I*J, order = 'F')
    fd = fd.reshape(T, I*J, order = 'F')
    mask = mask.reshape(I*J, order = 'F')

    print('Identifying flash drought')
    for ij in range(I*J):
        if mask[ij] == 0:
            continue
        else:
            pass
        
        # The criteria are EDDI must be 50% greater than EDDI 2 weeks (3 pentads) ago, or a 50 percentile increase in 2 weeks, and remain that intense for another 2 weeks.
        for t in range(3, T-3): 
            
            ind = np.where( (dates[t].month == months[climo_index]) & (dates[t].day == days[climo_index]) )[0]
            
            current_percent = stats.percentileofscore(eddi[ind,ij], eddi[t,ij])
            previous_percent = stats.percentileofscore(eddi[ind,ij], eddi[t-3,ij])
            
            # Note this checks for all pentads in the + 2 week period, so there cannot be moderation
            if ( (current_percent - previous_percent) > 50 ) & (eddi[t+1,ij] >= eddi[t,ij]) & (eddi[t+2,ij] >= eddi[t,ij]) & (eddi[t+3,ij] >= eddi[t,ij]): 
                fd[t,ij] = 1
            else:
                fd[t,ij] = 0
                
    fd = fd.reshape(T, I, J, order = 'F')
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
    
    sm = vsm.reshape(T, I*J, order = 'F')
    mask = mask.reshape(I*J, order = 'F')
    sm_percentiles = sm_percentiles.reshape(T, I*J, order = 'F')
    
    for t in range(T):
        ind = np.where( (dates[t].day == days[climo_index]) & (dates[t].month == months[climo_index]) )[0]
        
        for ij in range(I*J):
            if mask[ij] == 0:
                continue
            else:
                pass
    
            sm_percentiles[t,ij] = stats.percentileofscore(sm[ind,ij], sm[t,ij])
        
        
    # Begin drought identification process
    print('Identifying flash droughts')
    fd = np.ones((T, I, J)) * np.nan
    
    fd = fd.reshape(T, I*J, order = 'F')
    
    # Initialize up a variable to look up to 12 pentads ahead (from Otkin et al. 2021, that rapid intensification goes up to 10 pentads ahead); 12 ensures data after intensification is included
    fut_pentads = np.arange(0, 13)
    fp = len(fut_pentads)

    
    for ij in range(I*J):
        if (ij%1000) == 0:
            print('%d/%d'%(int(ij/1000), int(I*J/1000)))
        
        if mask[ij] == 0:
            continue
        else:
            pass
        
        for t in range(T-12): # Exclude the last few months in the dataset for simplicity since FD identification involves looking up to 12 pentads ahead
            # First determine if the soil moisture is below the 40 percentile
            # print(ij)
            # print(sm_percentiles2d[t,ij] <= 40)
            if sm_percentiles[t,ij] <= 40:
                
                
                R2 = np.ones((fp)) * np.nan
                ri_entries = np.ones((fp)) * np.nan
                
                # To determine when the percentiles level out (when the intensification ends), regress SM percentiles with pentads with increasing polynomial degrees until R^2 > 0.95 or until a 10th order polynomial is used (assumed accuracy is being lost here)
                for p in range(1, 11):
                    sm_est, R2p = polynomial_regress(fut_pentads, sm_percentiles[t:t+fp,ij], order = p)
                    
                    R2[p-1] = R2p
                    if (R2[p-1] >= 0.95):
                        order = p
                        break
                    elif (p >= 10):
                        # Find the maximum R2
                        ind = np.where(R2 == np.nanmax(R2))[0]
                        if len(ind) < 1: # If no maximum is found, the calculations are all NaNs and nothing can be determined
                            fd[t,ij] = 0
                            break
                        
                        order = ind[0]+1
                        
                        # Get the SM estimates for the polynomial regression with the highest R2
                        sm_est, R2p = polynomial_regress(fut_pentads, sm_percentiles[t:t+fp,ij], order = order)
                        break
                    else:
                        pass
                    
                # Next, determine where the change in sm_est is approximately 0 (within 0.01) to find when the rapid intensification ends
                for pent in fut_pentads[1:]:
                    ri_entries[pent-1] = (sm_percentiles[t+pent,ij] - sm_percentiles[t,ij])/pent # pent here is the difference between the current pentad and how many pentads ahead one is looking
                    
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
                if (sm_percentiles[t+ri_end,ij] <= 20) & ( (ri_mean >= 5) | (ri_max >= 7.5) ):
                    fd[t+ri_end,ij] = 1
                else:
                    fd[t+ri_end,ij] = 0
                    
                # Increment t to the end of the intensification period
                t = t + ri_end
                
            else:
                fd[t,ij] = 0
            #     continue
    
    
    fd = fd.reshape(T, I, J, order = 'F')    
    
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
    
    parser.add_argument('--dataset', type=str, default='./Data', help='Data set directory')

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
    
    if args.model == 'era5':
        globe = True
    else:
        globe = False
    
    # Load the land-sea mask?
    if args.mask:
        mask = load_mask(model = args.model, path = args.dataset)
        
        # For global models, add to the mask with the aridity index
        if globe == True:
            p = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
            
            # Times 24 because the the data was averaged over each hour in the initial pre-processing; times 24 makes it like a daily sum
            ai_mask = create_ai_mask(p['precip']*24, np.abs(pet['pevap'])*24, pet['lat'][:,0], mask, pet['ymd'], start_year = start_date.year, end_year = end_date.year)
            
            # Make a mask without the sea grid points
            I, J = ai_mask.shape
            ai_mask_no_sea = np.ones((1, I, J))
            ai_mask_no_sea[0,:,:] = ai_mask
            ai_mask_no_sea, _, _, _ = remove_sea(ai_mask_no_sea, p['lat'], p['lon'], mask)
            
            del p, pet
            gc.collect()
            
            # Write the full gridded aridity mask
            with Dataset('%s/ai_mask.nc'%dataset_dir, 'w', format = 'NETCDF4') as nc:
                I, J = ai_mask.shape
                
                # Create the spatial and temporal dimensions
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                
                # Create the main variable
                nc.createVariable('mask', ai_mask.dtype, ('x', 'y'))
                nc.variables['mask'][:,:] = ai_mask[:,:]
                
            # Now write the mask without the sea grid points
            with Dataset('%s/ai_mask_no_sea.nc'%dataset_dir, 'w', format = 'NETCDF4') as nc:
                T, I, J = ai_mask_no_sea.shape
                
                # Create the spatial and temporal dimensions
                nc.createDimension('x', size = I)
                nc.createDimension('y', size = J)
                
                # Create the main variable
                nc.createVariable('mask', ai_mask_no_sea.dtype, ('x', 'y'))
                nc.variables['mask'][:,:] = ai_mask_no_sea[0,:,:]
                
            
        
    if args.model == 'narr':
        level = '0-40cm'
    elif args.model == 'era5':
        level = '0-28cm'
    elif args.model == 'nldas':
        level = '50cm'
                
    
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
                    # Only NARR has soil moisture split into multiple layers
                    if args.model == 'narr':
                        sm0 = load_nc('soilm', 'soil_moisture.0cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                        sm10 = load_nc('soilm', 'soil_moisture.10cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                        sm40 = load_nc('soilm', 'soil_moisture.40cm.%s.pentad.nc'%args.model, sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    else:
                        if args.model == 'era5':
                            level = '0-28cm'
                        elif args.model == 'nldas':
                            level = '50cm'
                            
                        sm0 = load_nc('soilm', 'soil_moisture.%s.%s.pentad.nc'%(level, args.model), sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    
                    lat = sm0['lat']; lon = sm0['lon']; dates = sm0['ymd']
                    
                    # Calculate the index
                    if args.model == 'narr':
                        index_data = caculate_smi([sm0['soilm'], sm10['soilm'], sm40['soilm']], dates, mask, start_year = start_date.year, end_year = end_date.year)
                        
                        # Remove the no longer needed variables
                        del sm0, sm10, sm40
                    else:
                        index_data = caculate_smi([sm0['soilm']], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                        # Remove the no longer needed variables
                        del sm0
                
                elif index == 'sodi':
                    p = load_nc('precip', 'precipitation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    et = load_nc('evap', 'evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    pet = load_nc('pevap', 'potential_evaporation.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    sm = load_nc('soilm', 'soil_moisture.%s.%s.pentad.nc'%(level, args.model), sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    ro = load_nc('ro', 'runoff.%s.pentad.nc'%args.model, sm = False, path = '%s/Processed_Data/'%dataset_dir)
                    lat = p['lat']; lon = p['lon']; dates = p['ymd']
                    
                    # Calculate the index
                    index_data = calculate_sodi(p['precip'], et['evap'], pet['pevap'], sm['soilm'], ro['ro'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    # Remove the no longer needed variables
                    del p, et, pet, sm, ro
                
                elif index == 'fdii':
                    if args.model == 'narr':
                        level = '0-40cm'
                    elif args.model == 'era5':
                        level = '0-28cm'
                    elif args.model == 'nldas':
                        level = '50cm'
                
                    sm = load_nc('soilm', 'soil_moisture.%s.%s.pentad.nc'%(level, args.model), sm = True, path = '%s/Processed_Data/'%dataset_dir)
                    lat = sm['lat']; lon = sm['lon']; dates = sm['ymd']
                    
                    # Calcualte the index
                    fdii, fd_int, dro_sev = calculate_fdii(sm['soilm'], dates, mask, start_year = start_date.year, end_year = end_date.year)
                    
                    index_data = [fdii, fd_int, dro_sev]
                    index_names = ['fdii', 'fd_int', 'dro_sev']
                    
                    # Remove the no longer needed variables
                    del sm
                    
                # Write the index data so that it is available for future use
                if index != 'fdii':
                    index_data = index_data.astype(np.float32)
                
                    write_nc(index_data, lat, lon, dates, filename = '%s.%s.pentad.nc'%(index,args.model), VarSName = index, path = '%s/Indices/'%dataset_dir)
                else: # Special case for FDII, which gives 3 variabels to save
                    for datum, name in zip(index_data, index_names):
                        datum = datum.astype(np.float32)
                        write_nc(datum, lat, lon, dates, filename = '%s.%s.pentad.nc'%(name,args.model), VarSName = name, path = '%s/Indices/'%dataset_dir)
                
                
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
                        display_maximum_map(index_data, lat, lon, dates, datetime(2012, 5, 1), datetime(2012, 8, 1), index, globe = globe, path = dataset_dir)
                    else:
                        for datum, name in zip(index_data, index_names):
                            display_maximum_map(datum, lat, lon, dates, datetime(2012, 5, 1), datetime(2012, 8, 1), name, globe = globe, path = dataset_dir)
            
            
        
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
                index = load_nc('soilm', 'soil_moisture.%s.%s.pentad.nc'%(level, args.model), sm = True, path = '%s/Processed_Data/'%dataset_dir)
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
            
            # For global datasets, remove the highly arid grid points
            if globe == True:
                fd = apply_mask(fd, ai_mask)
                
                
            fd = fd.astype(np.float32)
            
            # Write the FD data so that it is available for future use
            write_nc(fd, lat, lon, dates, filename = '%s.%s.pentad.nc'%(method,args.model), VarSName = 'fd', path = '%s/FD_Data/'%dataset_dir)
            
        # Make a FD climatology map?
        if args.fd_climatology:
            display_fd_climatology([fd], lat, lon, dates, mask, [method], model = args.model, globe = globe, path = dataset_dir)
            
    
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
        if args.model == 'era5':
                ch_fd['fd'], ch_fd['lat'], ch_fd['lon'], _ = remove_sea(ch_fd['fd'], ch_fd['lat'], ch_fd['lon'], mask)
                
        nog_fd = load_nc('fd', 'nogeura.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        if args.model == 'era5':
                nog_fd['fd'], nog_fd['lat'], nog_fd['lon'], _ = remove_sea(nog_fd['fd'], nog_fd['lat'], nog_fd['lon'], mask)
                
        pen_fd = load_nc('fd', 'pendergrass.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        if args.model == 'era5':
                pen_fd['fd'], pen_fd['lat'], pen_fd['lon'], _ = remove_sea(pen_fd['fd'], pen_fd['lat'], pen_fd['lon'], mask)
                
        liu_fd = load_nc('fd', 'liu.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        if args.model == 'era5':
                liu_fd['fd'], liu_fd['lat'], liu_fd['lon'], _ = remove_sea(liu_fd['fd'], liu_fd['lat'], liu_fd['lon'], mask)
        # li_fd = load_nc('fd', 'li.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        #if args.model == 'era5':
        #        li_fd['fd'], li_fd['lat'], li_fd['lon'], _ = remove_sea(li_fd['fd'], li_fd['lat'], li_fd['lon'], mask)
        
        ot_fd = load_nc('fd', 'otkin.%s.pentad.nc'%args.model, path = '%s/FD_Data/'%dataset_dir)
        if args.model == 'era5':
                ot_fd['fd'], ot_fd['lat'], ot_fd['lon'], mask = remove_sea(ot_fd['fd'], ot_fd['lat'], ot_fd['lon'], mask)
        
        print(np.nanmin(ch_fd['fd']), np.nanmax(ch_fd['fd']))
        print(np.nanmin(nog_fd['fd']), np.nanmax(nog_fd['fd']))
        print(np.nanmin(pen_fd['fd']), np.nanmax(pen_fd['fd']))
        print(np.nanmin(liu_fd['fd']), np.nanmax(liu_fd['fd']))
        print(np.nanmin(ot_fd['fd']), np.nanmax(ot_fd['fd']))
        print(ch_fd['fd'].dtype, nog_fd['fd'].dtype, pen_fd['fd'].dtype, liu_fd['fd'].dtype, ot_fd['fd'].dtype)
        
        # Parse and save the label data into a pickle file
        parse_data([ch_fd['fd'], nog_fd['fd'], pen_fd['fd'], liu_fd['fd'], ot_fd['fd']], 
                    ch_fd['lat'], ch_fd['lon'], dates, dataset_dir, label_fname, args.model, ai_mask_no_sea)
        
            
            


