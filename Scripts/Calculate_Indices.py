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

Notes:
- Note EDDI and SEDI do not use the 1990 - 2020 climatology period, but the entire dataset
    - EDDI ranks the data and does not normalize anything by a climate period at any point, but uses the entire dataset
    - SEDI exhibited a strange behavior; if the 1990 - 2020 climatology is used to calculate the mean and std, then SEDI becomes a uniform distribution


TODO:
- Make the climatologies focus on 1990 - 2020 (EDDI)
- Modify map creation functions to include the whole world
- Update soil moisture calculations to include arbitrary layers between 0 and 40 cm
- Investigate EDDI further. The histogram for it is quite odd.
"""


#%%
##############################################

# Import libraries
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

# Import a custom script
from Raw_Data_Processing import *


#%%
##############################################
# Create a function to collect climatological data

def collect_climatology(X, dates, start_year, end_year):
    '''
    Extract data between the beginning and ending years.
    
    Inputs:
    :param X: Input 3D data, in time x lat x lon format
    :param dates: Array of datetimes corresponding to the time axis of X
    :param start_year: The beginning year in the interval being searched
    :param end_year: The ending year in the interval being searched
    
    Outputs:
    :param X_climo: X between the specified start and end dates
    '''
    
    # Turn the start and end years into datetimes
    begin_date = datetime(start_year, 1, 1)
    end_date   = datetime(end_year, 12, 31)
    
    # Determine all the points between the start and end points
    ind = np.where( (dates >= begin_date) & (dates <= end_date) )[0]
    
    if len(X.shape) < 3:
        X_climo = X[ind]
    else:
        X_climo = X[ind,:,:]
    
    return X_climo


#%% 
##############################################
# Calculate the climatological means and standard deviations
  
def calculate_climatology(X, pentad = True):
    '''
    Calculates the climatological mean and standard deviation of gridded data.
    Climatological data is calculated for all grid points and for all timestamps in the year.
    
    Inputs:
    :param X: 3D variable whose mean and standard deviation will be calculated.
    :param pentad: Boolean value giving if the time scale of X is 
                   pentad (5 day average) or daily.
              
    Outputs:
    :param clim_mean: Calculated mean of X for each day/pentad and grid point. 
                      clim_mean as the same spatial dimensions as X and 365 (73)
                      temporal dimension for daily (pentad) data.
    :param clim_std: Calculated standard deviation for each day/pentad and grid point.
                     clim_std as the same spatial dimensions as X and 365 (73)
                     temporal dimension for daily (pentad) data.
    '''
    
    # Obtain the dimensions of the variable
    if len(X.shape) < 3:
        T = X.size
    else:
        T, I, J = X.shape
    
    # Count the number of years
    if pentad is True:
        year_len = int(365/5)
    else:
        year_len = int(365)
        
    N_years = int(np.ceil(T/year_len))
    
    # Create a variable for each day, assumed starting at Jan 1 and no
    #   leap years (i.e., each year is only 365 days each)
    day = np.ones((T)) * np.nan
    
    n = 0
    for i in range(1, N_years+1):
        if i >= N_years:
            day[n:T+1] = np.arange(1, len(day[n:T+1])+1)
        else:
            day[n:n+year_len] = np.arange(1, year_len+1)
        
        n = n + year_len
    
    # Initialize the climatological mean and standard deviation variables
    if len(X.shape) < 3:
        clim_mean = np.ones((year_len)) * np.nan
        clim_std  = np.ones((year_len)) * np.nan
    else:
        clim_mean = np.ones((year_len, I, J)) * np.nan
        clim_std  = np.ones((year_len, I, J)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, year_len+1):
        ind = np.where(i == day)[0]
        
        if len(X.shape) < 3:
            clim_mean[i-1] = np.nanmean(X[ind], axis = 0)
            clim_std[i-1]  = np.nanstd(X[ind], axis = 0)
        else:
            clim_mean[i-1,:,:] = np.nanmean(X[ind,:,:], axis = 0)
            clim_std[i-1,:,:]  = np.nanstd(X[ind,:,:], axis = 0)
    
    return clim_mean, clim_std

#%%
##############################################

# Function to remove sea data points
def apply_mask(data, mask):
    '''
    Turn sea points into NaNs based on a land-sea mask where 0 is sea and 1 is land
    
    Inputs:
    :param data: Data to be masked
    :param mask: Land-sea mask. Must have the same spatial dimensions as data
    
    Outputs:
    :param data_mask: Data with all labeled sea grids at NaN
    '''
    
    T, I, J = data.shape
    
    data_mask = np.ones((T, I, J)) * np.nan
    for t in range(T):
        data_mask[t,:,:] = np.where(mask == 1, data[t,:,:], np.nan)
        
    return data_mask

#%%
##############################################

# Function to calculate SESR
# Details in SESR can be found in the Christian et al. 2019 paper.
def calculate_sesr(et, pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the standardized evaporative stress ratio (SESR) from ET and PET.
    
    Full details on SESR can be found in Christian et al. 2019 (for SESR): https://doi.org/10.1175/JHM-D-18-0198.1.
    
    Inputs:
    :param et: Input evapotranspiration (ET) data
    :param pet: Input potential evapotranspiration (PET) data. Should be in the same units as et
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
        
    Outputs:
    :param sesr: Calculate SESR, has the same shape and size as et/pet
    
    '''

    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])

    # Obtain the evaporative stress ratio (ESR); the ratio of ET to PET
    esr = et/pet

    # Remove values that exceed a certain limit as they are likely an error
    esr[esr < 0] = np.nan
    esr[esr > 3] = np.nan

    # Collect the climatological data for the ESR
    esr_climo = collect_climatology(esr, dates, start_year = start_year, end_year = end_year)

    # Determine the climatological mean and standard deviations of ESR
    esr_mean, esr_std = calculate_climatology(esr_climo, pentad = True)

    # Find the time stamps for a singular year
    ind = np.where(years == 1999)[0] # Note, any non-leap year will do
    one_year = dates[ind]

    # Calculate SESR; it is the standardized ESR
    T, I, J = esr.shape

    sesr = np.ones((T, I, J)) * np.nan

    for n, date in enumerate(one_year):
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        for t in ind:
            sesr[t,:,:] = (esr[t,:,:] - esr_mean[n,:,:])/esr_std[n,:,:]
            

    # Remove any unrealistic points
    sesr = np.where(sesr < -5, -5, sesr)
    sesr = np.where(sesr > 5, 5, sesr)
    
    # Remove any sea data points
    sesr = apply_mask(sesr, mask)
    # sesr[mask[:,:] == 0] = np.nan
    
    return sesr


#%%
##############################################

# Create a function to calculate SEDI
# Details on SEDI can be found in the Li et al. 2020a paper.
def calculate_sedi(et, pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the standardized evaporative deficit index (SEDI) from ET and PET.
    
    Full details on SEDI can be found in Li et al. 2020a: https://doi.org/10.1016/j.catena.2020.104763
    
    Inputs:
    :param et: Input evapotranspiration (ET) data
    :param pet: Input potential evapotranspiration (PET) data. Should be in the same units as et
    :param dates: Array of datetimes corresponding to the timestamps in et and pet
    :param mask: Land-sea mask for the et and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates.
    
    Outputs:
    :param sedi: Calculate SEDI, has the same shape and size as et/pet
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])

    # Obtain the evaporative deficit (ED); the difference between ET to PET
    ed = et - pet
    
    
    # Collect the climatological data for the ED
    ed_climo = collect_climatology(ed, dates, start_year = start_year, end_year = end_year)

    # Determine the climatological mean and standard deviations of ED
    ed_mean, ed_std = calculate_climatology(ed, pentad = True)
    
    print(np.nanmax(ed_mean), np.nanmin(ed_mean), np.nanmean(ed_mean))
    print(np.nanmax(ed_std), np.nanmin(ed_std), np.nanmean(ed_std))

    # Find the time stamps for a singular year
    ind = np.where(years == 1999)[0] # Note, any non-leap year will do
    one_year = dates[ind]

    # Calculate SEDI; it is the standardized ED
    T, I, J = ed.shape

    sedi = np.ones((T, I, J)) * np.nan
    
    for n, date in enumerate(one_year):
        ind = np.where( (date.month == months) & (date.day == days) )[0]
        
        for t in ind:
            sedi[t,:,:] = (ed[t,:,:] - ed_mean[n,:,:])/ed_std[n,:,:]
            

    # Remove any sea data points
    sedi = apply_mask(sedi, mask)
    # sedi[mask[:,:] == 0] = np.nan
    
    print(np.nanmax(sedi), np.nanmin(sedi), np.nanmean(sedi), np.nanstd(sedi))
    
    return sedi


#%%
##############################################

# Create functions to calculate SPEI and SAPEI
# Details for SPEI can be found in the Vicente-Serrano et al. 2010 paper.
# Details for SAPEI can be found in the Li et al. 2020b paper.

def transform_pearson3(data, time, climo = None, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Transform 3D gridded data in Pearson Type III distribution to a standard normal distribution.
    
    Inputs:
    :param data: Pearson Type III distributed data to be transformed
    :param time: Vector of datetimes corresponding to the timestamp in each timestep in precip and pet
    :param climo: Subset of data consisting of the data used to determine the parameters of Pearson Type III distribution. Default is data
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates
    
    Outputs:
    :param data_norm: data in a standard normal distribution, same shape and size as data
    '''
    
    print('Initializing some values')
    
    # Climo dataset specified?
    if climo.all() == None:
        climo = data

    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in time])
        
    if months == None:
        months = np.array([date.month for date in time])
        
    if days == None:
        days = np.array([date.day for date in time])
    
    
    # Initialize some needed variables.
    T, I, J = data.shape
    T_climo = climo.shape[0]

    climo_index = np.where((years >= start_year) & (years <= end_year))[0]
    
    N = int(T/len(np.unique(years))) # Number of observations per year
    N_obs = int(T_climo/N) # Number of observations per time series; number of years in climo
    #N_obs = len(np.unique(years)) # Number of observations per time series
    
    # Define the constants given in Vicente-Serrano et al. 2010
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328

    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    frequencies = np.ones((T, I, J)) * np.nan
    PWM0 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 0
    PWM1 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 1
    PWM2 = np.ones((N, I, J)) * np.nan # Probability weighted moment of 2
    
    # Determine the frequency estimator and moments according to equation in section 3 of the Vicente-Serrano et al. 2010 paper
    print('Calculating moments')
    for t, date in enumerate(time[:N]):
        ind = np.where( (months[climo_index] == date.month) & (days[climo_index] == date.day) )[0]

        # Get the frequency estimator
        frequencies[ind,:,:] = (stats.mstats.rankdata(climo[ind,:,:], axis = 0) - 0.35)/N_obs

        # Get the moments
        PWM0[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**0)*climo[ind,:,:], axis = 0)/N_obs
        PWM1[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**1)*climo[ind,:,:], axis = 0)/N_obs
        PWM2[t,:,:] = np.nansum(((1 - frequencies[ind,:,:])**2)*climo[ind,:,:], axis = 0)/N_obs

    # Calculate the parameters of log-logistic distribution, using the equations in the Vicente-Serrano et al. 2010 paper
    print('Calculating Pearson Type III distribution parameters')
    beta  = (2*PWM1 - PWM0)/(6*PWM1 - PWM0 - 6*PWM2) # Scale parameter
    
    alpha = (PWM0 - 2*PWM1)*beta/(gamma(1+1/beta)*gamma(1-1/beta)) # Shape parameter
    
    gamm  = PWM0 - alpha*gamma(1+1/beta)*gamma(1-1/beta) # Origin parameter; note gamm refers to the gamma parameter

    # Obtain the cumulative distribution of the moisture deficit.
    print('Calculating the cumulative distribution of P - PET')
    F = np.ones((T, I, J)) * np.nan

    for n, date in enumerate(time[:N]):
        ind = np.where( (date.month == months) & (date.day == days) )[0]

        for t in ind:
            F[t,:,:] = (1 + (alpha[n,:,:]/(data[t,:,:] - gamm[n,:,:]))**beta[n,:,:])**-1

    # Some variables are no longer needed. Remove them to conserve memory.
    del frequencies, PWM0, PWM1, PWM2, beta, alpha, gamm
    gc.collect() # Clears deleted variables from memory

    # Finally, use this to obtain the probabilities and convert the data to a standardized normal distribution
    prob = 1 - F
    prob = np.where(prob == 0, 1e-5, prob) # Remove probabilities of 0
    prob = np.where(prob == 1, 1-1e-5, prob) # Remove probabilities of 1
    
    data_norm = np.ones((T, I, J)) * np.nan 

    # Reshape arrays into 2D for calculations
    prob = prob.reshape(T, I*J)
    data_norm = data_norm.reshape(T, I*J)
    
    # Calculate SPEI based on the inverse normal approximation given in Vicente-Serrano et al. 2010, Sec. 3
    print('Converting P - PET probabilities into a normal distribution')
    data_sign = 1
    for ij in range(I*J):
        if (ij%1000) == 0:
            print('%d/%d'%(int(ij/1000), int(I*J/1000)))
        for t in range(T):
            
            # Determine if to multiple the equation by 1 or -1, and whether to use prob or 1 - prob
            if prob[t,ij] <= 0.5:
                prob[t,ij] = prob[t,ij]
                data_sign = 1
            else:
                prob[t,ij] = 1 - prob[t,ij]
                data_sign = -1

            # Determine W
            W = np.sqrt(-2 * np.log(prob[t,ij]))

            # Calculate the normal distribution
            data_norm[t,ij] = data_sign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
    
    # Reshape SPEI back into a 3D array
    data_norm = data_norm.reshape(T, I, J)
    
    print('Done')
    return data_norm

def calculate_spei(precip, pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the standardized precipitation evaporation index (SPEI) from precipitation and potential evaporation data.
    SPEI index is on the same time scale as the input data.
    
    Full details on SPEI can be found in Vicente-Serrano et al. 2010: https://doi.org/10.1175/2009JCLI2909.1
    
    Inputs:
    :param precip: Input precipitation data (should be for over 10+ years). Time x lat x lon format
    :param pet: Input potential evaporation data (should be over 10+ years). Time x lat x lon format. Should be in the same units as precip
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in precip and pet
    :param mask: Land-sea mask for the precip and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param spei: The SPEI drought index, has the same shape and size as precip/pet
    '''
    
    # Determine the moisture deficit
    D = (precip) - pet
    
    # Collect the climatological data for the ED
    D_climo = collect_climatology(D, dates, start_year = start_year, end_year = end_year)
    
    # Transform D from a Pearson Type III distribution to a standard normal distribution
    spei = transform_pearson3(D, dates, climo = D_climo, start_year = start_year, end_year = end_year, years = years, months = months, days = days)
    
    # Remove any sea data points
    spei = apply_mask(spei, mask)
    # spei[mask[:,:] == 0] = np.nan
    
    return spei


def calculate_sapei(precip, pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the standardized antecedent precipitation evaporation index (SAPEI) from precipitation and potential evaporation data.
    SAPEI index is on the same time scale as the input data.
    
    Full details on SAPEI can be found in Li et al. 2020b: https://doi.org/10.1175/JHM-D-19-0298.1
    
    Inputs:
    :param precip: Input precipitation data (should be for over 10+ years). Time x lat x lon format
    :param pet: Input potential evaporation data (should be over 10+ years). Time x lat x lon format. Should be in the same units as precip
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in precip and pet
    :param mask: Land-sea mask for the precip and pet variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param sapei: The SAPEI drought index, has the same shape and size as precip/pet
    '''
    
    a = 0.903 # Note this decay rate is defined by keeping the total decay (13%) after 100 days or 20 pentads.
              # These values may be adjusted, as SAPEI with this decay/memory is like unto a 3-month SPEI
              # (see Li et al. 2020b sections 3a and 4a).

    # Initialize the moisture deficit D
    T, I, J = precip.shape
    D = np.zeros((T, I, J))

    NDays = 100 # Number of days in the decay/memory
    counters = np.arange(1, (NDays/5)+1)

    # Determine the moisture deficit
    for t in range(T):
        for i in counters:
            i = int(i)
            if i > t:
                break
            
            moistDeficit = (a**i) * (precip[t-i,:,:] - pet[t-i,:,:])
            
            D[t,:,:] = D[t,:,:] + moistDeficit
    
    
    
    # Collect the climatological data for the ED
    D_climo = collect_climatology(D, dates, start_year = start_year, end_year = end_year)
    
    # Transform D from a Pearson Type III distribution to a standard normal distribution
    sapei = transform_pearson3(D, dates, climo = D_climo, start_year = start_year, end_year = end_year, years = years, months = months, days = days)
    
    # Remove any sea data points
    sapei = apply_mask(sapei, mask)
    # sapei[mask[:,:] == 0] = np.nan
    
    return sapei


#%%
##############################################

# Create functions to calculate EDDI
# Details are found in the Hobbins et al. 2016 paper.

def calculate_eddi(pet, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the evaporative drought demand index (EDDI) from the potential evaporation data.
    EDDI index is on the same time scale as the input data.
    
    Full details on EDDI can be found in Hobbins et al. 2016: https://doi.org/10.1175/JHM-D-15-0121.1
    
    Inputs:
    :param pet: Input potential evaporation data (should be over 10+ years). Time x lat x lon format
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in pet
    :param mask: Land-sea mask for the pet variable
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param eddi: The EDDI drought index, has the same shape and size as pet
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    climo_index = np.where((years >= start_year) & (years <= end_year))[0]
    months_climo = np.array([date.month for date in dates[climo_index]])
    days_climo = np.array([date.day for date in dates[climo_index]])
    
    # Initialize the set of probabilities of getting a certain PET.
    T, I, J = pet.shape
    # T = len(climo_index)

    prob = np.ones((T, I, J)) * np.nan
    eddi = np.ones((T, I, J)) * np.nan

    N = np.unique(years).size # Number of observations per time series

    # Find the time stamps for a singular year
    ind = np.where(years == 1999)[0] # Note, any non-leap year will do
    one_year = dates[ind]

    # Define the constants given in Hobbins et al. 2016
    C0 = 2.515517
    C1 = 0.802853
    C2 = 0.010328

    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    # Determine the probabilities of getting PET at time t.
    for date in one_year:
        ind = np.where( (months == date.month) & (days == date.day) )[0]
        
        # Collect the rank of the time series. Note in Hobbins et al. 2016, maximum PET is assigned rank 1 (so min PET has the highest rank)
        # This is opposite the order output by rankdata. (N+1) - rankdata puts the rank in the order specificied in Hobbins et al. 2016.
        rank = (N+1) - stats.mstats.rankdata(pet[ind,:,:], axis = 0)
        
        # Calculate the probabilities based on Tukey plotting in Hobbins et al. (Sec. 3a)
        prob[ind,:,:] = (rank - 0.33)/(N + 0.33)
        
        
    # Reorder data to reduce number of embedded loops
    prob2d = prob.reshape(T, I*J, order = 'F')
    eddi2d = eddi.reshape(T, I*J, order = 'F')

    # prob2d = 1 - prob2d

    # Calculate EDDI based on the inverse normal approximation given in Hobbins et al. 2016, Sec. 3a
    eddi_sign = 1
    for ij in range(I*J):
        if (ij%1000) == 0:
            print('%d/%d'%(int(ij/1000), int(I*J/1000)))
        
        for t in range(T):
            if prob2d[t,ij] <= 0.5:
                prob2d[t,ij] = prob2d[t,ij]
                eddi_sign = 1
            else:
                prob2d[t,ij] = 1 - prob2d[t,ij]
                eddi_sign = -1
                
            W = np.sqrt(-2 * np.log(prob2d[t,ij]))
            
            eddi2d[t,ij] = eddi_sign * (W - (C0 + C1 * W + C2 * (W**2))/(1 + d1 * W + d2 * (W**2) + d3 * (W**3)))
        
        # for date in one_year:
        #     ind = np.where( (months == date.month) & (days == date.day) )[0]
        #     eddi2d[ind,ij] = stats.norm.ppf(prob2d[ind,ij], loc = 0, scale = 1)
            
    # Reorder the data back to 3D
    eddi = eddi2d.reshape(T, I, J, order = 'F')

    # eddi = transform_pearson3(pet, dates, climo = pet, start_year = start_year, end_year = end_year)

    # Remove any sea data points
    eddi = apply_mask(eddi, mask)
    # eddi[mask[:,:] == 0] = np.nan
    
    # print(np.nanmin(eddi), np.nanmax(eddi), np.nanmean(eddi), np.nanstd(eddi))
    
    return eddi


#%%
##############################################

# Create functions to calculate SMI
# Details for SMI can be found in the Hunt et al. 2009 paper.
# In a similar vain to the Hunt et al. paper, SMI will be determined for 10, 25, and 40 cm averages of VSM
# These are then averaged together.

def caculate_smi(vsm, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the soil moisture index (SMI) from 0, 10, and 40 cm volumetric soil moisture data
    
    Inputs:
    :param vsm: List of Input volumetric soil moisture data. Time x lat x lon x depth format
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in vsm
    :param mask: Land-sea mask for the vsm variables
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param smi: The SMI drought index, has the same shape of time x lat x lon
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
        
    
    # Collect the individual vsm data
    vsm0 = vsm[0]
    vsm10 = vsm[1]
    vsm40 = vsm[2]
    
    # Initialize some other variables
    T, I, J = vsm0.shape
    
    WP_percentile = 5
    FC_percentile = 95
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    months_climo = months[climo_index]
    days_climo   = days[climo_index]
    
    grow_ind = np.where( (months_climo >= 4) & (months_climo <= 10) )[0] # Percentiles are determined from growing season values.
    
    smi = np.ones((T, I, J)) * np.nan
    
    
    # Reshape arrays into 2D arrays
    smi2d = smi.reshape(T, I*J, order = 'F')
    
    vsm0_2d = vsm0.reshape(T, I*J, order = 'F')
    vsm10_2d = vsm10.reshape(T, I*J, order = 'F')
    vsm40_2d = vsm40.reshape(T, I*J, order = 'F')
    
    
    # Calculate the SMI using the available water capacity based on the climatology period
    for ij in range(I*J):
        # First determine the wilting point and field capacity. This is done by examining 5th and 95th percentiles.
        vsm0_wp  = stats.scoreatpercentile(vsm0_2d[grow_ind,ij], WP_percentile)
        vsm10_wp = stats.scoreatpercentile(vsm10_2d[grow_ind,ij], WP_percentile)
        vsm40_wp = stats.scoreatpercentile(vsm40_2d[grow_ind,ij], WP_percentile)
        
        vsm0_fc  = stats.scoreatpercentile(vsm0_2d[grow_ind,ij], FC_percentile)
        vsm10_fc = stats.scoreatpercentile(vsm10_2d[grow_ind,ij], FC_percentile)
        vsm40_fc = stats.scoreatpercentile(vsm40_2d[grow_ind,ij], FC_percentile)
        
        # Determine the SMI at each level based on equation in section 1 of Hunt et al. 2009
        smi0  = -5 + 10*(vsm0_2d[:,ij] - vsm0_wp)/(vsm0_fc - vsm0_wp)
        smi10 = -5 + 10*(vsm10_2d[:,ij] - vsm10_wp)/(vsm10_fc - vsm10_wp)
        smi40 = -5 + 10*(vsm40_2d[:,ij] - vsm40_wp)/(vsm40_fc - vsm40_wp)
        
        # Average these values together to get the full SMI
        smi_tmp = np.stack([smi0, smi10, smi40], axis = 0)
        smi2d[:,ij] = np.nanmean(smi_tmp, axis = 0)
        
    # Reshape data back to a 3D array.
    smi = smi2d.reshape(T, I, J, order = 'F')

    # Remove any sea data points
    smi = apply_mask(smi, mask)
    # smi[mask[:,:] == 0] = np.nan
    
    return smi


#%%
##############################################

# Create functions to calculate SODI
# Details for SODI can be found in the Sohrabi et al. 2015 paper.

def calculate_sodi(precip, et, pet, vsm, ro, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the soil moisture drought index (SODI) from a number of moisture variables.
    SODI index is on the same time scale as the input data.
    
    Full details on SODI can be found in Sohrabi et al. 2015: https://doi.org/10.1061/(ASCE)HE.1943-5584.0001213
    
    Inputs:
    :param precip: Input precipitation data (should be over 10+ years). Time x lat x lon format
    :param et: Input evapotranspiration data (should be the same units as precip; should be over 10+ years). Time x lat x lon format
    :param pet: Input potential evaporation data (should be the same units as precip; should be over 10+ years). Time x lat x lon format
    :param vsm: Input volumetric soil moisture (0 - 40 cm average) data (should be the same units as precip; should be over 10+ years). Time x lat x lon format
    :param ro: Input runoff data (should be the same units as precip; should be over 10+ years). Time x lat x lon format
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in precip/et/pet/...
    :param mask: Land-sea mask for the pet variable
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param sodi: The SODI drought index, has the same shape and size as precip/et/pet/...
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
    
    # In order to get SODI, moisture loss from the soil column is needed. This is assumed to be the ET - P
    L = et - precip
    
    # If P > ET, there is no moisture loss.
    L[precip > et] = 0

    # Initialize some variables
    T, I, J = precip.shape
    
    WP_percentile = 5
    FC_percentile = 95
    
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]
    months_climo = months[climo_index]
    days_climo   = days[climo_index]
    
    grow_ind = np.where( (months_climo >= 4) & (months_climo <= 10) )[0] # Percentiles are determined from growing season values.
    
    awc = np.ones((I, J)) * np.nan
    
    # Find the time stamps for a singular year
    ind = np.where(years == 1999)[0] # Note, any non-leap year will do
    one_year = dates[ind]
    
    # Reshape data in 2D arrays
    vsm2d = vsm.reshape(T, I*J, order = 'F')
    awc2d = awc.reshape(I*J, order = 'F')
    
    for ij in range(I*J):
        # First determine the wilting point and field capacity. This is done by examining 5th and 95th percentiles.
        vsm_wp = stats.scoreatpercentile(vsm2d[grow_ind,ij], WP_percentile)
        
        vsm_fc = stats.scoreatpercentile(vsm2d[grow_ind,ij], FC_percentile)
        
        # The available water content is simply the difference between field capacity and wilting point
        awc2d[ij] = vsm_fc - vsm_wp
        
    # Convert AWC back to 2D data
    awc = awc2d.reshape(I, J, order = 'F')
    
    
    # The soil moisture deficiency then becomes the difference between AWC and soil moisutre
    smd = np.ones((T, I, J)) * np.nan
    for t in range(T):
        smd[t,:,:] = awc[:,:] - vsm[t,:,:]
        
    
    # Note to get the volumetric water content in fractional form, it is the mass of water lost divided by rho_l (to convert to volume), divided by sample volume.
    # To invert this, multiply this by rho_l to get the mass of water in a volume of soil, then multiply by soil depth to get the mass of water in an area of soil.
    ### Note: This is primarily done to bring the soil moisture variable (SMD) to the same units as the other variables (kg m^-2). I.e., it is done for unit consistency.
    soil_depth = 0.4  # m
    rho_l     = 1000 # kg m^-3

    smd = smd * soil_depth * rho_l

    # Next, calculate the moisture deficit given in equation 1 of Sohrabi et al. 2015.
    D = np.ones((T, I, J)) * np.nan

    for t in range(12, T): # Use a monthly average for variables in the previous month
        D[t,:,:] = (precip[t,:,:] + L[t,:,:] + np.nanmean(ro[t-12:t-6,:,:], axis = 0)) - (pet[t,:,:] + np.nanmean(smd[t-12:t-6,:,:], axis = 0))

    # Next, perform the Box-Car transformation and standardize the data to create SODI, according to equations 5 and 6 in Sohrabi et al. 2015
    sodi = np.ones((T, I, J)) * np.nan

    sodi2d = sodi.reshape(T, I*J, order = 'F')
    D2d    = D.reshape(T, I*J, order = 'F')
    
    
    for ij in range(I*J):
        
        # Estimate the lambda1 parameter for the climatological period
        if np.nanmin(D2d[climo_index,ij]) < 0: # Ensure the shift is positive
            lambda2 = -1 * np.nanmin(D2d[climo_index,ij])
    
        else:
            lambda2 = np.nanmin(D2d[climo_index,ij])
            
        y, lambda1 = stats.boxcox(D2d[climo_index,ij] + lambda2 + 0.001)
        
        
        # From looking around at various features, it seems as if lambda2 in the Box-Car transformation is the minimum value of the data, so that all values are > 0.
        if np.nanmin(D2d[:,ij]) < 0: # Ensure the shift is positive
            lambda2 = -1 * np.nanmin(D2d[:,ij])
    
        else:
            lambda2 = np.nanmin(D2d[:,ij])
        
        # Perform the Box-Car transformation. Note boxcar only accepts a vector, so this has to be done for 1 grid point at a time
        y = stats.boxcox(D2d[:,ij] + lambda2 + 0.001, lmbda = lambda1) # 0.001 should have a small impact on values, but ensure D + lambda2 is not 0 at any point
        
        # Collect the climatology
        y_climo = collect_climatology(y, dates, start_year = start_year, end_year = end_year)
        
        # Determine the climatology of the transformed data
        y_mean, y_std = calculate_climatology(y_climo, pentad = True)
        
        # Standardize the transformed data to calculate SODI for the grid point
        for n, date in enumerate(one_year):
            ind = np.where( (date.month == months) & (date.day == days) )[0]
            
            for t in ind:
                sodi2d[t,ij] = (y[t] - y_mean[n])/y_std[n]
                
    
    # Transform the data back into a 3D array
    sodi = sodi2d.reshape(T, I, J, order = 'F')

    sodi = np.where(sodi > 5, 5, sodi)
    sodi = np.where(sodi < -5, -5, sodi)

    # Remove any sea data points
    sodi = apply_mask(sodi, mask)
    # sodi[mask[:,:] == 0] = np.nan
    
    return sodi
    

#%%
##############################################

# Create functions to calculate FDII
# Details for FDII can be found in the Otkin et al. 2021 paper.

def calculate_fdii(vsm, dates, mask, start_year = 1990, end_year = 2020, years = None, months = None, days = None):
    '''
    Calculate the flash drought intensity index (FDII) from a number of moisture variables.
    FDII index is on the same time scale as the input data.
    
    Full details on FDII can be found in Otkin et al. 2021: https://doi.org/10.3390/atmos12060741
    
    Inputs:
    :param vsm: Input volumetric soil moisture (0 - 40 cm average) data (should be over 10+ years). Time x lat x lon format
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in vsm
    :param mask: Land-sea mask for the pet variable
    :param start_year: The start year in the climatological period used
    :param end_year: The last year in the climatological period used
    :param years: Array of intergers corresponding to the dates.year. If None, it is made from dates
    :param months: Array of intergers corresponding to the dates.month. If None, it is made from dates
    :param days: Array of intergers corresponding to the dates.day. If None, it is made from dates

    Outputs:
    :param fdii: The FDII drought index, has the same shape and size as vsm
    :param fd_int: The strength of the rapid intensification of the flash drought, has the same shape and size as vsm
    :param dro_sev: Severity of the drought component of the flash drought, has the same shape and size as vsm
    '''
    
    # Make the years, months, and/or days variables?
    if years == None:
        years = np.array([date.year for date in dates])
        
    if months == None:
        months = np.array([date.month for date in dates])
        
    if days == None:
        days = np.array([date.day for date in dates])
       
    print('Initializing some variables')
    # Define some base constants
    PER_BASE = 15 # Minimum percentile drop in 4 pentads
    T_BASE   = 4
    DRO_BASE = 20 # Percentiles must be below the 20th percentile to be in drought
    
    # Next, FDII can be calculated with the standardized soil moisture, or percentiles.
    # Use percentiles for consistancy with Otkin et al. 2021
    T, I, J = vsm.shape
    sm_percentile = np.ones((T, I, J))* np.nan
    
    # Reorder some data
    sm2d = vsm.reshape(T, I*J, order = 'F')
    sm_per2d = sm_percentile.reshape(T, I*J, order = 'F')
    
    # Get the climatology months
    climo_index = np.where( (years >= start_year) & (years <= end_year) )[0]

    # Determine soil moisture percentiles
    for ij in range(I*J):
        for t in range(T):
            ind = np.where( (days[t] == days[climo_index]) & (months[t] == months[climo_index]) )[0]
        
            sm_per2d[t,ij] = stats.percentileofscore(sm2d[ind,ij], sm2d[t,ij])
    
    
    print(np.nanmin(sm_per2d), np.nanmax(sm_per2d))
    print(np.nanmean(sm_per2d))
    
    print('Calculating rapid intensification of flash drought')
    # Determine the rapid intensification based on percentile changes based on equation 1 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    fd_int = np.ones((T, I, J)) * np.nan
    fd_int2d = fd_int.reshape(T, I*J, order = 'F')

    # Determine the intensification index
    for ij in range(I*J):
        for t in range(T-2): # Note the last two days are excluded as there is no change to examine
        
            obs = np.ones((9)) * np.nan # Note, the method detailed in Otkin et al. 2021 involves looking ahead 2 to 10 pentads (9 entries total)
            for npend in np.arange(2, 10+1, 1):
                npend = int(npend)
                if (t+npend) >= T: # If t + npend is in the future (beyond the dataset), break the loop and use NaNs for obs instead
                    break          # This should not effect results as this will only occur in November to December, outside of the growing season.
                else:
                    obs[npend-2] = (sm_per2d[t+npend,ij] - sm_per2d[t,ij])/npend # Note npend is the number of pentads the system is corrently looking ahead to.
            
            # If the maximum change in percentiles is less than the base change requirement (15 percentiles in 4 pentads), set FD_INT to 0.
            #  Otherwise, determine FD_INT according to eq. 1 in Otkin et al. 2021
            if np.nanmax(obs) < (PER_BASE/T_BASE):
                fd_int2d[t,ij] = 0
            else:
                fd_int2d[t,ij] = ((PER_BASE/T_BASE)**(-1)) * np.nanmax(obs)
                
    
    
    
    print('Calculating drought severity')
    # Next determine the drought severity component using equation 2 in Otkin et al. 2021 (and detailed in section 2.2 of the same paper)
    dro_sev = np.ones((T, I, J)) * np.nan
    dro_sev2d = dro_sev.reshape(T, I*J, order = 'F')

    dro_sev2d[0,:] = 0 # Initialize the first entry to 0, since there is no rapid intensification before it

    for ij in range(I*J):
        for t in range(1, T-1):
            if (fd_int2d[t,ij] > 0):
                
                dro_sum = 0
                for npent in np.arange(0, 18+1, 1): # In Otkin et al. 2021, the DRO_SEV can look up to 18 pentads (90 days) in the future for its calculation
                    
                    if (t+npent) >= T:      # For simplicity, set DRO_SEV to 0 when near the end of the dataset (this should not impact anything as it is not in
                        dro_sev2d[t,ij] = 0 # the growing season)
                        break
                    else:
                        dro_sum = dro_sum + (DRO_BASE - sm_per2d[t+npent,ij])
                        
                        if sm_per2d[t+npent,ij] > DRO_BASE: # Terminate the summation and calculate DRO_SEV if SM is no longer below the base percentile for drought
                            if npent < 4:
                                # DRO_SEV is set to 0 if drought was not consistent for at least 4 pentads after rapid intensificaiton (i.e., little to no impact)
                                dro_sev2d[t,ij] = 0
                                break
                            else:
                                dro_sev2d[t,ij] = dro_sum/npent # Terminate the loop and determine the drought severity if the drought condition is broken
                                break
                            
                        elif (npent >= 18): # Calculate the drought severity of the loop goes out 90 days, but the drought does not end
                            dro_sev2d[t,ij] = dro_sum/npent
                            break
                        else:
                            pass
            
            # In continuing consistency with Otkin et al. 2021, if the pentad does not immediately follow rapid intensification, drought is set 0
            else:
                dro_sev2d[t,ij] = 0
                continue
    
    
    
    print('Calculating FDII')
    # Reorder the data back into 3D data
    fd_int  = fd_int2d.reshape(T, I, J, order = 'F')
    dro_sev = dro_sev2d.reshape(T, I, J, order = 'F')
    
    # Finally, FDII is the product of the components
    fdii = fd_int * dro_sev
    
    # Remove any sea data points
    fd_int = apply_mask(fd_int, mask)
    dro_sev = apply_mask(dro_sev, mask)
    fdii = apply_mask(fdii, mask)
    
    # fd_int[mask[:,:] == 0] = np.nan
    # dro_sev[mask[:,:] == 0] = np.nan
    # fdii[mask[:,:] == 0] = np.nan
    
    print('Done')
    
    return fdii, fd_int, dro_sev


#%%
##############################################

# Functions to create test figures

def display_histogram(data, data_name, path = './Figures'):
    '''
    Display a histogram of the data
    
    Inputs:
    :param data: Data whose histogram is being examined
    :param data_name: String name of the data being plotted
    :param path: Path the data will be saved to
    '''
    
    fig = plt.figure(figsize = [12, 10])
    ax = fig.add_subplot(1, 1, 1)
    
    # Set the title
    ax.set_title('Histogram of %s Across the entire grid for all times'%(data_name), fontsize = 18)
    
    # Plot the histogram
    ax.hist(data.flatten(), bins = 100)
    
    # Set the labels
    ax.set_xlabel(data_name, fontsize = 18)
    ax.set_ylabel('Frequency', fontsize = 18)
    
    # Set the limits of -5 to 5 (for most indices)
    if np.nanmin(data) == 0: # Exception for FDII, which can go from 0 to 70+
        ax.set_xlim([0, np.ceil(np.nanmax(data))])
    elif (data_name == 'smi') | (data_name == 'sedi'): 
        ax.set_xlim([-10, 10])
    else:
        ax.set_xlim([-5, 5])
    
    # Save this example
    plt.savefig('%s/%s_histogram_example.png'%(path,data_name))


def display_maximum_map(data, lat, lon, dates, examine_start, examine_end, data_name, path = './Figures', years = None, months = None, days = None):
    '''
    Display a map of the maximum value of input data over a specified period
    
    Inputs:
    :param data: data to be plotted
    :param lat: Gridded latitude values corresponding to data
    :param lon: Gridded longitude values corresponding to data
    :param dates: Vector of datetimes corresponding to the timestamp in each timestep in vsm
    :param examine_start: Datetime of the beginning of the period to be examined
    :param examine_end: Datetime of the end of the period to be examined
    :param data_name: String name of the data being plotted
    :param path: Path the figure will be saved to
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

    # Determine the period to examine
    ind = np.where( (years == examine_start.year) & (months >= examine_start.month) & (months <= examine_end.month) )[0]


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
    if (np.ceil(np.nanmin(data[ind,:,:])) == 1) & (np.floor(np.nanmax(data[ind,:,:])) == 0): # Special case if the variable varies from 0 to 1
        cmin = np.round(np.nanmin(data[ind,:,:]), 2); cmax = np.round(np.nanmax(data[ind,:,:]), 2); cint = (cmax - cmin)/100
    else:
        cmin = np.ceil(np.nanmin(data[ind,:,:])); cmax = np.floor(np.nanmax(data[ind,:,:])); cint = (cmax - cmin)/100
        
    clevs = np.arange(cmin, cmax+cint, cint)
    nlevs = len(clevs) - 1
    cmap  = plt.get_cmap(name = 'gist_rainbow_r', lut = nlevs)
    
    data_proj = ccrs.PlateCarree()
    fig_proj  = ccrs.PlateCarree()



    # Create the figure
    fig = plt.figure(figsize = [12, 16])
    ax = fig.add_subplot(1, 1, 1, projection = fig_proj)
    
    # Set title
    ax.set_title('%s for %s - %s'%(data_name, examine_start.strftime('%Y-%m-%d'), examine_end.strftime('%Y-%m-%d')), fontsize = 16)
    
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
    cs = ax.contourf(lon, lat, np.nanmax(data[ind,:,:], axis = 0), levels = clevs, cmap = cmap,
                     transform = data_proj, extend = 'both', zorder = 1)
    
    # Create and set the colorbar
    cbax = fig.add_axes([0.92, 0.375, 0.02, 0.25])
    cbar = fig.colorbar(cs, cax = cbax)
    
    # Set the extent
    ax.set_extent([-130, -65, 25, 50], crs = fig_proj)
    
    # Save the figure
    plt.savefig('%s/test_%s_max_map.png'%(path, data_name))
    
    plt.show(block = False)




