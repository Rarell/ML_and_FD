# Program to create a txt file for collecting the raw NLDAS data
# Once the txt file is created, run wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -i nldas_urls.txt


#### Script is technically depricated; this will generate urls to download grb files, but a way to extract the grib data and turn them into netcdf files has not yet be found

import numpy as np
import argparse
from datetime import datetime, timedelta

# Function to create a parser using the terminal
def create_parser():
    '''
    Create argument parser
    '''
    
   
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ERA Downloader', fromfile_prefix_chars='@')

    parser.add_argument('--test', action='store_true', help='Perform a test download (only retrieves 1 year of data)?')
    parser.add_argument('--years', type=int, nargs=2, default=[1979,2021], help='Beginning and ending years to download data for.')
    
    return parser


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create the time modifiers
    hours = np.arange(0,24)
    
    # Make URLs for a test set?
    if args.test:
        years = [2021]
    else:
        years = np.arange(args.years[0], args.years[1]+1)
        
        
    # Create the file
    file = open('./nldas_urls.txt', 'w')
    
    # Create the URLs based on the year, day, hour
    for year in years:
        # Number of days will change on leap years
        if np.mod(year, 4) == 0:
            days  = np.arange(1, 366+1)
        else:
            days  = np.arange(1, 365+1)
            
        for day in days:
            # There is no data for Jan. 1, 1979
            if (year == 1979) & (day == 1):
                continue
            
            # Determine the date
            date = datetime(year, 1, 1)
            date = date + timedelta(days = int(day)-1)
            date_str = '{:02d}{:02d}'.format(date.month, date.day)
            
            # Skip leap days
            if (date.month == 2) & (date.day == 29):
                continue
                
            for hour in hours:
                # There is not a grb file for Jan. 2, 1979 at 00Z
                if (year == 1979) & (day == 2) & (hour == 0):
                    continue
                
                # Write the URL name
                file.write('https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.002/{year}/{:03d}/NLDAS_NOAH0125_H.A{year}{:s}.{:02d}00.002.grb'.format(day, date_str, hour, year=year))
                file.write('\n')
    
    # Close the file
    file.close()
    
    
    ## Pages 17 - 19 of https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/README.NLDAS2.pdf
    ## for list of variables, soil depths, etc.