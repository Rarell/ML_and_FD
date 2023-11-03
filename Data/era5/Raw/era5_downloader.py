import cdsapi

def downloader(variable, year):
    """
    API to download 1 year of hourly ERA5 data from cds.climate.copernicus.eu
    
    Inputs:
    :param variable: Str. Name of the variable to download
    :param variable: Year of the data to be downloaded
    
    Outputs:
    None. Data is downloaded into the current directory. 
    #### NOTE the data files are LARGE and require significant memory and time to download
         1 year of data ~17GB, and takes ~1 hour to download
    """
    # Create the downloader
    c = cdsapi.Client()

    # Retrieve data using the given information
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        'era5_raw_%d.nc'%(year))