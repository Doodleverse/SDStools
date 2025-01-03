## Script to download ERA5 daily data from https://cds.climate.copernicus.eu
## uses https://pypi.org/project/cdsapi/ and assumes the API and credentials are stored in file
## follow the instructions at https://pypi.org/project/cdsapi/ for how to obtain and store credentials

## Script to download ERA5 wave data over a grid defined by a geoJSON file input
## Downloads combined sea/swell Hs, mean T, peak T, and mean D
## computes wave setup and wave power assuming deep water
## written by Dr Daniel Buscombe, April-May, 2024

## Example usage, from cmd:
## python download_era5_dataframe_grid.py -i "my_location" -a 1984 -b 2023 -f geoJSON file -p 1

import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

import argparse
import warnings
warnings.filterwarnings("ignore")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the wave data download script.
    Arguments and their defaults are defined within the function.
    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to download ERA5 wave data over a grid defined by a geoJSON file input.")
    
    parser.add_argument(
        "-i",
        "-I",
        dest="fileprefix",
        type=str,
        required=True,
        help="Set the prefix for output files.",
    )
    parser.add_argument(
        "-F",
        "-f",
        dest="geofile",
        type=str,
        required=True,
        help="Set the geoJSON file.",
    )
    parser.add_argument(
        "-A",
        "-a",
        dest="start_year",
        type=int,
        required=True,
        help="Set the start year.",
    )
    parser.add_argument(
        "-B",
        "-b",
        dest="end_year",
        type=int,
        required=True,
        help="Set the end year.",
    )

    parser.add_argument(
        "-p",
        "-P",
        dest="doplot",
        type=int,
        required=False,
        default=0,
        help="1=make a plot, 0=no plot (default).",
    )


    return parser.parse_args()

##==========================================
def main():
    args = parse_arguments()
    start_year = args.start_year
    end_year = args.end_year
    fileprefix = args.fileprefix
    geofile = args.geofile
    doplot = args.doplot


    print(f"Data downloading....from {start_year} to {end_year}")

    gdf = gpd.read_file(geofile, driver='GeoJSON')

    minlon = float(gdf.bounds.minx.values)
    maxlon = float(gdf.bounds.maxx.values)
    minlat = float(gdf.bounds.miny.values)
    maxlat = float(gdf.bounds.maxy.values)

    # # initialise cdsapi
    c = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels"

    ## certain years
    years = np.arange(start_year,end_year)
    years = [str(i) for i in years]

    ## all months
    mnths = np.arange(1,13)
    mnths = [str(i) for i in mnths]

    ## all days
    days = np.arange(1,31)
    days = [str(i) for i in days]

    ## I chose 'daily' info at 12-noon
    times = ['12:00']

    for v in ['significant_height_of_combined_wind_waves_and_swell',
        'mean_wave_period',
        'peak_wave_period',
        'mean_wave_direction']:

        # api parameters 
        params = {
            'format': 'netcdf',
            'product_type': 'reanalysis',
            'variable': [
            v,
            ],
            'year': years,
            'month': mnths,    
            'day': days,
            'time': times,    
            'grid': [0.1, 0.1],
            'area': [maxlat, minlon, minlat, maxlon],  #[lat+offset, lon-offset, lat-offset, lon+offset], 
            }  
            
        # retrieves the path to the file
        fl = c.retrieve(dataset, params)

        # download the file 
        fl.download(f"./{fileprefix}_{dataset}_{v}.nc")



    if doplot==1:
        ##dostuff
            
        ##merge all nc data into a single csv
        # loop thru each variable and read netcdf file contents
        for counter,v in enumerate(['significant_height_of_combined_wind_waves_and_swell',
            'mean_wave_period',
            'peak_wave_period',
            'mean_wave_direction']):
            ## open dataset as xarray
            data = xr.open_dataset(f'./{fileprefix}_{dataset}_{v}.nc')

            if v == 'significant_height_of_combined_wind_waves_and_swell':
                plt.figure(figsize=(16,12))
                plt.subplot(2,2,counter+1)
                data.swh.mean(axis=0).plot()

            elif v == 'mean_wave_period':
                # df_dict['mwp']=data.mwp.values.squeeze()

                plt.subplot(2,2,counter+1)
                data.mwp.mean(axis=0).plot()

            elif v == 'peak_wave_period':
                # df_dict['pp1d']=data.pp1d.values.squeeze()

                plt.subplot(2,2,counter+1)
                data.pp1d.mean(axis=0).plot()

            elif v == 'mean_wave_direction':
                # df_dict['mwd']=data.mwd.values.squeeze()

                plt.subplot(2,2,counter+1)
                data.mwd.mean(axis=0).plot()
        #plt.show()
        plt.savefig(f'{fileprefix}_mean_2d.png',dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
