
## Script to download ERA5 daily data from https://cds.climate.copernicus.eu
## uses https://pypi.org/project/cdsapi/ and assumes the API and credentials are stored in file
## follow the instructions at https://pypi.org/project/cdsapi/ for how to obtain and store credentials

## Downloads combined sea/swell Hs, mean T, peak T, and mean D
## computes wave setup and wave power assuming deep water
## written by Dr Daniel Buscombe, April-May, 2024

## Example usage, from cmd:
## python download_era5_dataframe_singlelocation.py -f "my_location" -a 1984 -b 2023 -x -160.8052  -y 64.446

import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

import argparse
import warnings
warnings.filterwarnings("ignore")

#==========================================
def compute_setup_deep(H,T):
    """return setup (m) based on deep water wave condition"""
    g = 9.81
    L = (g*T**2) / (2*np.pi)
    return 0.016*(np.sqrt(H*L))

def wavepower_deep(Hs, Tp):
    """return wave power (kW/m) based on deep water wave condition"""
    rho = 1025
    g = 9.81
    P = (rho*(g**2)/(64*np.pi))*Tp*(Hs**2)
    P_kW_m = P /1000
    return P_kW_m

def wave_energy_deep(Hs):
    """return wave energy based on deep water wave condition"""
    rho = 1025
    g = 9.81
    E = (1/8)*rho*g*(Hs**2) #[J/m2]
    return E


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the wave data download script.
    Arguments and their defaults are defined within the function.
    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to download ERA5 wave data at a single location.")
    
    parser.add_argument(
        "-f",
        "-f",
        dest="fileprefix",
        type=str,
        required=True,
        help="Set the prefix for output files.",
    )
    parser.add_argument(
        "-X",
        "-x",
        dest="lon",
        type=float,
        required=True,
        help="Set the Longitude.",
    )
    parser.add_argument(
        "-Y",
        "-y",
        dest="lat",
        type=float,
        required=True,
        help="Set the Latitude.",
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
    return parser.parse_args()

##==========================================
def main():
    args = parse_arguments()
    lon = args.lon
    lat = args.lat
    start_year = args.start_year
    end_year = args.end_year
    fileprefix = args.fileprefix
    print(f"Data downloading.... Latitude: {lat}, Longitude: {lon}, from {start_year} to {end_year}")

    # # initialise cdsapi
    c = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels"
    offset = 0.001

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
            'grid': [0.25, 0.25],
            'area': [lat+offset, lon-offset, lat-offset, lon+offset], 
            }  
            
        # retrieves the path to the file
        fl = c.retrieve(dataset, params)

        # download the file 
        fl.download(f"./{fileprefix}_{dataset}_{v}.nc")

    ## we want a pandas dataframe to write to csv, so we'll make a dictionary of variables
    df_dict={}

    ##merge all nc data into a single csv
    # loop thru each variable and read netcdf file contents
    for v in ['significant_height_of_combined_wind_waves_and_swell',
        'mean_wave_period',
        'peak_wave_period',
        'mean_wave_direction']:
        ## open dataset as xarray
        data = xr.open_dataset(f'./{fileprefix}_{dataset}_{v}.nc')

        if v == 'significant_height_of_combined_wind_waves_and_swell':
            df_dict['date']=data.time.values.squeeze()
            try:
                df_dict['swh']=np.nanmedian(data.swh.values.squeeze(),axis=1)
            except:
                df_dict['swh']=data.swh.values.squeeze()
        elif v == 'mean_wave_period':
            try:
                df_dict['mwp']=np.nanmedian(data.mwp.values.squeeze(),axis=1)
            except:
                df_dict['mwp']=data.mwp.values.squeeze()
        elif v == 'peak_wave_period':
            try:
                df_dict['pp1d']=np.nanmedian(data.pp1d.values.squeeze(),axis=1)
            except:
                df_dict['pp1d']=data.pp1d.values.squeeze()
        elif v == 'mean_wave_direction':
            try:
                df_dict['mwd']=np.nanmedian(data.mwd.values.squeeze(),axis=1)
            except:
                df_dict['mwd']=data.mwd.values.squeeze()
    df_dict['wp']=wavepower_deep(df_dict['swh'], df_dict['pp1d'])
    df_dict['ws']=compute_setup_deep(df_dict['swh'], df_dict['pp1d'])
    df_dict['we']=wave_energy_deep(df_dict['swh'])

    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(f'{fileprefix}_{dataset}_swh_mwp_pp1d_mwd_wp_we_{lon}_{lat}_{start_year}_{end_year}.csv')


    #fig=plt.figure(figsize=(8,8))
    sns.jointplot(x=df['swh'], y=df['mwd'], kind="hex", color="#4CB391")
    plt.xlabel('Significant Wave Height (m)')
    plt.ylabel('Mean Wave Direction (degrees)')
    #plt.show()
    plt.savefig(f'{fileprefix}_{dataset}_joint_and_marginal_distributions_Hs_Dmean.png',dpi=300, bbox_inches='tight')
    plt.close()

    #fig=plt.figure(figsize=(8,8))
    sns.jointplot(x=df['swh'], y=df['mwp'], kind="hex", color="#4CB391")
    plt.xlabel('Significant Wave Height (m)')
    plt.ylabel('Mean Wave Period (s)')
    #plt.show()
    plt.savefig(f'{fileprefix}_{dataset}_joint_and_marginal_distributions_Hs_Tmean.png',dpi=300, bbox_inches='tight')
    plt.close()

    #fig=plt.figure(figsize=(8,8))
    sns.jointplot(x=df['swh'], y=df['pp1d'], kind="hex", color="#4CB391")
    plt.xlabel('Significant Wave Height (m)')
    plt.ylabel('Peak Wave Period (s)')
    #plt.show()
    plt.savefig(f'{fileprefix}_{dataset}_joint_and_marginal_distributions_Hs_Tpeak.png',dpi=300, bbox_inches='tight')
    plt.close()


    #plot sales by date
    plt.plot_date(df['date'].apply(pd.Timestamp), df.swh, 'k-')
    #rotate x-axis ticks 45 degrees and right-aline
    plt.xticks(rotation=45, ha='right')
    # plt.axvline(dt.datetime(2011, 8, 28),color='r')
    #plt.show()
    plt.savefig(f'{fileprefix}_{dataset}_timeseries_Hs.png',dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
