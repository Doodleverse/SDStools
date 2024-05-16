## Script to download ERA5 daily data from https://cds.climate.copernicus.eu
## uses https://pypi.org/project/cdsapi/ and assumes the API and credentials are stored in file
## follow the instructions at https://pypi.org/project/cdsapi/ for how to obtain and store credentials

## Downloads combined sea/swell Hs, mean T, peak T, and mean D
## computes wave setup and wave power assuming deep water
## written by Dr Daniel Buscombe, April-May, 2024

## Example usage, from cmd:
## python download_era5_dataframe_grid.py -f "my_location" -a 1984 -b 2023 -x -160.8052  -y 64.446

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

def wave_energy(Hs):
    """return wave energy based on deep water wave condition"""
    rho = 1025
    g = 9.81
    E = (1/8)*rho*g*(Hs**2)
    return E
