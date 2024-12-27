"""
Mark Lundine
This script takes the CoastSeg matrix and resamples it the time domain.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import analysis

    
def main(transect_timeseries_path,
         config_gdf_path,
         output_folder,
         transect_spacing,
         which_timedelta,
         which_spacedelta,
         median_filter_window=3,
         timedelta=None):
    """
    Performs timeseries and spatial series analysis cookbook on each
    transect in the transect_time_series matrix from CoastSeg
    inputs:
    transect_timeseries_path (str): path to the transect_time_series.csv
    config_gdf_path (str): path to the config_gdf.geojson
    output_folder (str): path to save outputs to
    median_filter_window (odd int): kernel size for median filter on timeseries
    which_timedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is what the timeseries is resampled at
    which_spacedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is the matrix is sampled at in the longshore direction
    timedelta (str, optional): the custom time spacing (e.g., '30D' is 30 days)
    beware of choosing minimum, with a mix of satellites, the minimum time spacing can be so low that you run into fourier transform problems
    spacedelta (int, optional): custom longshore spacing, do not make this finer than the input transect spacing!!!!
    """
    ##make directories
    time_dir = os.path.join(output_folder, 'time')
    dirs = [time_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'],
                                              format='%Y-%m-%dT%H:%M:%S')
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf #config_gdf[config_gdf['type']=='transect']

    ##Loop over transects (space)
    transect_ids = [None]*len(transects)
    timeseries_dfs = [None]*len(transects)
    timedeltas = [None]*len(transects)
    for i in range(len(transects)):
        transect_id = str(transects['OBJECTID'].iloc[i])
        dates = timeseries_data['dates']
        try:
            select_timeseries = np.array(timeseries_data[transect_id])
        except:
            continue
        
        transect_ids[i] = transect_id
        
        ##Timeseries processing
        data = pd.DataFrame({'date':dates,
                             'position':select_timeseries})
        timeseries_analysis_result, output_df, new_timedelta = stas.main_df(data,
                                                                            time_dir,
                                                                            transect_id,
                                                                            which_timedelta,
                                                                            median_filter_window=median_filter_window,
                                                                            timedelta=timedelta)
        output_df = output_df.set_index(['date'])
        output_df = output_df.rename(columns = {'position':transect_id})
        timeseries_dfs[i] = output_df
        timedeltas[i] = new_timedelta

    ##Remove Nones in case there were transects in config_gdf with no timeseries data
    transect_ids = [ele for ele in transect_ids if ele is not None]
    timeseries_dfs = [ele for ele in timeseries_dfs if ele is not None]
    timedeltas = [ele for ele in timedeltas if ele is not None]

    ##Make new matrix 
    new_matrix = pd.concat(timeseries_dfs, 1)
    new_matrix.to_csv(os.path.join(output_folder, 'timeseries_mat_resample_time.csv'))
    new_matrix = pd.read_csv(os.path.join(output_folder, 'timeseries_mat_resample_time.csv'))

    ##Stacked csv
    stacked = new_matrix.melt(id_vars=['date'],
                              var_name='transect_id',
                              value_name='distances')
    stacked.to_csv(os.path.join(output_folder, 'timeseries_resample_time_stacked.csv'))
        
        
        
        





        
