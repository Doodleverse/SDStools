"""
Mark Lundine
This script takes the CoastSeg matrix and resamples it intelligently
in the time and space domain so that it is equally spaced temporally and spatially.
This will allow for predictive modeling from satellite shoreline data obtained from CoastSeg.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import shoreline_timeseries_analysis_single as stas
import shoreline_timeseries_analysis_single_spatial as stasp

def main(transect_timeseries_path,
         config_gdf_path,
         output_folder,
         median_filter_window,
         transect_spacing,
         which_timedelta,
         which_spacedelta,
         timedelta=None,
         spacedelta=None):
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
    space_dir = os.path.join(output_folder, 'space')
    dirs = [time_dir, space_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'],
                                              format='%Y-%m-%d %H:%M:%S+00:00')
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf[config_gdf['type']=='transect']

    ##Loop over transects (space)
    transect_ids = [None]*len(transects)
    timeseries_dfs = [None]*len(transects)
    timedeltas = [None]*len(transects)
    for i in range(len(transects)):
        transect_id = transects['id'].iloc[i]
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
                                                                            median_filter_window,
                                                                            which_timedelta,
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
    

    ##Loop over time
    datetimes = new_matrix.index
    space_series_dfs = [None]*len(datetimes)
    spacedeltas = [None]*len(datetimes)
    for j in range(len(datetimes)):
        date = datetimes[j]
        try:
            select_timeseries = np.array(new_matrix.loc[date])
        except:
            continue
        
        ##space series processing
        data = pd.DataFrame({'transect_id':transect_ids,
                             'position':select_timeseries})
        space_series_analysis_result, output_df, new_spacedelta = stasp.main_df(data,
                                                                                space_dir,
                                                                                'timestep'+str(j),
                                                                                transect_spacing,
                                                                                which_spacedelta,
                                                                                spacedelta=spacedelta)
        output_df.set_index(['longshore_position'])
        output_df = output_df.rename(columns = {'position':date})
        output_df = output_df.drop(columns =['longshore_position'])
        space_series_dfs[j] = output_df
        spacedeltas[j] = new_spacedelta

    ##Remove nones in case there were times with no data
    space_series_dfs = [ele for ele in space_series_dfs if ele is not None]
    spacedeltas = [ele for ele in spacedeltas if ele is not None]
    new_matrix = pd.concat(space_series_dfs,1)
    new_matrix = new_matrix.T
    new_matrix.index.name = 'date'
    longshore_index_mat_path = os.path.join(output_folder, 'timeseries_mat_resample_time_space_longshore_index.csv')
    new_matrix.to_csv(longshore_index_mat_path)
    new_matrix.columns = transect_ids
    new_matrix_path = os.path.join(output_folder, 'timeseries_mat_resample_time_space.csv')
    new_matrix.to_csv(new_matrix_path)
    return new_matrix_path
        
        
        
        





        
