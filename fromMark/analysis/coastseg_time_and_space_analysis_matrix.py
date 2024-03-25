"""
Mark Lundine
This script is in progress.
The goal is to take the CoastSeg matrix and resample it intelligently.
in the time and space domain so that it is equally spaced temporally and spatially.
This will allow for predictive modeling from satellite shoreline data obtained from CoastSeg.
"""
import os
import shoreline_timeseries_analysis_single as stas
import shoreline_timeseries_analysis_single_spatial as stasp

def main(transect_timeseries_path,
         config_gdf,
         output_folder,
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
    which_timedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is what the timeseries is resampled at
    which_spacedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is the matrix is sampled at in the longshore direction
    timedelta (str, optional): the custom time spacing (e.g., '30D' is 30 days)
    beware of choosing minimum, with a mix of satellites, the minimum time spacing can be so low that you run into fourier transform problems
    spacedelta (int, optional): custom longshore spacing, do not make this finer than the input transect spacing!!!!
    """

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
            i=i+1
            continue

        transect_ids[i] = transect_id
        
        ##Timeseries processing
        data = pd.DataFrame({'date':dates,
                             'position':select_timeseries})
        timeseries_analysis_result, output_df, new_timedelta = stas.main_df(df,
                                                                            output_folder,
                                                                            transect_id,
                                                                            which_timedelta,
                                                                            timedelta=timedelta)
        output_df = output_df.set_index(['date'])
        output_df = output_df.rename(columns = {'position':transect_id})
        timeseries_dfs[i] = output_df
        timedeltas[i] = new_timedelta

    ##Remove Nones in case there were transects in config_gdf with no timeseries data
    transect_ids = [ele for ele in transect_ids if ele is not None]
    timeseries_dfs = [ele for ele in transect_ids if ele is not None]
    timedeltas = [ele for ele in transect_ids if ele is not None]
    
    ##Make new matrix 
    new_matrix = pd.concat(timeseries_dfs, 1)

    ##Loop over time
    datetimes = new_matrix.index
    space_series_dfs = [None]*len(datetimes)
    spacedeltas = [None]*len(datetimes)
    for j in range(len(datetimes)):
        date = datetimes[j]
        try:
            select_timeseries = np.array(new_matrix.loc[datetime])
        except:
            i=i+1
            continue
        
        ##space series processing
        data = pd.DataFrame({'transect_id':transect_ids,
                             'position':select_timeseries})
        space_series_analysis_result, output_df, new_spacedelta = stasp.main_df(df,
                                                                                output_folder,
                                                                                'timestep'+str(i),
                                                                                transect_spacing,
                                                                                which_spacedelta,
                                                                                spacedelta=spacedelta)
        output_df.set_index(['longshore_position'])
        output_df = output_df.rename(columns = {'date':date})
        space_series_dfs[i] = output_df
        spacedeltas[i] = new_spacedelta


    ##Remove nones in case there were times with no data
    space_series_dfs = [ele for ele in transect_ids if ele is not None]
    spacedeltas = [ele for ele in transect_ids if ele is not None]
    new_matrix = pd.concat(space_series_dfs,1)
    new_matrix_path = os.path.join(output_folder, 'timeseries_mat_resample.csv')
    new_matrix.to_csv(new_matrix_path)

    return new_matrix_path
    
        
        
        
        
        
        





        
