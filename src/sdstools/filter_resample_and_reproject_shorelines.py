"""
Mark Lundine
This script takes the CoastSeg matrix and resamples it the time domain and then reprojects new shorelines.
"""
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import analysis
import datetime
import shapely
import timeseries_filter
import timeseries_resample

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def simplify_lines(shorelines_path, tolerance=1):
    """
    Uses shapely simplify function to smooth out the extracted shorelines
    inputs:
    shapefile: path to extracted shorelines
    tolerance (optional): simplification tolerance
    outputs:
    save_path: path to simplified shorelines
    """

    save_path = os.path.splitext(shorelines_path)[0]+'_simplify'+str(tolerance)+'.geojson'
    lines = gpd.read_file(shorelines_path)
    lines['geometry'] = lines['geometry'].simplify(tolerance)
    lines.to_file(save_path)
    return save_path

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs: line
    outputs: array of xy tuples
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def chaikins_corner_cutting(coords, refinements=5):
    """
    Smooths out lines or polygons with Chaikin's method
    """
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def smooth_lines(shorelines,refinements=5):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    shorelines (str): path to extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    save_path (str): path of output file in UTM
    """
    dirname = os.path.dirname(shorelines)
    save_path = os.path.join(dirname,os.path.splitext(os.path.basename(shorelines))[0]+'_smooth.geojson')
    lines = gpd.read_file(shorelines)
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line.geometry)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines.to_file(save_path)
    return save_path

def wgs84_to_utm_file(geojson_file):
    """
    Converts wgs84 to UTM
    inputs:
    geojson_file (path): path to a geojson in wgs84
    outputs:
    geojson_file_utm (path): path to a geojson in utm
    """

    geojson_file_utm = os.path.splitext(geojson_file)[0]+'_utm.geojson'

    gdf_wgs84 = gpd.read_file(geojson_file)
    utm_crs = gdf_wgs84.estimate_utm_crs()

    gdf_utm = gdf_wgs84.to_crs(utm_crs)
    gdf_utm.to_file(geojson_file_utm)
    return geojson_file_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84

def transect_timeseries_to_wgs84(transect_timeseries_merged_path,
                                 config_gdf_path,
                                 savename_lines,
                                 savename_points):
    """
    Takes merged transect timeseries path and outputs new shoreline lines and points files
    inputs:
    transect_timeseries_merged_path (str): path to the transect_timeseries_merged.csv
    config_gdf_path (str): path to the the config_gdf as geojson
    savename_lines (str): basename of the output lines ('..._lines.geojson')
    savename_points (str): basename of the output points ('...._points.geojson')
    """
    ##Load in data, make some new paths
    timeseries_data = pd.read_csv(transect_timeseries_merged_path).dropna().reset_index(drop=True)
    timeseries_data = timeseries_data.sort_values('transect_id', ascending=True).reset_index(drop=True)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'], utc=True)
    timeseries_data['transect_id'] = timeseries_data['transect_id'].astype(int)
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf[config_gdf['type']=='transect'].reset_index(drop=True)
    transects['transect_id'] = transects['id']

    transects = transects.sort_values('transect_id', ascending=True).reset_index(drop=True)

    ##save paths
    new_gdf_shorelines_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_lines)    
    points_wgs84_path = os.path.join(os.path.dirname(transect_timeseries_merged_path), savename_points)
    
    ##Gonna do this in UTM to keep the math simple...problems when we get to longer distances (10s of km)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)

    ##need some placeholders
    shore_x_vals = [None]*len(timeseries_data)
    shore_y_vals = [None]*len(timeseries_data)
    timeseries_data['shore_x'] = shore_x_vals
    timeseries_data['shore_y'] = shore_y_vals
    
    ##make an empty gdf to hold points
    size = len(timeseries_data)
    transect_ids = [None]*size
    dates = [None]*size
    points = [None]*size
    points_gdf_utm = gpd.GeoDataFrame({'geometry':points,
                                      'dates':dates,
                                      'id':transect_ids},
                                      crs=utm_crs)
    
    ##loop over all transects
    for i in range(len(transects_utm)):
        transect = transects_utm.iloc[i]
        transect_id = transect['transect_id']
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]
        idx = timeseries_data.index[timeseries_data['transect_id'] == transect_id].tolist()
        ##in case there is a transect in the config_gdf that doesn't have any intersections
        ##skip that transect
        if np.any(idx):
            timeseries_data_filter = timeseries_data.iloc[idx]
        else:
            print('skip')
            continue

        idxes = timeseries_data_filter.index
        distances = timeseries_data_filter['cross_distance']
        angle = np.arctan2(last[1] - first[1], last[0] - first[0])

        shore_x_utm = first[0]+distances*np.cos(angle)
        shore_y_utm = first[1]+distances*np.sin(angle)
        points_utm = [shapely.Point(xy) for xy in zip(shore_x_utm, shore_y_utm)]

        #conversion from utm to wgs84, put them in the transect_timeseries csv and utm gdf
        dummy_gdf_utm = gpd.GeoDataFrame({'geometry':points_utm},
                                         crs=utm_crs)
        dummy_gdf_wgs84 = dummy_gdf_utm.to_crs(org_crs)

        points_wgs84 = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84.geometry]
        points_wgs84 = np.array(points_wgs84)
        points_wgs84 = points_wgs84.reshape(len(points_wgs84),2)
        x_wgs84 = points_wgs84[:,0]
        y_wgs84 = points_wgs84[:,1]
        timeseries_data.loc[idxes,'shore_x'] = x_wgs84
        timeseries_data.loc[idxes,'shore_y'] = y_wgs84
        dates = timeseries_data['dates'].loc[idxes]
        points_gdf_utm.loc[idxes,'geometry'] = points_utm
        points_gdf_utm.loc[idxes,'dates'] = dates
        points_gdf_utm.loc[idxes,'id'] = [transect_id]*len(dates)
        
    ##get points as wgs84 gdf
    points_gdf_wgs84 = points_gdf_utm.to_crs(org_crs)
    points_gdf_wgs84 = points_gdf_wgs84.mask(points_gdf_wgs84.eq('None')).dropna().reset_index(drop=True)

    ##Need to loop over unique dates to make shoreline gdf from points
    new_dates = np.unique([points_gdf_wgs84['dates']])
    new_lines = [None]*len(np.unique(new_dates))
    for i in range(len(new_lines)):
        date = new_dates[i]
        points_filter = points_gdf_wgs84[points_gdf_wgs84['dates']==date]
        if len(points_filter)<2:
            continue
        new_line = shapely.LineString(points_filter['geometry'])
        new_lines[i] = new_line
        new_dates[i] = date
    
    new_gdf_shorelines_wgs84 = gpd.GeoDataFrame({'dates':new_dates,
                                                 'geometry':new_lines},
                                                crs=org_crs)
    new_gdf_shorelines_wgs84 = new_gdf_shorelines_wgs84.mask(new_gdf_shorelines_wgs84.eq('None')).dropna().reset_index(drop=True)

    ##convert to utm, save wgs84 and utm geojsons
    new_gdf_shorelines_wgs84.to_file(new_gdf_shorelines_wgs84_path)
    points_gdf_wgs84.to_file(points_wgs84_path)
    
def main(transect_timeseries_path,
         transects_path,
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
                                              utc=True,
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
        data = pd.DataFrame({'dates':dates,
                             'cross_distance':select_timeseries})
        filtered_data = timeseries_filter.hampel_filter_loop(df, hampel_window=5, hampel_sigma=3)
        filtered_data = timeseries_filter.change_filter_loop(filtered_data, iterations=1, q=0.75)
        filtered_data = timeseries_resample.resample_timeseries(filtered_data, timedelta)
        filtered_data = timeseries_resample.fill_nans(filtered_data)
        output_df = output_df.rename(columns = {'cross_distance':transect_id})
        timeseries_dfs[i] = output_df
        timedeltas[i] = timedelta

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
    save_csv_path = os.path.join(output_folder, 'timeseries_resample_time_stacked.csv')
    savename_lines = os.path.join(output_folder, 'reprojected_lines.geojson')
    savename_points = os.path.join(output_folder, 'reprojected_points.geojson')
    stacked.to_csv(save_csv_path)
    transect_timeseries_to_wgs84(save_csv_path,
                                 config_gdf_path,
                                 savename_lines,
                                 savename_points)
        
        
        
        





        
