import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import shapely

def split_list_at_none(lst):
    # Initialize variables
    result = []
    temp = []

    # Iterate through the list
    for item in lst:
        if item is None:
            # Append the current sublist to the result and reset temp
            result.append(temp)
            temp = []
        else:
            # Add item to the current sublist
            temp.append(item)

    # Append the last sublist if not empty
    if temp:
        result.append(temp)

    return result

def remove_nones(my_list):
    new_list = [x for x in my_list if x is not None]
    return new_list

def lists_to_Polygon(list1, list2):
    new_list = [None]*len(list1)
    for i in range(len(list1)):
        new_list[i] = (list1[i], list2[i])
    points = [None]*len(new_list)
    for i in range(len(new_list)):
        x,y = new_list[i]
        point = shapely.geometry.Point(x,y)
        points[i] = point
    polygon = shapely.geometry.Polygon(points)
    return polygon

def lists_to_LineString(list1, list2):
    new_list = [None]*len(list1)
    for i in range(len(list1)):
        new_list[i] = (list1[i], list2[i])
    points = [None]*len(new_list)
    for i in range(len(new_list)):
        x,y = new_list[i]
        point = shapely.geometry.Point(x,y)
        points[i] = point
    line = shapely.geometry.LineString(points)
    return line

def wgs84_to_utm_df(geo_df):
    """
    Converts gdf from wgs84 to UTM
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in wgs84
    outputs:
    geo_df_utm (geopandas  dataframe): a geopandas dataframe in utm
    """
    utm_crs = geo_df.estimate_utm_crs()
    gdf_utm = geo_df.to_crs(utm_crs)
    return gdf_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts gdf from utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84


def merge_multiple_transect_time_series(transect_time_series_list,
                                        transects_path,
                                        mean_savepath,
                                        conf_savepath):
    """
    Computes uncertainty bands from list of transect_time_series_merged.csvs

    inputs:
    transect_time_series_list (list): list of transect_time_series_merged.csvs
    transects_path (str): path to transects, must have col 'transect_id', these should be integers in ascending order along the shore
    mean_savepath (str): path to save the mean shorelines
    conf_savepath (str): path to save the confidence polygons

    """

    ##load all the transect time series, compute cross distance means, mins, maxes
    cross_distance_means = [None]*len(transect_time_series_list)
    cross_distance_maxes = [None]*len(transect_time_series_list)
    dfs = [None]*len(transect_time_series_list)
    i=0
    for ts in transect_time_series_list:
        dfs[i] = pd.read_csv(ts)
        i=i+1
    dfList = [df.set_index(['dates', 'transect_id']) for df in dfs]
    big_df = pd.concat(dfList, axis=1)
    big_df = big_df.dropna()
    big_df['cross_distance_means'] = np.mean(big_df['cross_distance'],axis=1)
    big_df['cross_distance_maxes'] = np.max(big_df['cross_distance'],axis=1)
    big_df['cross_distance_mins'] = np.min(big_df['cross_distance'],axis=1)
    keep_cols = ['cross_distance_means','cross_distance_maxes','cross_distance_mins']
    for col in big_df.columns:
        if col not in keep_cols:
            try:
                big_df = big_df.drop(columns=[col])
            except:
                pass
    big_df = big_df.reset_index(level=[0,1])
    big_df['transect_id'] = big_df['transect_id'].astype(int)
    big_df['dates'] = pd.to_datetime(big_df['dates'],utc=True)

    ##load transects, get start and end coords
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf = wgs84_to_utm_df(transects_gdf)
    transects_gdf['transect_id'] = transects_gdf['transect_id'].astype(int)
    crs = transects_gdf.crs
    transects_gdf = transects_gdf.reset_index(drop=True)
    transects_gdf['geometry_saved'] = transects_gdf['geometry']
    coords = transects_gdf['geometry_saved'].get_coordinates()
    start = coords[~coords.index.duplicated(keep='first')]
    end = coords[~coords.index.duplicated(keep='last')]
    transects_gdf['x_start'] = start['x']
    transects_gdf['y_start'] = start['y']
    transects_gdf['x_end'] = end['x']
    transects_gdf['y_end'] = end['y']

    ##compute utm coords
    big_df = big_df.merge(transects_gdf,how='left',left_on='transect_id',right_on='transect_id')
    big_df['angle'] = np.arctan2(big_df['y_end'] - big_df['y_start'], big_df['x_end'] - big_df['x_start'])
    big_df['shore_x_utm_mean'] = big_df['x_start']+big_df['cross_distance_means']*np.cos(big_df['angle'])
    big_df['shore_y_utm_mean'] = big_df['y_start']+big_df['cross_distance_means']*np.sin(big_df['angle'])
    big_df['shore_x_utm_max'] = big_df['x_start']+big_df['cross_distance_maxes']*np.cos(big_df['angle'])
    big_df['shore_y_utm_max'] = big_df['y_start']+big_df['cross_distance_maxes']*np.sin(big_df['angle'])
    big_df['shore_x_utm_min'] = big_df['x_start']+big_df['cross_distance_mins']*np.cos(big_df['angle'])
    big_df['shore_y_utm_min'] = big_df['y_start']+big_df['cross_distance_mins']*np.sin(big_df['angle'])
    ##make mean shoreline file and uncertainy polygon
    dates = np.unique(big_df['dates'])
    transect_ids = np.unique(big_df['transect_id'])
    ###Should have length of projected time

    gdf_mean_dates = []
    gdf_mean_geoms = []

    gdf_confidence_intervals_dates = [] 
    gdf_confidence_intervals_geoms = []
    ###Loop over projected time
    for i in range(len(dates)):
        ###Make empty lists to hold mean coordinates, upper and lower conf coordinates
        ###These are for one time
        date = dates[i]
        proj_df = big_df[big_df['dates']==date].reset_index(drop=True)
        shoreline_eastings = [None]*len(transect_ids)
        shoreline_northings = [None]*len(transect_ids)
        shoreline_eastings_upper = [None]*len(transect_ids)
        shoreline_northings_upper = [None]*len(transect_ids)
        shoreline_eastings_lower = [None]*len(transect_ids)
        shoreline_northings_lower = [None]*len(transect_ids)
        timestamp = [date]*len(transect_ids)
        for j in range(len(transect_ids)):
            transect_id = transect_ids[j]
            proj_df_filt = proj_df[proj_df['transect_id']==transect_id].reset_index(drop=True)
            if proj_df_filt.empty:
                continue
            else:
                shoreline_eastings[j] = proj_df_filt['shore_x_utm_mean'].iloc[0]
                shoreline_northings[j] = proj_df_filt['shore_y_utm_mean'].iloc[0]
                shoreline_eastings_upper[j] = proj_df_filt['shore_x_utm_max'].iloc[0]
                shoreline_northings_upper[j] = proj_df_filt['shore_y_utm_max'].iloc[0]
                shoreline_eastings_lower[j] = proj_df_filt['shore_x_utm_min'].iloc[0]
                shoreline_northings_lower[j] = proj_df_filt['shore_y_utm_min'].iloc[0]
        shoreline_eastings_upper = split_list_at_none(shoreline_eastings_upper)
        shoreline_eastings_lower = split_list_at_none(shoreline_eastings_lower)
        shoreline_northings_upper = split_list_at_none(shoreline_northings_upper)
        shoreline_northings_lower = split_list_at_none(shoreline_northings_lower)
        shoreline_eastings = split_list_at_none(shoreline_eastings)
        shoreline_northings = split_list_at_none(shoreline_northings)
        for k in range(len(shoreline_eastings_upper)):
            if len(shoreline_eastings_upper[k])>1:
                confidence_interval_x = np.concatenate((shoreline_eastings_upper[k], list(reversed(shoreline_eastings_lower[k]))))
                confidence_interval_y = np.concatenate((shoreline_northings_upper[k], list(reversed(shoreline_northings_lower[k]))))
                confidence_interval_polygon = lists_to_Polygon(confidence_interval_x, confidence_interval_y)
                gdf_confidence_intervals_geoms.append(confidence_interval_polygon)
                gdf_confidence_intervals_dates.append(date)
                mean_shoreline_line = lists_to_LineString(shoreline_eastings[k], shoreline_northings[k])
                gdf_mean_geoms.append(mean_shoreline_line)
                gdf_mean_dates.append(date)
            elif len(shoreline_eastings_upper[k])==1:
                # create a new point by adding a small amount to the x value and y value of the point
                shoreline_eastings_upper_list = [shoreline_eastings_upper[k][0],shoreline_eastings_upper[k][0]+0.00001]
                shoreline_eastings_lower_list = [shoreline_eastings_lower[k][0],shoreline_eastings_lower[k][0]+0.00001]
                shoreline_northings_upper_list = [shoreline_northings_upper[k][0],shoreline_northings_upper[k][0]+0.00001]
                shoreline_northings_lower_list = [shoreline_northings_lower[k][0],shoreline_northings_lower[k][0]+0.00001]
                shoreline_eastings_list = [shoreline_eastings[k][0],shoreline_eastings[k][0]+0.00001]
                shoreline_northings_list = [shoreline_northings[k][0],shoreline_northings[k][0]+0.00001]

                confidence_interval_x = np.concatenate((shoreline_eastings_upper_list , list(reversed(shoreline_eastings_lower_list))))
                confidence_interval_y = np.concatenate((shoreline_northings_upper_list, list(reversed(shoreline_northings_lower_list))))
                confidence_interval_polygon = lists_to_Polygon(confidence_interval_x, confidence_interval_y)
                gdf_confidence_intervals_geoms.append(confidence_interval_polygon)
                gdf_confidence_intervals_dates.append(date)
                mean_shoreline_line = lists_to_LineString(shoreline_eastings_list, shoreline_northings_list)
                gdf_mean_geoms.append(mean_shoreline_line)
                gdf_mean_dates.append(date)


    gdf_mean_geodf = gpd.GeoDataFrame({'dates':gdf_mean_dates}, geometry = gdf_mean_geoms)
    gdf_mean_geodf = gdf_mean_geodf.set_crs(crs)
    gdf_mean_geodf = utm_to_wgs84_df(gdf_mean_geodf)
    gdf_confidence_intervals_geodf = gpd.GeoDataFrame({'dates':gdf_confidence_intervals_dates}, geometry = gdf_confidence_intervals_geoms)
    gdf_confidence_intervals_geodf = gdf_confidence_intervals_geodf.set_crs(crs)
    gdf_confidence_intervals_geodf = utm_to_wgs84_df(gdf_confidence_intervals_geodf)
    
    gdf_mean_geodf.to_file(mean_savepath)
    gdf_confidence_intervals_geodf.to_file(conf_savepath)


##big_df,gdf_mean_geodf, gdf_confidence_intervals_df = merge_multiple_transect_time_series([r'E:\TCA\analysis_ready_data\Elwha\no_corrections\raw_transect_time_series_merged.csv',
##                                                                                          r'E:\TCA\analysis_ready_data\Elwha\low_slope\tidally_corrected_transect_time_series_merged.csv',
##                                                                                          r'E:\TCA\analysis_ready_data\Elwha\average_slope\tidally_corrected_transect_time_series_merged.csv',
##                                                                                          r'E:\TCA\analysis_ready_data\Elwha\high_slope\tidally_corrected_transect_time_series_merged.csv'
##                                                                                          ],
##                                                                                         r'E:\TCA\analysis_ready_data\Elwha\transects\transects.geojson',
##                                                                                         r'E:\TCA\analysis_ready_data\Elwha\mean_shorelines.geojson',
##                                                                                         r'E:\TCA\analysis_ready_data\Elwha\conf_poly.geojson')
##

