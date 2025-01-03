"""
Mark Lundine
Taking outputs from lstm_parallel_coastseg.py and re-casting them into
geographic coordinates to make GIS outputs
(mean shorelines and confidence interval polygons).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import shapely
import geopandas as gpd
import glob
from math import degrees, atan2, radians

def gb(x1, y1, x2, y2):
    """
    Gets compass bearing between two points
    """
    bearing = degrees(atan2(y2 - y1, x2 - x1))
    return bearing

def lists_to_Polygon(list1, list2):
    """
    Makes a shapely polygon from a list of x and y coordinates
    """
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
    """
    Makes a shapely line from a list of x and y coordinates
    """
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


def single_transect(model_df,
                    transect,
                    crs_wgs84,
                    crs_utm,
                    switch_dir=False):
    """
    Gets utm and wgs84 coordinates from model outputs
    Returns a df with added columns for the mean positions, upper conf interval, and lower conf interval
    (eastings, northings) in UTM and WGS84
    """
    try:
        means = model_df['forecast_mean_position']
        uppers = model_df['forecast_upper_conf']
        lowers = model_df['forecast_lower_conf']
    except:
        means = model_df['predicted_mean_position']
        uppers = model_df['predicted_upper_conf']
        lowers = model_df['predicted_lower_conf']

    ##Get the angle of the transect
    coords = shapely.get_coordinates(transect.geometry.boundary)
    first = coords[0]
    last = coords[1]
    firstX = first[0]
    firstY = first[1]
    lastX = last[0]
    lastY = last[1]
    if switch_dir == False:
        angle = np.arctan2(lastY - firstY, lastX - firstX)
    else:
        angle = np.arctan2(firstY - lastY, firstX - lastX)
    
    ##Compute model X,Y positions,utm
    northings_mean_utm = firstY + means*np.sin(angle)
    eastings_mean_utm = firstX + means*np.cos(angle)
    northings_upper_utm = firstY + uppers*np.sin(angle)
    eastings_upper_utm = firstX + uppers*np.cos(angle)
    northings_lower_utm = firstY + lowers*np.sin(angle)
    eastings_lower_utm = firstX + lowers*np.cos(angle)

    ##Compute model lat,lon positions, wgs84
    ###TODO, need points, need dummy gdf, need convert to wgs84, need to pull points from geom
    points_utm_mean = [shapely.Point(xy) for xy in zip(eastings_mean_utm, northings_mean_utm)]
    points_utm_upper = [shapely.Point(xy) for xy in zip(eastings_upper_utm, northings_upper_utm)]
    points_utm_lower = [shapely.Point(xy) for xy in zip(eastings_lower_utm, northings_lower_utm)]

    ##Conversion to wgs84 with dummy gdfs
    #mean
    dummy_gdf_utm_mean = gpd.GeoDataFrame({'geometry':points_utm_mean},
                                          crs=crs_utm)
    dummy_gdf_wgs84_mean = dummy_gdf_utm_mean.to_crs(crs_wgs84)
    #upper
    dummy_gdf_utm_upper = gpd.GeoDataFrame({'geometry':points_utm_upper},
                                           crs=crs_utm)
    dummy_gdf_wgs84_upper = dummy_gdf_utm_upper.to_crs(crs_wgs84)
    #lower
    dummy_gdf_utm_lower = gpd.GeoDataFrame({'geometry':points_utm_lower},
                                           crs=crs_utm)
    dummy_gdf_wgs84_lower = dummy_gdf_utm_lower.to_crs(crs_wgs84)

    ##Get wgs84 coordinates as arrays
    #mean
    points_wgs84_mean = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84_mean.geometry]
    points_wgs84_mean = np.array(points_wgs84_mean)
    points_wgs84_mean = points_wgs84_mean.reshape(len(points_wgs84_mean),2)
    eastings_mean_wgs84 = points_wgs84_mean[:,0]
    northings_mean_wgs84 = points_wgs84_mean[:,1]
    #upper
    points_wgs84_upper = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84_upper.geometry]
    points_wgs84_upper = np.array(points_wgs84_upper)
    points_wgs84_upper = points_wgs84_upper.reshape(len(points_wgs84_upper),2)
    eastings_upper_wgs84 = points_wgs84_upper[:,0]
    northings_upper_wgs84 = points_wgs84_upper[:,1]
    #lower
    points_wgs84_lower = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84_lower.geometry]
    points_wgs84_lower = np.array(points_wgs84_lower)
    points_wgs84_lower = points_wgs84_lower.reshape(len(points_wgs84_lower),2)
    eastings_lower_wgs84 = points_wgs84_lower[:,0]
    northings_lower_wgs84 = points_wgs84_lower[:,1]
    
    ##Put utm positions into dataframe
    model_df['northings_mean_utm'] = northings_mean_utm
    model_df['eastings_mean_utm'] = eastings_mean_utm
    model_df['northings_upper_utm'] = northings_upper_utm
    model_df['eastings_upper_utm'] = eastings_upper_utm
    model_df['northings_lower_utm'] = northings_lower_utm
    model_df['eastings_lower_utm'] = eastings_lower_utm
    ##Put wgs84 positions into dataframe
    model_df['northings_mean_wgs84'] = northings_mean_wgs84
    model_df['eastings_mean_wgs84'] = eastings_mean_wgs84
    model_df['northings_upper_wgs84'] = northings_upper_wgs84
    model_df['eastings_upper_wgs84'] = eastings_upper_wgs84
    model_df['northings_lower_wgs84'] = northings_lower_wgs84
    model_df['eastings_lower_wgs84'] = eastings_lower_wgs84

    ##Clean Up
    means = None
    uppers = None
    lowers = None
    northings_mean_utm = None
    eastings_mean_utm = None
    northings_upper_utm = None
    eastings_upper_utm = None
    northings_lower_utm = None
    eastings_lower_utm = None
    northings_mean_wgs84 = None
    eastings_mean_wgs84 = None
    northings_upper_wgs84 = None
    eastings_upper_wgs84 = None
    northings_lower_wgs84 = None
    eastings_lower_wgs84 = None
    points_wgs84_mean = None
    points_wgs84_upper = None
    points_wgs84_lower = None
    dummy_gdf_utm_mean = None
    dummy_gdf_wgs84_mean = None
    dummy_gdf_utm_upper = None
    dummy_gdf_wgs84_upper = None
    dummy_gdf_utm_lower = None
    dummy_gdf_wgs84_lower = None
    
    return model_df

def multiple_transects(forecast_stacked_df,
                       predict_stacked_df,
                       transect_ids,
                       config_gdf_path,
                       forecast_times,
                       prediction_times,
                       savefolder,
                       sitename,
                       switch_dir=False):
    """
    Build two shapefiies
    One that contains mean LSTM shoreline projections (so lines with a timestamp)
    One that contains confidence interval polygons (polygons with a timestamp)
    """
    ##make paths to save outputs to
    mean_forecast_savepath = os.path.join(savefolder, sitename+'_'+str(transect_ids[0])+'to'+str(transect_ids[-1])+'forecast_mean_shorelines.geojson')
    conf_forecast_savepath = os.path.join(savefolder, sitename+'_'+str(transect_ids[0])+'to'+str(transect_ids[-1])+'forecast_confidence_intervals.geojson')
    mean_pred_savepath = os.path.join(savefolder, sitename+'_'+str(transect_ids[0])+'to'+str(transect_ids[-1])+'predicted_mean_shorelines.geojson')
    conf_pred_savepath = os.path.join(savefolder, sitename+'_'+str(transect_ids[0])+'to'+str(transect_ids[-1])+'predicted_confidence_intervals.geojson')

    ##load in config_gdf and transects, get crs for wgs84 and utm
    config_gdf = gpd.read_file(config_gdf_path)
    crs_wgs84 = config_gdf.crs
    transects = config_gdf[config_gdf['type']=='transect']
    crs_utm = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(crs_utm)

    

    ##############Forecast section start
    ##getting just the years into an array
    time = forecast_times
    years = [None]*len(time)
    for i in range(len(time)):
        ts = time[i]
        year = ts[0:4]
        years[i] = int(year)

    ##loop over transects to get coordinates of modelled cross-shore positions
    ##overwrite original dfs and stacked df to have coordinates
    forecast_dfs = [None]*len(transect_ids)
    for i in range(len(transect_ids)):
        transect_id = transect_ids[i]
        site = sitename+'_'+str(transect_ids[i])
        transect = transects_utm[transects_utm['id']==transect_id].reset_index()
        forecast_df = forecast_stacked_df[forecast_stacked_df['transect_id']==transect_id].reset_index()
        forecast_df = single_transect(forecast_df,
                                      transect,
                                      crs_wgs84,
                                      crs_utm,
                                      switch_dir=switch_dir)
        forecast_df.drop(columns=['Unnamed: 0'],inplace=True)
        forecast_df.to_csv(os.path.join(savefolder, site+'_forecast.csv'),index=False)
        forecast_dfs[i] = forecast_df
    forecast_stacked_df = pd.concat(forecast_dfs, keys=transect_ids).reset_index()
    forecast_stacked_df.drop(columns=['level_1'],inplace=True)
    forecast_stacked_df.drop(columns={'level_0'},inplace=True)
    forecast_stacked_df.drop(columns=['index'])
    forecast_stacked_df.to_csv(os.path.join(savefolder, 'forecast_stacked.csv'), index=False)
    
    ###Should have length of forecasted time
    gdf_mean_dict = {'date':time,
                     'year':years}
    gdf_mean_df = pd.DataFrame(gdf_mean_dict)
    gdf_mean_geoms = [None]*len(gdf_mean_df)

    gdf_confidence_intervals_dict = {'date':time,
                                     'year':years}
    gdf_confidence_intervals_df = pd.DataFrame(gdf_confidence_intervals_dict)
    gdf_confidence_intervals_geoms = [None]*len(gdf_confidence_intervals_df)
    
    ###Loop over forecasted time
    for i in range(len(time)):
        ###Make empty lists to hold mean coordinates, upper and lower conf coordinates
        ###These are for one time
        shoreline_eastings = [None]*len(transect_ids)
        shoreline_northings = [None]*len(transect_ids)
        shoreline_eastings_upper = [None]*len(transect_ids)
        shoreline_northings_upper = [None]*len(transect_ids)
        shoreline_eastings_lower = [None]*len(transect_ids)
        shoreline_northings_lower = [None]*len(transect_ids)
        timestamp = [time[i]]*len(transect_ids)
        ##loop over transects
        for j in range(len(transect_ids)):
            transect_id = transect_ids[j]
            forecast_df = forecast_stacked_df[forecast_stacked_df['transect_id']==transect_id].reset_index()
            shoreline_eastings[j] = forecast_df['eastings_mean_wgs84'][i]
            shoreline_northings[j] = forecast_df['northings_mean_wgs84'][i]
            shoreline_eastings_upper[j] = forecast_df['eastings_upper_wgs84'][i]
            shoreline_northings_upper[j] = forecast_df['northings_upper_wgs84'][i]
            shoreline_eastings_lower[j] = forecast_df['eastings_lower_wgs84'][i]
            shoreline_northings_lower[j] = forecast_df['northings_lower_wgs84'][i]
        confidence_interval_x = np.concatenate((shoreline_eastings_upper, list(reversed(shoreline_eastings_lower))))
        confidence_interval_y = np.concatenate((shoreline_northings_upper, list(reversed(shoreline_northings_lower))))
        
        confidence_interval_polygon = lists_to_Polygon(confidence_interval_x, confidence_interval_y)
        gdf_confidence_intervals_geoms[i] = confidence_interval_polygon
        
        mean_shoreline_line = lists_to_LineString(shoreline_eastings, shoreline_northings)
        gdf_mean_geoms[i] = mean_shoreline_line

    gdf_mean_geodf = gpd.GeoDataFrame(gdf_mean_df, geometry = gdf_mean_geoms)
    gdf_mean_geodf = gdf_mean_geodf.set_crs(crs_wgs84)
    gdf_confidence_intervals_geodf = gpd.GeoDataFrame(gdf_confidence_intervals_df, geometry = gdf_confidence_intervals_geoms)
    gdf_confidence_intervals_geodf = gdf_confidence_intervals_geodf.set_crs(crs_wgs84)
    
    gdf_mean_geodf.to_file(mean_forecast_savepath)
    gdf_confidence_intervals_geodf.to_file(conf_forecast_savepath)
    ##############Forecast section end




    ##############Prediction section start
    ##get years as array
    time = prediction_times
    years = [None]*len(time)
    for i in range(len(time)):
        ts = time[i]
        year = ts[0:4]
        years[i] = int(year)
    
    ##loop over transects to get coordinates of modelled cross-shore positions
    ##overwrite original dfs and stacked df to have coordinates
    predict_dfs = [None]*len(transect_ids)
    for i in range(len(transect_ids)):
        transect_id = transect_ids[i]
        site = sitename+'_'+str(transect_ids[i])
        transect = transects_utm[transects_utm['id']==transect_id].reset_index()
        predict_df = predict_stacked_df[predict_stacked_df['transect_id']==transect_id].reset_index()
        predict_df = single_transect(predict_df,
                                     transect,
                                     crs_wgs84,
                                     crs_utm,
                                     switch_dir=switch_dir)
        predict_df.drop(columns=['Unnamed: 0'],inplace=True)
        predict_df.to_csv(os.path.join(savefolder, site+'_predict.csv'),index=False)
        predict_dfs[i] = predict_df
    predict_stacked_df = pd.concat(predict_dfs, keys=transect_ids).reset_index()
    predict_stacked_df.drop(columns=['level_1'],inplace=True)
    predict_stacked_df.drop(columns={'level_0'},inplace=True)
    predict_stacked_df.drop(columns=['index'])
    predict_stacked_df.to_csv(os.path.join(savefolder, 'predict_stacked.csv'), index=False)

    ###Should have length of predicted time
    gdf_mean_dict = {'date':time,
                     'year':years}
    gdf_mean_df = pd.DataFrame(gdf_mean_dict)
    gdf_mean_geoms = [None]*len(gdf_mean_df)

    gdf_confidence_intervals_dict = {'date':time,
                                     'year':years}
    gdf_confidence_intervals_df = pd.DataFrame(gdf_confidence_intervals_dict)
    gdf_confidence_intervals_geoms = [None]*len(gdf_confidence_intervals_df)
                                     
    ###Loop over predicted time
    for i in range(len(time)):
        ###Make empty lists to hold mean coordinates, upper and lower conf coordinates
        ###These are for one time
        shoreline_eastings = [None]*len(transect_ids)
        shoreline_northings = [None]*len(transect_ids)
        shoreline_eastings_upper = [None]*len(transect_ids)
        shoreline_northings_upper = [None]*len(transect_ids)
        shoreline_eastings_lower = [None]*len(transect_ids)
        shoreline_northings_lower = [None]*len(transect_ids)
        timestamp = [time[i]]*len(transect_ids)
        ##loop over transects
        for j in range(len(transect_ids)):
            transect_id = transect_ids[j]
            pred_df = predict_stacked_df[predict_stacked_df['transect_id']==transect_id].reset_index()
            shoreline_eastings[j] = pred_df['eastings_mean_wgs84'][i]
            shoreline_northings[j] = pred_df['northings_mean_wgs84'][i]
            shoreline_eastings_upper[j] = pred_df['eastings_upper_wgs84'][i]
            shoreline_northings_upper[j] = pred_df['northings_upper_wgs84'][i]
            shoreline_eastings_lower[j] = pred_df['eastings_lower_wgs84'][i]
            shoreline_northings_lower[j] = pred_df['northings_lower_wgs84'][i]
        confidence_interval_x = np.concatenate((shoreline_eastings_upper, list(reversed(shoreline_eastings_lower))))
        confidence_interval_y = np.concatenate((shoreline_northings_upper, list(reversed(shoreline_northings_lower))))
        
        confidence_interval_polygon = lists_to_Polygon(confidence_interval_x, confidence_interval_y)
        gdf_confidence_intervals_geoms[i] = confidence_interval_polygon
        
        mean_shoreline_line = lists_to_LineString(shoreline_eastings, shoreline_northings)
        gdf_mean_geoms[i] = mean_shoreline_line
        
    gdf_mean_geodf = gpd.GeoDataFrame(gdf_mean_df, geometry = gdf_mean_geoms)
    gdf_mean_geodf = gdf_mean_geodf.set_crs(crs_wgs84)
    gdf_confidence_intervals_geodf = gpd.GeoDataFrame(gdf_confidence_intervals_df, geometry = gdf_confidence_intervals_geoms)
    gdf_confidence_intervals_geodf = gdf_confidence_intervals_geodf.set_crs(crs_wgs84)
    
    gdf_mean_geodf.to_file(mean_pred_savepath)
    gdf_confidence_intervals_geodf.to_file(conf_pred_savepath)
    ##############Prediction section end
    
def main(sitename,
         coastseg_matrix_path,
         forecast_stacked_df_path,
         predict_stacked_df_path,
         save_folder,
         config_gdf_path,
         switch_dir=False):
    """
    Takes projected cross-shore positions and uncertainties and constructs 2D projected shorelines/uncertainties
    Saves these to two shapefiles (mean shorelines and confidence intervals)
    inputs:
    sitename: Name of site (str)
    coastseg_matrix_path: path to the resampled coastseg matrix (str)
    model_folder: path ot the folder containing lstm results (str)
    save_folder: folder to save projected shoreline shapefiles to (str)
    config_gdf_path: path to config_gdf.geojson containing transects (str)
    switch_dir: Optional, if True, then transect direction is reversed
    """
    coastseg_matrix = pd.read_csv(coastseg_matrix_path)
    transect_ids = list(coastseg_matrix.columns[1:])
    coastseg_matrix = None
    forecast_stacked_df = pd.read_csv(forecast_stacked_df_path)
    predict_stacked_df = pd.read_csv(predict_stacked_df_path)

    
    
    forecast_times = forecast_stacked_df[forecast_stacked_df['transect_id']==transect_ids[0]]['date'].values
    prediction_times = predict_stacked_df[predict_stacked_df['transect_id']==transect_ids[0]]['date'].values

    multiple_transects(forecast_stacked_df,
                       predict_stacked_df,
                       transect_ids,
                       config_gdf_path,
                       forecast_times,
                       prediction_times,
                       save_folder,
                       sitename,
                       switch_dir=switch_dir)


