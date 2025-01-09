"""
Mostly vectorized version of computing shoreline intersections with transects

Mark Lundine
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import shapely
import datetime
import math 
import warnings
warnings.filterwarnings("ignore")

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

def cross_distance(start_x, start_y, end_x, end_y):
    """distance formula, sqrt((x_1-x_0)^2 + (y_1-y_0)^2)"""
    dist = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    return dist

def transect_timeseries(shorelines_path,
                        transects_path,
                        output_merged_path,
                        output_mat_path):
    """
    Generates timeseries of shoreline cross-shore position
    given a geojson/shapefile containing shorelines and a
    geojson/shapefile containing cross-shore transects.
    Computes interesection points between shorelines
    and transects. Saves the merged transect timeseries.
    
    inputs:
    shoreline_path (str): path to file containing shorelines
    transect_path (str): path to file containing cross-shore transects
    output_merged path (str): path to save the merged csv file 
    output_mat_path (str): path to save the matrix csv file
    """
    # load transects, project to utm, get start x and y coords
    print('Loading transects, computing start coordinates')
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf = wgs84_to_utm_df(transects_gdf)
    crs = transects_gdf
    transects_gdf = transects_gdf.reset_index(drop=True)
    transects_gdf['geometry_saved'] = transects_gdf['geometry']
    coords = transects_gdf['geometry_saved'].get_coordinates()
    coords = coords[~coords.index.duplicated(keep='first')]
    transects_gdf['x_start'] = coords['x']
    transects_gdf['y_start'] = coords['y']
    
    # load shorelines, project to utm, smooth
    shorelines_gdf = gpd.read_file(shorelines_path)
    shorelines_gdf = wgs84_to_utm_df(shorelines_gdf)

    print('computing intersections')
    # spatial join shorelines to transects
    joined_gdf = gpd.sjoin(shorelines_gdf, transects_gdf, predicate='intersects')
    
    # get points, keep highest cross distance point if multipoint (most seaward intersection)
    joined_gdf['intersection_point'] = joined_gdf.geometry.intersection(joined_gdf['geometry_saved'])
    for i in range(len(joined_gdf['intersection_point'])):
        point = joined_gdf['intersection_point'].iloc[i]
        start_x = joined_gdf['x_start'].iloc[i]
        start_y = joined_gdf['y_start'].iloc[i]
        if type(point) == shapely.MultiPoint:
            points = [shapely.Point(coord) for coord in point.geoms]
            points = gpd.GeoSeries(points, crs=crs)
            coords = points.get_coordinates()
            dists = [None]*len(coords)
            for j in range(len(coords)):
                dists[j] = cross_distance(start_x, start_y, coords['x'].iloc[j], coords['y'].iloc[j])
            max_dist_idx = np.argmax(dists)
            last_point = points[max_dist_idx]
            joined_gdf['intersection_point'].iloc[i] = last_point
    # get x's and y's for intersections
    intersection_coords = joined_gdf['intersection_point'].get_coordinates()
    joined_gdf['intersect_x'] = intersection_coords['x']
    joined_gdf['intersect_y'] = intersection_coords['y']
    
    # get cross distance
    joined_gdf['cross_distance'] = cross_distance(joined_gdf['x_start'], 
                                                  joined_gdf['y_start'], 
                                                  joined_gdf['intersect_x'], 
                                                  joined_gdf['intersect_y'])
    ##clean up columns
    joined_gdf = joined_gdf.rename(columns={'date':'dates'})
    keep_columns = ['dates','satname','geoaccuracy','cloud_cover','transect_id',
                    'intersect_x','intersect_y','cross_distance']
    joined_gdf = joined_gdf.rename(columns={'date':'dates'}).reset_index(drop=True)

    for col in joined_gdf.columns:
        if col not in keep_columns:
            joined_gdf = joined_gdf.drop(columns=[col])

    joined_df = joined_gdf.reset_index(drop=True)
    
    ##pivot to make the matrix
    joined_mat = joined_df.pivot(index='dates', columns='transect_id', values='cross_distance')
    joined_mat.columns.name = None
    joined_mat.to_csv(output_mat_path)
    
    ##save file
    joined_df.to_csv(output_merged_path)
    print('intersections computed')


