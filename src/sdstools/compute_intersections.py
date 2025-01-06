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
    Converts wgs84 to UTM
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
    Converts utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs:
    
    outputs:
    line (shapely.LineString): LineString of the coords
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs:
    line (shapely.LineString): LineString of the coords
    outputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)]
    """
    list_array = []
    for pp in line.coords:
        list_array.append(pp)
    coords = np.array(list_array)
    return coords

def chaikins_corner_cutting(coords, refinements=5):
    """
    Smooths out lines or polygons with Chaikin's method
    inputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)]
    outputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)],
                              this is the smooth line
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

def smooth_lines(lines,refinements=5):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM (or another planar coordinate system)

    inputs:
    shorelines (gdf): gdf of extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    new_lines (gdf): gdf of smooth lines in UTM
    """
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line.geometry)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines
    return new_lines

def cross_distance(start_x, start_y, end_x, end_y):
    """distance formula, sqrt((x_1-x_0)^2 + (y_1-y_0)^2)"""
    dist = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    return dist

def transect_timeseries(shorelines_path,
                        transects_path,
                        output_csv_path):
    """
    Generates timeseries of shoreline cross-shore position
    given a geojson/shapefile containing shorelines and a
    geojson/shapefile containing cross-shore transects.
    Computes interesection points between shorelines
    and transects. Saves the merged transect timeseries.
    
    inputs:
    shoreline_path (str): path to file containing shorelines
    transect_path (str): path to file containing cross-shore transects
    output_path (str): path to the csv file to output results to
    """
    # load transects, project to utm, get start x and y coords
    print('Loading transects, computing start coordinates')
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf = wgs84_to_utm_df(transects_gdf)
    transects_gdf = transects_gdf.reset_index()
    transects_gdf['geometry_saved'] = transects_gdf['geometry']
    coords = transects_gdf['geometry_saved'].get_coordinates()
    coords = coords[~coords.index.duplicated(keep='first')]
    transects_gdf['x_start'] = coords['x']
    transects_gdf['y_start'] = coords['y']
    
    # load shorelines, project to utm, smooth
    print('smoothing shorelines')
    shorelines_gdf = gpd.read_file(shorelines_path)
    shorelines_gdf = wgs84_to_utm_df(shorelines_gdf)
    shorelines_gdf = smooth_lines(shorelines_gdf)
    print('shorelines now smooth')

    print('computing intersections')
    # spatial join shorelines to transects
    joined_gdf = gpd.sjoin(shorelines_gdf, transects_gdf, predicate='intersects')
    
    # get points, keep second point if multipoint (most seaward intersection)
    joined_gdf['intersection_point'] = joined_gdf.geometry.intersection(joined_gdf['geometry_saved'])
    for i in range(len(joined_gdf['intersection_point'])):
        point = joined_gdf['intersection_point'].iloc[i]
        if type(point) == shapely.MultiPoint:
            points = [shapely.Point(coord) for coord in point.geoms]
            last_point = points[-1]
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
    joined_df = joined_gdf.drop(columns=['geometry', 'type', 'geometry_saved',
                                         'x_start', 'y_start', 'intersection_point',
                                          'index_right0','index'
                                         ]
                                )
    ##save file
    joined_df.to_csv(output_csv_path)
    print('intersections computed')


