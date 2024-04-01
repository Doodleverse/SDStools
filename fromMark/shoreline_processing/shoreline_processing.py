import numpy as np
import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy import stats
import shapely
from shapely import geometry
import warnings
warnings.filterwarnings("ignore")


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

def utm_to_wgs84_file(geojson_file):
    """
    Converts utm to wgs84
    inputs:
    geojson_file (path): path to a geojson in utm
    outputs:
    geojson_file_wgs84 (path): path to a geojson in wgs84
    """
    geojson_file_wgs84 = os.path.splitext(geojson_file)[0]+'_wgs84.geojson'

    gdf_utm = gpd.read_file(geojson_file)
    wgs84_crs = 'epsg:4326'

    gdf_wgs84 = gdf_wgs84.to_crs(wgs84_crs)
    gdf_wgs84.to_file(geojson_file_wgs84)
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

def vertex_filter(shorelines):
    """
    Recursive 3-sigma filter on vertices in shorelines
    Will filter out shorelines that have too many or too few
    vertices until all of the shorelines left in the file are within
    Mean+/-3*std
    
    Saves output to the same directory with same name but with (_vtx) appended.

    inputs:
    shorelines (str): path to the extracted shorelines geojson
    outputs:
    new_path (str): path to the filtered file 
    """
    gdf = gpd.read_file(shorelines)
    
    count = len(gdf)
    new_count = None
    for index, row in gdf.iterrows():
        gdf.at[index,'vtx'] = len(row['geometry'].coords)
    filter_gdf = gdf.copy()

    while count != new_count:
        count = len(filter_gdf)
        sigma = np.std(filter_gdf['vtx'])
        mean = np.mean(filter_gdf['vtx'])
        limit = mean+3*sigma
        filter_gdf = gdf[gdf['vtx']< limit]
        if mean < 5:
            break
        new_count = len(filter_gdf)
    
    new_path = os.path.splitext(shorelines)[0]+'_vtx.geojson'
    filter_gdf.to_file(new_path)
    return new_path

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

def smooth_lines(shorelines):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    shorelines (str): path to extracted shorelines in UTM
    outputs:
    save_path (str): path of output file in UTM
    """
    dirname = os.path.dirname(shorelines)
    dirname = os.path.dirname(dirname)
    save_path = os.path.join(dirname,os.path.splitext(os.path.basename(shorelines))[0]+'_smooth.geojson')
    lines = gpd.read_file(shorelines)
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line.geometry)
        refined = chaikins_corner_cutting(coords)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines.to_file(save_path)
    return save_path

def ref_shoreline_filter(reference_shoreline, model_shorelines, distance_threshold=250):
    """
    filters extracted shoreline points that are not contained within a buffer radius of a reference shoreline
    saves output with '_ref_shoreline_filter' appended to original filename in same directory
    inputs:
    reference_shoreline (str): path to reference shoreline geojson
    model_shorelines (str): path to extracted shorelines
    distance_threshold (float): buffer radius
    outputs:
    save_path (str): path to the output file
    """
    save_path = os.path.splitext(model_shorelines)[0]+'_ref_shoreline_filter.geojson'
    reference_gdf = gpd.read_file(reference_shoreline)
    model_gdf = gpd.read_file(model_shorelines)
    
    ##First need to get rid of lines that are completely outside of the ref shoreline buffer
    buffer = reference_gdf.buffer(distance_threshold,resolution=1)
    buffer_vals = [None]*len(model_gdf)
    for i in range(len(model_gdf)):
        line_entry = model_gdf.iloc[i]
        line = line_entry.geometry
        bool_val = buffer.intersects(line).values[0]
        buffer_vals[i] = bool_val
    model_gdf['buffer_vals'] = buffer_vals
    model_gdf_filter = model_gdf[model_gdf['buffer_vals']]
    
    ##Now get rid of points that lie outside buffer but preserve the rest of the shoreline
    new_lines = [None]*len(model_gdf)
    for i in range(len(model_gdf_filter)):
        line_entry = model_gdf_filter.iloc[i]
        line = line_entry.geometry
        line_arr = LineString_to_arr(line)
        bool_vals = [None]*len(line_arr)
        j = 0
        for point in line_arr:
            point = geometry.Point(point)
            bool_val = buffer.contains(point)[0]
            bool_vals[j] = bool_val
            j=j+1
        new_line_arr = line_arr[bool_vals]
        new_line_LineString = arr_to_LineString(new_line_arr)
        new_lines[i] = new_line_LineString

    ##Assign the new geometries, save the output
    model_gdf_filter['geometry'] = new_lines
    model_gdf_filter.to_file(save_path)
    return save_path


def ref_poly_filter(reference_polygon, model_shorelines):
    """
    filters extracted shoreline points that are not contained within a reference region/polygon
    saves output with '_ref_region_filter' appended to original filename in same directory
    inputs:
    reference_region_path (str): path to reference region geojson
    model_shorelines (str): path to extracted shorelines
    outputs:
    save_path (str): path to the output file
    """

    ##Loading stuff in
    save_path = os.path.splitext(model_shorelines)[0]+'_ref_shoreline_filter.geojson'
    model_gdf = gpd.read_file(model_shorelines)
    ref_poly_gdf = gpd.read_file(reference_polygon)

    ##First need to get rid of lines that are completely outside of the ref polygon
    ref_polygon = ref_poly_gdf.geometry
    buffer_vals = [None]*len(model_gdf)
    for i in range(len(model_gdf)):
        line_entry = model_gdf.iloc[i]
        line = line_entry.geometry
        bool_val = ref_polygon.intersects(line).values[0]
        buffer_vals[i] = bool_val
    model_gdf['buffer_vals'] = buffer_vals
    model_gdf_filter = model_gdf[model_gdf['buffer_vals']]

    ##Now get rid of points that lie outside ref polygon but preserve the rest of the shoreline
    new_lines = [None]*len(model_gdf_filter)
    for i in range(len(model_gdf_filter)):
        line_entry = model_gdf_filter.iloc[i]
        line = line_entry.geometry
        line_arr = LineString_to_arr(line)
        bool_vals = [None]*len(line_arr)
        j = 0
        for point in line_arr:
            point = geometry.Point(point)
            bool_val = ref_polygon.contains(point)[0]
            bool_vals[j] = bool_val
            j=j+1
        new_line_arr = line_arr[bool_vals]
        new_line_LineString = arr_to_LineString(new_line_arr)
        new_lines[i] = new_line_LineString

    ##Assign the new geometries, save the output
    model_gdf_filter['geometry'] = new_lines
    model_gdf_filter.to_file(save_path)

    return save_path
    
        
def add_year_as_field(shorelines):
    """
    Just adding the year as a field for visualization purposes
    inputs:
    shorelines (str): path to extracted shorelines
    outputs:
    shorelines (str): path to output
    """
    model_gdf = gpd.read_file(shorelines)
    years = [None]*len(model_gdf)
    for i in range(len(model_gdf)):
        line_entry = model_gdf.iloc[i]
        year = line_entry['date'].year
        years[i] = year
    model_gdf['year'] = years
    model_gdf.to_file(shorelines)
    return shorelines



    
    
