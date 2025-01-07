"""
Breaking up and smoothing CoastSeg generated extracted shorelines

Mark Lundine
"""

import os
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import warnings
import rasterio
import rioxarray
from scipy import stats
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

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs:
    line (shapely.geometry.LineString): shapely linestring
    outputs:
    coords (List[tuples]): list of x,y coordinate pairs
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

def arr_to_LineString(coords):
    """
    Makes a line feature from an array of xy tuples
    inputs:
    coords (List[tuples]): list of x,y coordinate pairs
    outputs:
    line (shapely.geometry.LineString): shapely linestring
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def chaikins_corner_cutting(coords, refinements=3):
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

def smooth_lines(lines, refinements=5):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM (or another planar coordinate system)

    inputs:
    lines (gdf): gdf of extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    new_lines (gdf): gdf of smooth lines in UTM
    """
    lines['geometry'] = lines['geometry']
    new_lines = lines.copy()
    for i in range(len(new_lines)):
        simplify_param = new_lines.iloc[i]['simplify_param']
        line = new_lines.iloc[i]['geometry'].simplify(simplify_param)
        coords = LineString_to_arr(line)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom
    new_lines
    return new_lines

def split_line(input_lines_or_multipoints_path,
               output_path,
               linestrings_or_multi_points):
    """
    Breaks up linestring into multiple linestrings if point to point distance is too high
    inputs:
    input_lines_or_multipoints_path (str): path to the output from output_gdf (
                                           extracted_shorelines_lines.geojson or extracted_shorelines_points.geojson)
    output_path (str): path to save output to (
                       extracted_shorelines_lines.geojson or extracted_shorelines_points.geojson)
    linestrings_or_multi_points (str): 'LineString' to make LineStrings, 'MultiPoint' to make MultiPoints
    returns:
    output_path (str): path to the geodataframe with the new broken up lines
    """

    ##load shorelines, project to utm, get crs
    input_lines_or_multipoints = gpd.read_file(input_lines_or_multipoints_path)
    input_lines_or_multipoints = wgs84_to_utm_df(input_lines_or_multipoints)
    source_crs = input_lines_or_multipoints.crs

    ##these lists are gonna hold the broken up lines and their simplified tolerance
    simplify_params = []
    all_lines = []
    print('splitting lines')
    for idx,row in input_lines_or_multipoints.iterrows():
        line = input_lines_or_multipoints[input_lines_or_multipoints.index==idx].reset_index(drop=True)

        ##setting distance threshold and simplify tolerance based on satellite
        satname = line['satname'].iloc[0]
        if (satname == 'L5') or (satname == 'L7') or (satname == 'L8') or (satname == 'L9'):
            dist_threshold = 45
            simplify_param = np.sqrt(30**2 + 30**2 + 30**2)
        elif (satname=='S2'):
            dist_threshold = 15
            simplify_param = np.sqrt(10**2 + 10**2 + 10**2)
        elif (satname=='PS'):
            dist_threshold = 8
            simplify_param = np.sqrt(5**2 + 5**2 + 5**2)

        column_names = list(line.columns)
        column_names.remove('geometry')
        points_geometry = [shapely.Point(x,y) for x,y in line['geometry'].iloc[0].coords]
        attributes = [[line[column_name].values[0]]*len(points_geometry) for column_name in column_names]
        input_coords_dict = dict(zip(column_names, attributes))
        input_coords_dict['geometry'] = points_geometry
        input_coords = gpd.GeoDataFrame(input_coords_dict, crs=source_crs)
        
        ##make the shifted geometries to compute point to point distance
        input_coords_columns = input_coords.columns[:]
        new_geometry_column = 'geom_2'
        input_coords[new_geometry_column] = input_coords['geometry'].shift(-1)

        ##compute distance
        def my_dist(in_row):
            return in_row['geometry'].distance(in_row['geom_2'])
        input_coords['dist'] = input_coords.loc[:input_coords.shape[0]-2].apply(my_dist, axis=1)
        ##break up line into multiple lines
        input_coords['break'] = (input_coords['dist'] > dist_threshold).shift(1)
        input_coords.loc[0,'break'] = True
        input_coords['line_id'] = input_coords['break'].astype(int).cumsum()

        ##make the lines
        def my_line_maker(in_grp):
            if len(in_grp) == 1:
                return list(in_grp)[0]
            elif linestrings_or_multi_points == 'LineString':
                return shapely.geometry.LineString(list(in_grp))
            elif linestrings_or_multi_points == 'MultiPoint':
                return shapely.geometry.MultiPoint(list(in_grp))
        new_lines_gdf = input_coords.groupby(['line_id']).agg({'geometry':my_line_maker}).reset_index()
        
        ##drop points and only keep linestrings
        new_lines_gdf['geom_type'] = [type(a) for a in new_lines_gdf['geometry']]
        new_lines_gdf = new_lines_gdf[new_lines_gdf['geom_type']!=shapely.Point].reset_index(drop=True)
        for column in column_names:
            new_lines_gdf[column] = [line[column].values[0]]*len(new_lines_gdf)
        new_lines_gdf = new_lines_gdf.drop(columns=['geom_type', 'line_id'])
        all_lines.append(new_lines_gdf)
        simplify_params.append(simplify_param)

    ##concatenate everything into one gdf, set geometry and crs
    all_lines_gdf = pd.concat(all_lines)
    all_lines_gdf['simplify_param'] = simplify_param
    all_lines_gdf = all_lines_gdf.set_geometry('geometry')
    all_lines_gdf = all_lines_gdf.set_crs(source_crs)
    print('lines split')

    ##smooth the lines
    print('smoothing lines')
    smooth_lines_gdf = smooth_lines(all_lines_gdf)
    print('lines smooth')

    ##put back in wgs84, save new file
    smooth_lines_gdf = utm_to_wgs84_df(all_lines_gdf)
    smooth_lines_gdf.to_file(output_path)
    
    return output_path
    
