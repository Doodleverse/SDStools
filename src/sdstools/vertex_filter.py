import os
import geopandas as gpd
import numpy as np

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

def vertex_filter(shorelines,n_sigma):
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
    gdf = wgs84_to_utm_df(gdf)
    
    count = len(gdf)
    new_count = None
    for index, row in gdf.iterrows():
        vtx = len(row['geometry'].coords)
        length = row['geometry'].length
        gdf.at[index,'vtx'] = vtx
        gdf.at[index,'length'] = length
        gdf.ad[index,'length:vtx'] = length/vtx
    filter_gdf = gdf.copy()

    while count != new_count:
        count = len(filter_gdf)
        sigma = np.std(filter_gdf['length:vtx'])
        mean = np.mean(filter_gdf['length:vtx'])
        high_limit = mean+n_sigma*sigma
        low_limit = mean-n_sigma*sigma
        filter_gdf = gdf[gdf['length:vtx']< high_limit]
        filter_gdf = filter_gdf[filter_gdf['length:vtx']> low_limit]
        if mean < 5:
            break
        new_count = len(filter_gdf)
        
    filter_gdf = filter_gdf.reset_index(drop=True)
    return filter_gdf
