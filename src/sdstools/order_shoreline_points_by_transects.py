"""
Order extracted shoreline points by transects
Mark Lundine, USGS
"""

import geopandas as gpd
import os
import numpy as np
import pandas as pd
import shapely

def order_shoreline_points_with_transects(shoreline_points_path,
                                          config_gdf_path,
                                          output_vectors_path):
    """
    Takes extracted shoreline points along transects
    and orders them in the order the transects are given.
    Make sure the transects are in the correct order and orientation (land to sea)
    inputs:
    shoreline_points (str): path to the extracted shoreline points,
                            either raw_transect_time_series_points.geojson
                            or tidally_corrected_transect_time_series.geojson
    config_gdf (str): path to the config_gdf, which contains the transects
    output_vectors_path (str): path to save the shorelines to
    outputs:
    output_vectors_path (str): path to save the shorelines to
    """
    config_gdf = gpd.read_file(config_gdf_path)
    shoreline_points = gpd.read_file(shoreline_points_path)

    dates = np.unique(shoreline_points['date'])
    transects = config_gdf[config_gdf['type']=='transect']
    transects = transects.sort_values(by='id').reset_index(drop=True)
    i=0
    new_lines = [None]*len(dates)
    for date in dates:
        shoreline_points_filter = shoreline_points[shoreline_points['date']==date]
        points = [None]*len(transects)
        for k in range(len(transects)):
            transect = transects.iloc[k].geometry
            intersect = transect.intersection(shoreline_points_filter.geometry)
            points[k] = intersect
            print(intersect)
        new_lines[i] = shapely.LineString(points)
        i=i+1
    new_gdf_dict = {'date':dates,
                    'geometry':new_lines}
    new_gdf = gpd.GeoDataFrame(pd.DataFrame(new_gdf_dict), crs=shoreline_points.crs)
    new_gdf.to_file(output_vectors_path)

    return output_vectors_path


