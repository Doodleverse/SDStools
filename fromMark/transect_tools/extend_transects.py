# load modules
import os
import numpy as np
import geopandas as gpd
import shapely


def extend_transects_seaward(in_transects_geojson,
                             extension_distance):
    """
    Extends transects seaward by a specified distance in meters
    inputs:
    in_transects_geojson (str): path to the transects geojson file
    extension_distance (int): extension distance in meters
    outputs:
    out_transects_geojson (str): path to the output geojson file
    """
    ##loading things in
    in_transects = gpd.read_file(in_transects_geojson)
    
    ##this needed to be dropped to make a shapefile so I could check in gis
    #in_transects = in_transects.drop(columns=['ProcTime'])
    
    crs_wgs84 = in_transects.crs
    crs_utm = in_transects.estimate_utm_crs()
    in_transects_utm = in_transects.to_crs(crs_utm)

    ##making save paths
    out_transects_geojson = os.path.splitext(in_transects_geojson)[0]+'_extended.geojson'
    
    #out_transects_shp = os.path.splitext(in_transects_geojson)[0]+'_extended.shp'
    
    out_transects_utm = in_transects_utm.copy()

    new_lines = [None]*len(in_transects_utm)
    ##loop over all transects
    for i in range(len(in_transects_utm)):
        transect = in_transects_utm.iloc[i]
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]

        ##get bearing and extend end of transect
        angle = np.arctan2(last[1] - first[1], last[0] - first[0])
        new_end_x = last[0]+extension_distance*np.cos(angle)
        new_end_y = last[1]+extension_distance*np.sin(angle)

        newLine = shapely.LineString([first, (new_end_x, new_end_y)])
        new_lines[i] = newLine
    
    out_transects_utm['geometry'] = new_lines

    ##project into wgs84, save file
    out_transects_wgs84 = out_transects_utm.to_crs(crs_wgs84)
    out_transects_wgs84.to_file(out_transects_geojson)

    ##for saving shapefile
    #out_transects_wgs84.to_file(out_transects_shp)
    return out_transects_geojson








        
