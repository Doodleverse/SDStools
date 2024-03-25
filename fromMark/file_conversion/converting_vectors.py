import geopandas as gpd
import os
import glob

def shapefile_to_geojson(my_shape_file, out_dir=None):
    """
    converts shapefile to geojson
    inputs:
    my_shape_file (str): path to the shapefile
    out_dir (optional, str): directory to save to
    if this is not provided then geojson is saved
    to same directory as the input shapefile
    """
    name = os.path.basename(my_shape_file)
    name_no_ext = os.path.splitext(my_shape_file)[0]
    folder = os.path.dirname(my_shape_file)
    
    if out_dir = None:
        new_name = os.path.join(folder, name_no_ext+'.geojson'
    else:
        new_name = os.path.join(out_dir, name_no_ext+'.geojson'
        
    myshpfile = gpd.read_file(my_shape_file)
    myshpfile.to_file(new_name, driver='GeoJSON')

def batch_shapefile_to_geojson(in_dir, out_dir):
    """
    converts directory of shapefiles to geojsons
    calls shapefile_to_geojson
    inputs:
    in_dir (str): path to the directory of shapefiles
    out_dir (str): directory to save to
    
    """
    shapefiles = glob.glob(in_dir + '/*.shp')
    for shp in shapefiles:
        shapefile_to_geojson(shp, out_dir = out_dir)

def geojson_to_shapefile(my_geojson, out_dir=None):
    """
    converts geojson to shapefile
    inputs:
    my_geojson (str): path to the geojson
    out_dir (optional, str): directory to save to
    if this is not provided then shapefile is saved
    to same directory as the input shapefile
    """
    name = os.path.basename(my_geojson)
    name_no_ext = os.path.splitext(my_geojson)[0]
    folder = os.path.dirname(my_geojson)
    
    if out_dir = None:
        new_name = os.path.join(folder, name_no_ext+'.shp'
    else:
        new_name = os.path.join(out_dir, name_no_ext+'.shp'
        
    myshpfile = gpd.read_file(my_geojson)
    myshpfile.to_file(new_name)

def batch_geojson_to_shapefile(in_dir, out_dir):
    """
    converts directory of geojsons to shapefiles
    calls geojson_to_shapefile
    inputs:
    in_dir (str): path to the directory of geojsons
    out_dir (str): directory to save to
    
    """
    geojsons = glob.glob(in_dir + '/*.geojson')
    for jsn in shapefiles:
        geojson_to_shapefile(jsn, out_dir = out_dir)       




        
