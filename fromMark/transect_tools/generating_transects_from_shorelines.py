from osgeo import ogr, gdal
gdal.UseExceptions() 
from shapely.geometry import MultiLineString, LineString, Point
from shapely import wkt
import sys, math
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely

def break_into_parts(input_shorelines):
    new_dir = os.path.join(os.path.dirname(input_shorelines), 'split')
    try:
        os.mkdir(new_dir)
    except:
        pass
    gdf = gpd.read_file(input_shorelines)
    crs = gdf.crs
    new_paths = [None]*len(gdf)
    for i in range(len(gdf)):
        row = gdf.iloc[i]
        print(row)
        new_gdf = gpd.GeoDataFrame({'geometry':row.geometry},
                                   index=[0],
                                   crs=crs)
        new_name = os.path.join(new_dir, 'shore'+str(i)+'.geojson')
        new_gdf.to_file(new_name)
        new_paths[i] = new_name
    return new_paths

def wgs84_to_utm(file):
    in_gdf = gpd.read_file(file)
    utm_crs = in_gdf.estimate_utm_crs()
    out_gdf = in_gdf.to_crs(utm_crs)
    new_name = os.path.splitext(file)[0]+'_utm.geojson'
    out_gdf.to_file(new_name)
    return new_name
## http://wikicode.wikidot.com/get-angle-of-line-between-two-points
## angle between two points
def getAngle(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.degrees(math.atan2(y_diff, x_diff))


## start and end points of chainage tick
## get the first end point of a tick
def getPoint1(pt, bearing, dist):
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)


## get the second end point of a tick
def getPoint2(pt, bearing, dist):
    bearing = math.radians(bearing)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)

def shapefile_to_geojson(my_shape_file, out_dir=None):
    """
    converts shapefile to geojson
    inputs:
    my_shape_file (str): path to the shapefile
    out_dir (optional, str): directory to save to
    if this is not provided then geojson is saved
    to same directory as the input shapefile
    """
    if out_dir == None:
        new_name = os.path.splitext(os.path.basename(myshpfile))[0]+r'.geojson'
    else:
        new_name = os.path.join(out_dir, new_name)
        
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
    if out_dir == None:
        new_name = os.path.splitext(my_geojson)[0]+r'.shp'
    else:
        new_name = os.path.join(out_dir, new_name)
        
    myshpfile = gpd.read_file(my_geojson)
    myshpfile.to_file(new_name)
    return new_name

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

def make_transects(input_path,
                   transect_spacing,
                   transect_length):
    """
    Generates normal transects to an input line shapefile
    inputs:
    input_path: path to shapefile containing the input line
    transect_spacing: distance between each transect in meters
    transect_length: length of each transect in meters
    outputs:
    output_path: path to output shapefile containing transects
    """

    output_path = os.path.splitext(input_path)[0]+'_transects_'+str(transect_spacing)+'m.shp'
    ## set the driver for the data
    driver = ogr.GetDriverByName("Esri Shapefile")
    
    ## open the shapefile in write mode (1)
    ds = driver.Open(input_path)
    shape = ds.GetLayer(0)
    
    ## distance between each points
    distance = transect_spacing
    ## the length of each tick
    tick_length = transect_length

    ## output tick line fc name
    ds_out = driver.CreateDataSource(output_path)
    layer_out = ds_out.CreateLayer('line',shape.GetSpatialRef(),ogr.wkbLineString)

    ## list to hold all the point coords
    list_points = []


    ## distance/chainage attribute
    chainage_fld = ogr.FieldDefn("CHAINAGE", ogr.OFTReal)
    layer_out.CreateField(chainage_fld)
    ## check the geometry is a line
    first_feat = shape.GetFeature(0)

    ln = first_feat
    ## list to hold all the point coords
    list_points = []
    ## set the current distance to place the point
    current_dist = distance
    ## get the geometry of the line as wkt
    line_geom = ln.geometry().ExportToWkt()
    ## make shapely LineString object
    shapely_line = LineString(wkt.loads(line_geom))
    ## get the total length of the line
    line_length = shapely_line.length
    ## append the starting coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[0]))
    ## https://nathanw.net/2012/08/05/generating-chainage-distance-nodes-in-qgis/
    ## while the current cumulative distance is less than the total length of the line
    while current_dist < line_length:
        ## use interpolate and increase the current distance
        list_points.append(shapely_line.interpolate(current_dist))
        current_dist += distance
    ## append end coordinate to the list
    list_points.append(Point(list(shapely_line.coords)[-1]))

    ## add lines to the layer
    ## this can probably be cleaned up better
    ## but it works and is fast!
    for num, pt in enumerate(list_points, 1):
        ## start chainage 0
        if num == 1:
            angle = getAngle(pt, list_points[num])
            line_end_1 = getPoint1(pt, angle, tick_length/2)
            angle = getAngle(line_end_1, pt)
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", 0)
            layer_out.CreateFeature(feat_ln)

        ## everything in between
        if num < len(list_points) - 1:
            angle = getAngle(pt, list_points[num])
            line_end_1 = getPoint1(list_points[num], angle, tick_length/2)
            angle = getAngle(line_end_1, list_points[num])
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", distance * num)
            layer_out.CreateFeature(feat_ln)

        ## end chainage
        if num == len(list_points):
            angle = getAngle(list_points[num - 2], pt)
            line_end_1 = getPoint1(pt, angle, tick_length/2)
            angle = getAngle(line_end_1, pt)
            line_end_2 = getPoint2(line_end_1, angle, tick_length)
            tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])
            feat_dfn_ln = layer_out.GetLayerDefn()
            feat_ln = ogr.Feature(feat_dfn_ln)
            feat_ln.SetGeometry(ogr.CreateGeometryFromWkt(tick.wkt))
            feat_ln.SetField("CHAINAGE", int(line_length))
            layer_out.CreateFeature(feat_ln)

    del ds
    return output_path

def chaikins_corner_cutting(coords, refinements=10):
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
def smooth_lines(shorelines):
    """
    Smooths out shorelines with Chaikin's method
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    shorelines (str): path to extracted shorelines
    outputs:
    save_path (str): path of output file
    """
    dirname = os.path.dirname(shorelines)
    save_path = os.path.join(dirname,os.path.splitext(os.path.basename(shorelines))[0]+'_smooth.shp')
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

def shorelines_to_transects(input_shorelines_path,
                            output_transects_path,
                            transect_spacing=50,
                            transect_length=1000):
    """
    Constructs transects given a geojson of a bunch of shorelines
    Smooths the input shorelines out to help with making better output transects
    Source crs must be wgs84
    Output crs will be wgs84
    inputs:
    input_shorelines_path (str): path to the geojson with reference shorelines
    output_transects_path (str): geojson path to save the output transects to
    """
    
    input_shorelines = gpd.read_file(input_shorelines_path)
    wgs84_crs = input_shorelines.crs
    utm_crs = input_shorelines.estimate_utm_crs()
    input_shorelines = input_shorelines.to_crs(utm_crs)
    utm_path_input = os.path.splitext(input_shorelines_path)[0]+'utm.shp'
    input_shorelines.to_file(utm_path_input)
    smooth_shorelines = smooth_lines(utm_path_input)

    ##Original code works on shapefiles in utm and also on only one shoreline at a time
    lines = break_into_parts(smooth_shorelines)
    out_transects = [None]*len(lines)
    i=0
    for line in lines:
        input_shorelines_shp = geojson_to_shapefile(line, out_dir=None)
        out_transect = make_transects(input_shorelines_shp,
                                      transect_spacing,
                                      transect_length)
        out_transects[i] = out_transect
        i=i+1

    merged_gdfs = [gpd.read_file(g).to_crs(wgs84_crs) for g in out_transects]
    merged = gpd.GeoDataFrame(pd.concat(merged_gdfs))
    merged.to_file(output_transects_path)
shorelines_to_transects(r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\Elim2Unalakeet_transects\Elim2Unalakeet_shoreline.geojson',
                        r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\Elim2Unalakeet_transects\Elim2Unalakeet_transects_50m.geojson',
                        transect_spacing=50,
                        transect_length=1000)

##input_shorelines_path = r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_3.geojson'
##input_shorelines = gpd.read_file(input_shorelines_path)
##wgs84_crs = input_shorelines.crs
##utm_crs = input_shorelines.estimate_utm_crs()
##input_shorelines = input_shorelines.to_crs(utm_crs)
##utm_path_input = os.path.splitext(input_shorelines_path)[0]+'utm.shp'
##input_shorelines.to_file(utm_path_input)
##smooth_lines(utm_path_input)
##make_transects(r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_3utm_smooth.shp',
##               50,
##               1000)
##input_shorelines_path = r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_4.geojson'
##input_shorelines = gpd.read_file(input_shorelines_path)
##wgs84_crs = input_shorelines.crs
##utm_crs = input_shorelines.estimate_utm_crs()
##input_shorelines = input_shorelines.to_crs(utm_crs)
##utm_path_input = os.path.splitext(input_shorelines_path)[0]+'utm.shp'
##input_shorelines.to_file(utm_path_input)
##smooth_lines(utm_path_input)
##make_transects(r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_4utm_smooth.shp',
##               50,
##               1000)
##input_shorelines_path = r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_5.geojson'
##input_shorelines = gpd.read_file(input_shorelines_path)
##wgs84_crs = input_shorelines.crs
##utm_crs = input_shorelines.estimate_utm_crs()
##input_shorelines = input_shorelines.to_crs(utm_crs)
##utm_path_input = os.path.splitext(input_shorelines_path)[0]+'utm.shp'
##input_shorelines.to_file(utm_path_input)
##smooth_lines(utm_path_input)
##make_transects(r'C:\Users\mlundine\OneDrive - DOI\MarkLundine\Data\Alaska\AlaskaRefShoreline\AlaskaRefShoreline_5utm_smooth.shp',
##               50,
##               1000)
