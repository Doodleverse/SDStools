"""
Computing beach slopes (or bluff slopes) from DEMs
Just some basic min/max finding
Mark Lundine, USGS
"""
import os
import sys
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np
import glob
from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst
from os.path import realpath
import shapely
from shapely import wkt
from shapely.geometry import LineString
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import argrelextrema
from kneed import KneeLocator
import warnings


plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 14})  # Set font size to 14

# Ignore all warnings
warnings.simplefilter('ignore')

def geojson_to_shapefile(gdf, my_geojson, out_dir):
    """
    converts geojson to shapefile
    inputs:
    my_geojson (str): path to the geojson
    out_dir (optional, str): directory to save to
    if this is not provided then shapefile is saved
    to same directory as the input shapefile
    """
    try:
        os.mkdir(out_dir)
    except:
        pass
    new_name = os.path.splitext(os.path.basename(my_geojson))[0]+r'.shp'
    new_name = os.path.join(out_dir, new_name) 
    myshpfile = gdf
    myshpfile.to_file(new_name)
    return new_name

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def make_profile_plot(csv_file, NO_DATA, res=2, clip_land=0, vertical_datum='WGS84 Ellipsoid'):
    smooth = int(10/res)
    df = pd.read_csv(csv_file)
    filter_df = df[df['elevation']!=NO_DATA]
    filter_df = filter_df.fillna(method='ffill').fillna(method='bfill')

    ##Clip a bunch of land points since the transects started pretty far inland
    #filter_df = filter_df.iloc[clip_land:].reset_index()

    ##Clip sea removes offshore points that were heavilly interpolated
    ##The profiles reach a minimum value and then flatten out
    try:
        clip_sea = len(filter_df)
    except:
        print('no elevation values along transect')
        max_slope = np.nan
        max_tan_beta = np.nan
        avg_slope = np.nan
        avg_tan_beta = np.nan
        median_slope = np.nan
        median_tan_beta = np.nan
        ip_x = np.nan
        ip_y = np.nan
        crest_x = np.nan
        crest_y = np.nan
        crest2_x = np.nan
        crest2_y = np.nan
        crest3_x = np.nan
        crest3_y = np.nan
        toe_x = np.nan
        toe_y = np.nan

        return slope, tan_beta, ip_x, ip_y, crest_x, crest_y, toe_x, toe_y
    
    filter_df = filter_df.iloc[:clip_sea].reset_index()

    ##Getting distances and elevations and instantaneous differences
    ##Smooth out to remove some of the noise so the algorithm works 
    x = filter_df['distance']-(clip_land*res)
    z = filter_df['elevation'].rolling(window=smooth).mean()
    z = z.fillna(method='ffill').fillna(method='bfill')
    diffs = z.diff()/res
    diffs = diffs.rolling(window=smooth).mean()
    diffs = diffs.fillna(method='ffill').fillna(method='bfill')

    ##Get crest point
    ##also get the coordinates of this point
    try:
        crest = np.argmax(z-diffs)
        crest_x = filter_df['x'].iloc[crest]
        crest_y = filter_df['y'].iloc[crest]
        crest2 = np.argmax(z)
        crest2_x = filter_df['x'].iloc[crest2]
        crest2_y = filter_df['y'].iloc[crest2]
        x_search = x[crest2:]
        z_search = z[crest2:]
        crest3 = KneeLocator(list(range(len(x_search))),
                          list(z_search.values),
                          online=True,
                          curve='concave',
                          direction='decreasing',
                          interp_method='polynomial'
                          )
        crest3 = crest3.knee + crest2
        crest3_x = filter_df['x'].iloc[crest3]
        crest3_y = filter_df['y'].iloc[crest3]
    except:
        crest = np.nan
        crest_x = np.nan
        crest_y = np.nan
        crest2 = np.nan
        crest2_x = np.nan
        crest2_y = np.nan
        crest3 = np.nan
        crest3_x = np.nan
        crest3_y = np.nan
    ##get inflection point on profile, maximum negative slope
    ##also get the coordinates of this point
    try:
        diffs2 = diffs[crest:]
        inflection_point = np.argmin(diffs2).flatten()[0]
        inflection_point = inflection_point + crest
        ip_x = filter_df['x'].iloc[inflection_point]
        ip_y = filter_df['y'].iloc[inflection_point]
    except:
        inflection_point = np.nan
        ip_x = np.nan
        ip_y = np.nan
        
    ##get the elbow point (toe), only look for it after the crest
    ##if it is computed before the inflection point, make it the last value in the profile
    ##also get the coordinates of this point
    ##we use the Kneedle algorithm here
    try:
        x_2 = x[crest:]
        z_2 = z[crest:]
        toe = KneeLocator(list(range(len(x_2))),
                          list(z_2.values),
                          online=True,
                          curve='convex',
                          direction='decreasing',
                          interp_method='polynomial'
                          )
        toe = toe.knee + crest
        toe_x = filter_df['x'].iloc[toe]
        toe_y = filter_df['y'].iloc[toe]
        if toe == crest or toe < inflection_point:
            toe = -1
            toe_x = filter_df['x'].iloc[toe]
            toe_y = filter_df['y'].iloc[toe]          
    except:
        toe = np.nan
        toe_x = np.nan
        toe_y = np.nan
    if crest >= toe:
        crest = np.nan
    if crest2 >= toe:
        crest2 = np.nan
    if crest3 >= toe:
        crest3 = np.nan
    try:
        max_slope = abs(diffs[inflection_point])
        max_tan_beta = np.tan(max_slope)
        points = [crest, crest2, crest3, inflection_point, toe]
        min_points = min(points)
        max_points = max(points)
        x_slope = x[min_points:max_points]
        z_slope = z[min_points:max_points]
        diffs_slope = diffs[min_points:max_points]
        avg_slope = abs(np.mean(diffs_slope))
        median_slope = abs(np.median(diffs_slope))
        avg_tan_beta = np.tan(avg_slope)
        median_tan_beta = np.tan(median_slope)
    except:
        max_slope = np.nan
        max_tan_beta = np.nan
        avg_slope = np.nan
        avg_tan_beta = np.nan
        median_slope = np.nan
        median_tan_beta = np.nan


    ##plotting everything (profile, points extracted on profile, computed slope)
    lab = ('Elevation Profile')
    
    plt.subplot(2,1,1)
    plt.plot(x, z, label=lab, color='k')
    plt.fill_between(x, z, -100, color='tan')
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='blue', alpha=0.1)

    lab_max = ('Max \u03B2 = ' +
               str(np.round(max_slope, decimals=4)) +
               '\n Max tan(\u03B2) = ' +
               str(np.round(max_tan_beta, decimals=4)) + '\n'
               )
    lab_avg = ('Mean \u03B2 = ' +
               str(np.round(avg_slope, decimals=4)) +
               '\n Mean tan(\u03B2) = ' +
               str(np.round(avg_tan_beta, decimals=4)) + '\n'
               )
    lab_median = ('Median \u03B2 = ' +
               str(np.round(median_slope, decimals=4)) +
               '\n Median tan(\u03B2) = ' +
               str(np.round(median_tan_beta, decimals=4)) + '\n'
               )
    current_ax = plt.gca()
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, lab_max + lab_avg + lab_median, transform=current_ax.transAxes, fontsize=14,
             verticalalignment='top', bbox=props)
    try:
        plt.scatter(x.iloc[inflection_point], z.iloc[inflection_point], label='Inflection Point', color='green')
        plt.scatter(x.iloc[crest], z.iloc[crest], label = 'Crest', color='blue')
        plt.scatter(x.iloc[crest2], z.iloc[crest2], label = 'Crest2', color='skyblue')
        plt.scatter(x.iloc[crest3], z.iloc[crest3], label = 'Crest3', color='steelblue')
        plt.scatter(x.iloc[toe], z.iloc[toe], label = 'Toe', color='red')
    except:
        pass
    
    plt.xlabel('Cross-Shore Distance (m)')
    plt.ylabel('Elevation\n(m, '+ vertical_datum + ')')
    try:
        plt.xlim(min(x), max(x))
        plt.ylim(min(z), max(z)+4)
    except:
        pass
    plt.legend(loc='best')

    ##also plot the instantaneous slopes with the extracted points
    plt.subplot(2,1,2)
    plt.plot(x, diffs, label='instantaneous slopes', color='blue')
    try:
        plt.xlim(min(x), max(x))
    except:
        pass
    try:
        plt.scatter(x[inflection_point], diffs[inflection_point], color='green', label='Inflection Point')
        plt.scatter(x.iloc[crest], diffs.iloc[crest], label = 'Crest', color='blue')
        plt.scatter(x.iloc[crest2], diffs.iloc[crest2], label = 'Crest2', color='skyblue')
        plt.scatter(x.iloc[crest3], diffs.iloc[crest3], label = 'Crest3', color='steelblue')
        plt.scatter(x.iloc[toe], diffs.iloc[toe], label = 'Toe', color='red')
    except:
        pass
    plt.xlabel('Cross-Shore Distance (m)')
    plt.ylabel('Slope')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.splitext(csv_file)[0]+'_profile.png', dpi=300)
    plt.close()

    output_dict = {'max_slope':max_slope,
                   'max_tan_beta':max_tan_beta,
                   'avg_slope':avg_slope,
                   'avg_tan_beta':avg_tan_beta,
                   'median_slope':median_slope,
                   'median_tan_beta':median_tan_beta,
                   'ip_x':ip_x,
                   'ip_y':ip_y,
                   'crest_x':crest_x,
                   'crest_y':crest_y,
                   'crest2_x':crest2_x,
                   'crest2_y':crest2_y,
                   'crest3_x':crest3_x,
                   'crest3_y':crest3_y,
                   'toe_x':toe_x,
                   'toe_y':toe_y
                   }

    return output_dict


def get_elevation(x_coord, y_coord, raster, bands, gt, NO_DATA):
    """
    get the elevation value of each pixel under
    location x, y
    inpusts:
    x_coord (float): x coordinate
    y_coord (float): y coordinate
    raster: gdal raster open object
    bands (int): number of bands in image
    gt: raster limits
    
    returns:
    elevation (float): value of raster at point x,y
    """
    elevation = []
    xOrigin = gt[0]
    yOrigin = gt[3]
    pixelWidth = gt[1]
    pixelHeight = gt[5]
    px = int((x_coord - xOrigin) / pixelWidth)
    py = int((y_coord - yOrigin) / pixelHeight)
    for j in range(bands):
        band = raster.GetRasterBand(j + 1)
        data = band.ReadAsArray(px, py, 1, 1)
        try:
            elevation.append(data[0][0])
        except:
            elevation.append(NO_DATA)
    return elevation


def write_to_csv(csv_out,result_profile_x_z):
    # check if output file exists on disk if yes delete it
    if os.path.isfile(csv_out):
        os.remove(csv_out)
   
    # create new CSV file containing X (distance) and Z value pairs
    with open(csv_out, 'a') as outfile:
        # write first row column names into CSV
        outfile.write("distance,elevation,x,y" + "\n")
        # loop through each pair and write to CSV
        for dist, z, x, y in result_profile_x_z:
            outfile.write(str(round(dist, 2)) + ',' + str(round(z, 2)) + ',' + str(x) + ',' + str(y) + '\n')
           

def main(in_raster, in_line, csv_file, res, NO_DATA, batch=False, vertical_datum = 'WGS84 Ellipsoid'):
    """
    Extracts elevation profile given an input raster dem and an input shapefile line
    inputs:
    in_raster: path to raster dem
    in_line: path to shapefile profile line
    csv_file: path to save extracted distance, elevation pairs to
    res: resolution to sample at in meters
    N0_DATA: raster no data value
    batch (optional): this should stay as False if just one profile is taken, use batch_main function for multiple profiles
    """
    # open the image
    ds = gdal.Open(in_raster, gdalconst.GA_ReadOnly)
   
    if ds is None:
        print('Could not open image')
        sys.exit(1)
   
    # get raster bands
    bands = ds.RasterCount
   
    # get georeference info
    transform = ds.GetGeoTransform()
    if batch == False:
        # line defining the profile
        line_file = ogr.Open(in_line)
        shape = line_file.GetLayer(0)
        #first feature of the shapefile
        line = shape.GetFeature(0)
        line_geom = line.geometry().ExportToWkt()
        shapely_line = LineString(wkt.loads(line_geom))
        # length in meters of profile line
        length_m = shapely_line.length
    else:
        line_geom = in_line.geometry().ExportToWkt()
        shapely_line = LineString(wkt.loads(line_geom))
        # length in meters of profile line
        length_m = shapely_line.length
    # lists of coords and elevations
    x = []
    y = []
    z = []
    # distance of the topographic profile
    distance = []
    for currentdistance in range(0, int(length_m), res):
        # creation of the point on the line
        point = shapely_line.interpolate(currentdistance)
        xp, yp = point.x, point.y
        x.append(xp)
        y.append(yp)
        # extraction of the elevation value from the MNT
        z.append(get_elevation(xp, yp, ds, bands, transform, NO_DATA)[0])
        distance.append(currentdistance)  
   
    # combine distance and elevation vales as pairs
    profile_x_z = zip(distance,z, x, y)
   
    # output final csv data
    write_to_csv(csv_file, profile_x_z)
    output_dict = make_profile_plot(csv_file, NO_DATA, res=res, vertical_datum=vertical_datum)
    return output_dict

def get_no_data_value(in_raster):
    """
    gets no data value from raster
    inputs:
    in_raster (str): path to geotiff
    outputs:
    no_data_value (float): no data value
    """
    with rasterio.open(in_raster) as src:
        no_data_value = src.nodata

    return no_data_value

def batch_main(in_raster, in_lines_path, out_folder, res, section_string, v, crs=6393, vertical_datum='WGS84 Ellipsoid'):
    """
    Repeatedly take elevation profiles from a raster dem with an input shapefile containing all of the lines
    inputs:
    in_raster (str): path to raster dem
    in_lines (str): path to shapefile containing profile lines
    out_folder (str): path to folder to save csvs and png figures to
    res (float): resolution to sample at in meters
    section_string (str): shoreline section
    v (str): transect version
    crs (int): epsg code
    vertical_datum (str): string for plotting defining the vertical datum
    """
    try:
        os.mkdir(out_folder)
    except:
        pass
    NO_DATA = get_no_data_value(in_raster)
    shp_folder = os.path.join(os.path.dirname(in_lines_path), 'transects_shapefile')
    in_lines = gpd.read_file(in_lines_path)
    in_lines = in_lines.to_crs(epsg=crs)
    in_lines_shp = geojson_to_shapefile(in_lines, in_lines_path, shp_folder)
    line_file = ogr.Open(in_lines_shp)
    layer = line_file.GetLayer(0)
    max_slopes = [None]*len(layer)
    max_tan_betas = [None]*len(layer)
    avg_slopes = [None]*len(layer)
    avg_tan_betas = [None]*len(layer)
    median_slopes = [None]*len(layer)
    median_tan_betas = [None]*len(layer)
    ip_xs = [None]*len(layer)
    ip_ys = [None]*len(layer)
    crest_xs = [None]*len(layer)
    crest_ys = [None]*len(layer)
    crest2_xs = [None]*len(layer)
    crest2_ys = [None]*len(layer)
    crest3_xs = [None]*len(layer)
    crest3_ys = [None]*len(layer)
    toe_xs = [None]*len(layer)
    toe_ys = [None]*len(layer)
    transect_ids = [None]*len(layer)
    i=0
    for feature in layer:
        print(i/len(layer)*100)
        transect_id = section_string+v+str(i*50).zfill(6)
        csv_path = os.path.join(out_folder, transect_id + '.csv')
        output_dict = main(in_raster, feature, csv_path, res, NO_DATA, batch=True, vertical_datum=vertical_datum)
        transect_ids[i] = transect_id
        max_slopes[i] = output_dict['max_slope']
        max_tan_betas[i] = output_dict['max_tan_beta']
        avg_slopes[i] = output_dict['avg_slope']
        avg_tan_betas[i] = output_dict['avg_tan_beta']
        median_slopes[i] = output_dict['median_slope']
        median_tan_betas[i] = output_dict['median_tan_beta']
        ip_xs[i] = output_dict['ip_x']
        ip_ys[i] = output_dict['ip_y']
        crest_xs[i] = output_dict['crest_x']
        crest_ys[i] = output_dict['crest_y']
        crest2_xs[i] = output_dict['crest2_x']
        crest2_ys[i] = output_dict['crest2_y']
        crest3_xs[i] = output_dict['crest3_x']
        crest3_ys[i] = output_dict['crest3_y']
        toe_xs[i] = output_dict['toe_x']
        toe_ys[i] = output_dict['toe_y']
        i=i+1
    df_dict = {'transect_id':transect_ids,
               'max_slope':max_slopes,
               'max_tan_beta':max_tan_betas,
               'avg_slope':avg_slopes,
               'avg_tan_beta':avg_tan_betas,
               'median_slope':median_slopes,
               'median_tan_beta':median_tan_betas,
               'inflection_x':ip_xs,
               'inflection_y':ip_ys,
               'crest_x':crest_xs,
               'crest_y':crest_ys,
               'crest2_x':crest2_xs,
               'crest2_y':crest2_ys,
               'crest3_x':crest3_xs,
               'crest3_y':crest3_ys,
               'toe_x':toe_xs,
               'toe_y':toe_ys,
               }
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(os.path.dirname(out_folder), section_string+'_slopes.csv'))

    ##remove temporary shapefile
    os.rmtree(shp_folder)
    
    return df

def batch_main_custom(site, in_raster, in_lines_path, out_folder, res, crs=6393, vertical_datum='WGS84 Ellipsoid'):
    """
    Repeatedly take elevation profiles from a raster dem with an input shapefile containing all of the lines
    inputs:
    site (str): site name
    in_raster (str): path to raster dem
    in_lines (str): path to shapefile containing profile lines
    out_folder (str): path to folder to save csvs and png figures to
    res (float): resolution to sample at in meters
    crs (int): epsg code
    vertical_datum (str): string for plotting defining the vertical datum
    """
    try:
        os.mkdir(out_folder)
    except:
        pass
    NO_DATA = get_no_data_value(in_raster)
    shp_folder = os.path.join(os.path.dirname(in_lines_path), 'transects_shapefile')
    in_lines = gpd.read_file(in_lines_path)
    in_lines = in_lines.to_crs(epsg=crs)
    in_lines_shp = geojson_to_shapefile(in_lines, in_lines_path, shp_folder)
    line_file = ogr.Open(in_lines_shp)
    layer = line_file.GetLayer(0)
    max_slopes = [None]*len(layer)
    max_tan_betas = [None]*len(layer)
    avg_slopes = [None]*len(layer)
    avg_tan_betas = [None]*len(layer)
    median_slopes = [None]*len(layer)
    median_tan_betas = [None]*len(layer)
    ip_xs = [None]*len(layer)
    ip_ys = [None]*len(layer)
    crest_xs = [None]*len(layer)
    crest_ys = [None]*len(layer)
    crest2_xs = [None]*len(layer)
    crest2_ys = [None]*len(layer)
    crest3_xs = [None]*len(layer)
    crest3_ys = [None]*len(layer)
    toe_xs = [None]*len(layer)
    toe_ys = [None]*len(layer)
    transect_ids = [None]*len(layer)
    i=0
    for feature in layer:
        print(i/len(layer)*100)
        transect_id = str(i)
        csv_path = os.path.join(out_folder, transect_id + '.csv')
        output_dict = main(in_raster, feature, csv_path, res, NO_DATA, batch=True, vertical_datum=vertical_datum)
        transect_ids[i] = transect_id
        max_slopes[i] = output_dict['max_slope']
        max_tan_betas[i] = output_dict['max_tan_beta']
        avg_slopes[i] = output_dict['avg_slope']
        avg_tan_betas[i] = output_dict['avg_tan_beta']
        median_slopes[i] = output_dict['median_slope']
        median_tan_betas[i] = output_dict['median_tan_beta']
        ip_xs[i] = output_dict['ip_x']
        ip_ys[i] = output_dict['ip_y']
        crest_xs[i] = output_dict['crest_x']
        crest_ys[i] = output_dict['crest_y']
        crest2_xs[i] = output_dict['crest2_x']
        crest2_ys[i] = output_dict['crest2_y']
        crest3_xs[i] = output_dict['crest3_x']
        crest3_ys[i] = output_dict['crest3_y']
        toe_xs[i] = output_dict['toe_x']
        toe_ys[i] = output_dict['toe_y']
        i=i+1
    df_dict = {'transect_id':transect_ids,
               'max_slope':max_slopes,
               'max_tan_beta':max_tan_betas,
               'avg_slope':avg_slopes,
               'avg_tan_beta':avg_tan_betas,
               'median_slope':median_slopes,
               'median_tan_beta':median_tan_betas,
               'inflection_x':ip_xs,
               'inflection_y':ip_ys,
               'crest_x':crest_xs,
               'crest_y':crest_ys,
               'crest2_x':crest2_xs,
               'crest2_y':crest2_ys,
               'crest3_x':crest3_xs,
               'crest3_y':crest3_ys,
               'toe_x':toe_xs,
               'toe_y':toe_ys,
               }
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(os.path.dirname(out_folder), site+'_slopes.csv'))

    ##remove temporary shapefile
    os.rmtree(shp_folder)
    
    return df

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
    saves output with '_smooth' appended to original filename in same directory

    inputs:
    shorelines (str): path to extracted shorelines
    outputs:
    save_path (str): path of output file
    """
    dirname = os.path.dirname(shorelines)
    save_path = os.path.join(dirname,os.path.splitext(os.path.basename(shorelines))[0]+'_smooth.geojson')
    lines = gpd.read_file(shorelines)
    lines = simplify_lines(lines)
    new_lines = lines.copy()
    for i in range(len(lines)):
        line = lines.iloc[i]
        coords = LineString_to_arr(line['geometry'])
        refined = chaikins_corner_cutting(coords)
        refined_geom = arr_to_LineString(refined)
        new_lines['geometry'][i] = refined_geom

    new_lines = new_lines.to_crs('epsg:4326')
    new_lines.to_file(save_path)
    return new_lines

def simplify_lines(lines, tolerance=50):
    """
    Uses shapely simplify function to smooth out the extracted shorelines
    """
    lines['geometry'] = lines['geometry'].simplify(tolerance)
    return lines

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs: line
    outputs: array of xy tuples
    """
    listarray = []
    for pp in line.coords:
        listarray.append((pp[0], pp[1]))
    nparray = np.array(listarray)
    return nparray

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

def geojson_points_to_lines(gdf, epsg):
    points = [None]*len(gdf)
    i=0
    for point in gdf['geometry']:
        points[i] = (point.x, point.y)
        i=i+1
    line = LineString(points)
    new_gdf_dict = {'id':[0]}
    new_gdf_df = pd.DataFrame(new_gdf_dict)
    new_gdf = gpd.GeoDataFrame(new_gdf_df)
    new_gdf['geometry'] = line
    new_gdf = new_gdf.set_crs(epsg=epsg)
    return new_gdf
        

#utqiagvik tbdem crs = 6393
#arctic dem crs = 3413

def make_vrt(folder_path, vrt_path):
    """
    Makes a vrt and saves it in the folder of rasters provided
    inputs:
    folder_path (str): path to folder of geotiffs
    vrt_path (str): path to the vrt file to save
    outputs:
    vrt_path (str): path to the vrt file
    """
    print(folder_path)
    filepaths = glob.glob(folder_path + '/*.tif')
    print(filepaths)
    gdal.BuildVRT(vrt_path, filepaths)      
    return vrt_path

def profile_shoreline_section(G,
                              C,
                              RR,
                              V,
                              RR_home,
                              crs=3413,
                              res=2,
                              all_sections=False,
                              custom_sections = None,
                              arctic_dem=True,
                              custom_dem=None,
                              vertical_datum='WGS84 Ellipsoid'):
    """
    Makes elevation profiles and computes beach slope from profile
    Be sure to check the elevation profiles to ensure that that slopes make sense and the detectected points are correct
    Specific for Alaska shoreline sections

    Outputs:
    --transects geojson with slopes
    --inflection point lines 
    --three different crest lines
    --toe line
    --profile figures and profile csvs (cross_distance, elevation) for each transect

    inputs:
    G (str): global region
    C (str): coastal area
    RR (str): subregion
    V (str): version
    RR_home (str): path to subregion
    crs (int): epsg code for raster DEM
    res (float): horizontal resolution of raster
    all_sections (bool): profile all shoreline sections in subregion or not
    custom_sections (list): list of sections to profile, for all sections just feed None
    arctic_dem (bool): whether or not to profile arctic dem or not
    custom_dem (str): path to the custom DEM to profile
    vertical_datum (str): just a string for plotting to define the vertical datum
    """
    if custom_sections != None:
        sections = custom_sections
    else:
        sections = get_immediate_subdirectories(r_home)
    for section in sections:
        print(section)
        section_dir = os.path.join(RR_home, section)
        section_string = G+C+RR+section[3:]
        transects_path = os.path.join(section_dir, section_string + '_transects.geojson')
        if arctic_dem == True:
            res=2
            crs=3413
            raster_path = os.path.join(section_dir, 'DEMs', section_string+'_artic_DEM.tif')
        else:
            raster_path = custom_dem
        out_folder = os.path.join(section_dir, 'elevation_profiles')
        try:
            os.mkdir(out_folder)
        except:
            pass
        
        df = batch_main(raster_path, transects_path, out_folder, res, section_string, V, crs=crs, vertical_datum=vertical_datum)

        df_no_nans = df.dropna()

        ##make geojsons for inflection points, crest points, toe points
        ##inflection
        ##3413 is polar stereo north
        ip_points = gpd.GeoDataFrame(df_no_nans,
                                     geometry=gpd.points_from_xy(df_no_nans['inflection_x'],
                                                                 df_no_nans['inflection_y']),
                                     crs=crs)
        ip_lines = geojson_points_to_lines(ip_points, crs)
        ip_lines_path = os.path.join(section_dir, section_string + '_inflection_points.geojson')
        ip_lines.to_file(ip_lines_path)
        ip_lines = smooth_lines(ip_lines_path)

        ##crest
        crest_points = gpd.GeoDataFrame(df_no_nans,
                                      geometry=gpd.points_from_xy(df_no_nans['crest_x'],
                                                                  df_no_nans['crest_y']),
                                      crs=crs)
        crest_lines = geojson_points_to_lines(crest_points, crs)
        crest_lines_path = os.path.join(section_dir, section_string + '_crest_points.geojson')
        crest_lines.to_file(crest_lines_path)
        crest_lines = smooth_lines(crest_lines_path)

        ##crest2
        crest2_points = gpd.GeoDataFrame(df_no_nans,
                                      geometry=gpd.points_from_xy(df_no_nans['crest2_x'],
                                                                  df_no_nans['crest2_y']),
                                      crs=crs)
        crest2_lines = geojson_points_to_lines(crest2_points, crs)
        crest2_lines_path = os.path.join(section_dir, section_string + '_crest2_points.geojson')
        crest2_lines.to_file(crest2_lines_path)
        crest2_lines = smooth_lines(crest2_lines_path)

        ##crest3
        crest3_points = gpd.GeoDataFrame(df_no_nans,
                                      geometry=gpd.points_from_xy(df_no_nans['crest3_x'],
                                                                  df_no_nans['crest3_y']),
                                      crs=crs)
        crest3_lines = geojson_points_to_lines(crest3_points, crs)
        crest3_lines_path = os.path.join(section_dir, section_string + '_crest3_points.geojson')
        crest3_lines.to_file(crest3_lines_path)
        crest3_lines = smooth_lines(crest3_lines_path)

        ##toe
        toe_points = gpd.GeoDataFrame(df_no_nans,
                                      geometry=gpd.points_from_xy(df_no_nans['toe_x'],
                                                                  df_no_nans['toe_y']),
                                      crs=crs)
        toe_lines = geojson_points_to_lines(toe_points, crs)
        toe_lines_path = os.path.join(section_dir, section_string + '_toe_points.geojson')
        toe_lines.to_file(toe_lines_path)
        toe_lines = smooth_lines(toe_lines_path)



        transects_df = gpd.read_file(transects_path)
        transects_df['transect_id'] = transects_df['transect_id'].astype(int)
        df['transect_id'] = df['transect_id'].astype(int)
        transects_df = transects_df.merge(df, on='transect_id', suffixes=['_transects', '_slopes'])
        transects_df.to_file(os.path.splitext(transects_path)[0]+'_slopes.geojson')

def generic_profile(site,
                    dem_path,
                    transects_path,
                    output_folder,
                    crs,
                    res,
                    vertical_datum):
    """
    Makes elevation profiles and computes beach slope from profile
    Be sure to check the elevation profiles to ensure that that slopes make sense and the detectected points are correct

    Outputs:
    --transects geojson with slopes
    --inflection point lines 
    --three different crest lines
    --toe line
    --profile figures and profile csvs (cross_distance, elevation) for each transect

    inputs:
    site (str): site name
    dem_path (str): path to the DEM (.tif)
    transects_path (str): path to the transects (.geojson)
    output_folder (str): path to save outputs to
    crs (int): epsg code for raster DEM
    res (float): horizontal resolution of raster
    vertical_datum (str): just a string for plotting to define the vertical datum
    """

    raster_path = dem_path
    try:
        os.mkdir(output_folder)
    except:
        pass
    
    df = batch_main_custom(site, raster_path, transects_path, out_folder, res, crs=crs, vertical_datum=vertical_datum)

    df_no_nans = df.dropna()

    ##make geojsons for inflection points, crest points, toe points
    ##inflection
    ##3413 is polar stereo north
    ip_points = gpd.GeoDataFrame(df_no_nans,
                                 geometry=gpd.points_from_xy(df_no_nans['inflection_x'],
                                                             df_no_nans['inflection_y']),
                                 crs=crs)
    ip_lines = geojson_points_to_lines(ip_points, crs)
    ip_lines_path = os.path.join(output_folder, site + '_inflection_points.geojson')
    ip_lines.to_file(ip_lines_path)
    ip_lines = smooth_lines(ip_lines_path)

    ##crest
    crest_points = gpd.GeoDataFrame(df_no_nans,
                                  geometry=gpd.points_from_xy(df_no_nans['crest_x'],
                                                              df_no_nans['crest_y']),
                                  crs=crs)
    crest_lines = geojson_points_to_lines(crest_points, crs)
    crest_lines_path = os.path.join(output_folder, site +'_crest_points.geojson')
    crest_lines.to_file(crest_lines_path)
    crest_lines = smooth_lines(crest_lines_path)

    ##crest2
    crest2_points = gpd.GeoDataFrame(df_no_nans,
                                  geometry=gpd.points_from_xy(df_no_nans['crest2_x'],
                                                              df_no_nans['crest2_y']),
                                  crs=crs)
    crest2_lines = geojson_points_to_lines(crest2_points, crs)
    crest2_lines_path = os.path.join(output_folder, site + '_crest2_points.geojson')
    crest2_lines.to_file(crest2_lines_path)
    crest2_lines = smooth_lines(crest2_lines_path)

    ##crest3
    crest3_points = gpd.GeoDataFrame(df_no_nans,
                                  geometry=gpd.points_from_xy(df_no_nans['crest3_x'],
                                                              df_no_nans['crest3_y']),
                                  crs=crs)
    crest3_lines = geojson_points_to_lines(crest3_points, crs)
    crest3_lines_path = os.path.join(output_folder, site + '_crest3_points.geojson')
    crest3_lines.to_file(crest3_lines_path)
    crest3_lines = smooth_lines(crest3_lines_path)

    ##toe
    toe_points = gpd.GeoDataFrame(df_no_nans,
                                  geometry=gpd.points_from_xy(df_no_nans['toe_x'],
                                                              df_no_nans['toe_y']),
                                  crs=crs)
    toe_lines = geojson_points_to_lines(toe_points, crs)
    toe_lines_path = os.path.join(output_folder, site + '_toe_points.geojson')
    toe_lines.to_file(toe_lines_path)
    toe_lines = smooth_lines(toe_lines_path)



    transects_df = gpd.read_file(transects_path)
    transects_df['transect_id'] = transects_df['transect_id'].astype(int)
    df['transect_id'] = df['transect_id'].astype(int)
    transects_df = transects_df.merge(df, on='transect_id', suffixes=['_transects', '_slopes'])
    transects_df.to_file(output_folder, site + '_transects_slopes.geojson')
