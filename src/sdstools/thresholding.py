import geopandas as gpd
import pandas as pd
import os
import glob
import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def get_script_path():
    return os.path.dirname(os.path.abspath(__file__))

def compute_otsu_threshold(in_tiff, out_tiff):
    """
    Otsu binary thresholding on a geotiff.
    inputs:
    in_tiff (str): path to the input geotiff
    out_tiff (str): path to the output geotiff
    outputs:
    out_tiff (str): path to the output geotiff
    """
    def rescale(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)
    
    def average_img(images_list):
        # Alternative method using numpy mean function
        images = np.array(images_list)
        arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
        return arr
    
    with rasterio.open(in_tiff) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        nir = src.read(4)
        swir = src.read(5)
            
    ndwi = (green - nir)/(green+nir)
    mndwi = (green - swir)/(green+swir)
    rgb = np.dstack([red,green,blue])
    rgb_disp = 255.0*rescale(np.dstack([red,green,blue]))
    stacked = np.dstack([red,green,blue,nir,swir,ndwi,mndwi])
    
    # Compute Otsu's threshold
    # Need to make nodata values zero or else the threshold will be just data vs. nodata    
    swir[swir==src.meta['nodata']]=0
    nir[nir==src.meta['nodata']]=0
    blue[blue==src.meta['nodata']]=0
    red[red==src.meta['nodata']]=0
    green[green==src.meta['nodata']]=0

    stacked[stacked==src.meta['nodata']]=0
    rgb[rgb==src.meta['nodata']]=0

    ##compute thresholds
    thresholds_swir = threshold_multiotsu(swir)
    thresholds_nir = threshold_multiotsu(nir)

    # Apply the threshold to create a binary image
    binary_image_swir = swir > min(thresholds_swir)
    binary_image_nir = nir > min(thresholds_nir)

    # comparison plot, not much difference between swir and nir
    plt.subplot(3,1,1)
    plt.title('RGB')
    plt.imshow(rgb_disp.astype('uint8'))
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(3,1,2)
    plt.title('Otsu Threshold SWIR')
    plt.imshow(rgb_disp.astype('uint8'))
    plt.imshow(binary_image_swir, alpha=0.25, cmap='jet')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.subplot(3,1,3)
    plt.title('Otsu Threshold NIR')
    plt.imshow(rgb_disp.astype('uint8'))
    plt.imshow(binary_image_nir, alpha=0.25, cmap='jet')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.savefig(os.path.splitext(out_tiff)[0]+'_results.jpg',dpi=500)
    plt.close('all')
    
    # Define the metadata for the new geotiff
    transform = from_origin(src.bounds.left, src.bounds.top, src.res[0], src.res[1])
    new_meta = src.meta.copy()
    new_meta.update({
        'dtype': 'uint8',
        'count': 1,
        'transform': transform,
        'nodata':None
    })

    # Save the binary image
    with rasterio.open(out_tiff, 'w', **new_meta) as dst:
        dst.write(binary_image_nir.astype(np.uint8), 1)
    
    return out_tiff

def batch_compute_otsu_threshold(images, out_folder):
    """
    Compute Otsu threshold on list of images

    inputs:
    images (list): list of ms tiffs
    out_folder (str): path to save outputs to

    outputs:
    
    """
    try:
        os.mkdir(out_folder)
    except:
        pass
    for image in images:
        name = os.path.basename(image)
        satname = os.path.basename(os.path.dirname(os.path.dirname(image)))
        name_no_ext = os.path.splitext(name)[0]
        out_image = os.path.join(out_folder, name_no_ext + '_otsu.tif')
        compute_otsu_threshold(image, out_image)
        
    return out_folder

def binary_raster_to_vector(in_tiff, out_geojson):
    """
    Converts a binary raster to a vector file using gdal_polygonize.
    Uses itself as a mask to isolate the raster where cell values == 1.
    Currently running gdal_polygonize as a script from os.system, might want to change this
    inputs:
    in_tiff (str): path to the input binary raster as geotiff
    out_geojson (str): path to the output vector file as geojson
    outputs:
    out_geojson (str): path to the output vector file as geojson
    """
    #module = os.path.join(get_script_path(), 'gdal_polygonize.py')
    cmd = 'gdal_contour -a val ' + in_tiff + ' ' + out_geojson + ' -i 1'
    os.system(cmd)
    return out_geojson

def batch_binary_raster_to_vector(in_folder, out_folder):
    """
    Converts a binary raster to a vector file using gdal_contour.

    inputs:
    in_folder (str): path to input otsu threshold rasters (.tif)
    out_folder (str): path to dir to save contours to

    outputs:
    out_folder (str): path to dir to save contours to
    """
    try:
        os.mkdir(out_folder)
    except:
        pass
    images = glob.glob(in_folder + '/*.tif')
    for image in images:
        name = os.path.basename(image)
        name_no_ext = os.path.splitext(name)[0]
        out_geojson = os.path.join(out_folder, name_no_ext + '_contours.geojson')
        binary_raster_to_vector(image, out_geojson)    
    return out_folder

def contours_to_lines(in_geojson, out_geojson, date):
    """
    Converts contour to shoreline

    inputs:
    in_geojson (str): path to the contour (.geojson)
    out_geojson (str): path to save the shoreline to (.geojson)
    date (str): datetime str in '%Y-%m-%d-%H-%M-%S' format

    outputs:
    out_geojson (str): path to save shoreline to (.geojson)
    """
    gdf = gpd.read_file(in_geojson)
    gdf = gdf[gdf['val']==1]
    gdf['date'] = date
    gdf['date'] = pd.to_datetime(gdf['date'], utc=True, format = '%Y-%m-%d-%H-%M-%S')
    gdf.to_file(out_geojson)
    return out_geojson

def batch_contours_to_lines(in_folder, out_folder):
    """
    converts contours to lines

    inputs:
    in_folder (str): path to dir containing contours
    out_folder (str): path to dir to save shorelines to

    outputs:
    out_folder (str): path to dir to save shorelines to
    """
    try:
        os.mkdir(out_folder)
    except:
        pass
    polygons = glob.glob(in_folder + '/*.geojson')
    for polygon in polygons:
        name = os.path.basename(polygon)
        name_no_ext = os.path.splitext(name)[0]
        out_polygon = os.path.join(out_folder, name_no_ext + '_lines.geojson')
        date = name[0:19]
        contours_to_lines(polygon, out_polygon, date)    
    return out_folder

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

def merge_lines(in_folder, out_file):
    """
    Merges shorelines into one geojson

    inputs:
    in_folder (str): dir containing lines
    out_file (str): file to save shorelines to (.geojson)

    outputs:
    out_file (str): file to save shorelines to (.geojson)
    """
    lines = glob.glob(in_folder + '/*.geojson')
    gdfs = [gpd.read_file(line) for line in lines]
    final_gdf = pd.concat(gdfs)
    final_gdf = utm_to_wgs84_df(final_gdf)
    final_gdf.to_file(out_file)
    return out_file

def get_images_for_analysis(folder, good_bad, thresh=0.335):
    """
    gets satellite images suitabile for analysis

    inputs:
    folder (str): folder of ms images
    good_bad (str): path to the image suitability results
    thresh (float): threshold for model

    outputs:
    out_images (list): list of suitable images for analysis
    """
    good_bad_df = pd.read_csv(good_bad)
    good_bad_df['dates'] = [os.path.basename(path)[0:19] for path in good_bad_df['im_paths']]
    images = glob.glob(folder + '/*.tif')
    out_images = []
    for image in images:
        name = os.path.basename(image)
        date = name[0:19]
        filter_df = good_bad_df[good_bad_df['dates']==date]
        score = filter_df['model_scores'].iloc[0]
        if score>=thresh:
            out_images.append(image)
    return out_images



##"""
##Experiment with different thresholding results
##"""
##data_folder = """path/to/CoastSeg/data/site"""
##in_folder_L5 = os.path.join(data_folder, 'L5', 'ms')
##in_folder_L7 = os.path.join(data_folder, 'L7', 'ms')
##in_folder_L8 = os.path.join(data_folder, 'L8', 'ms')
##in_folder_L9 = os.path.join(data_folder, 'L9', 'ms')
##in_folder_S2 = os.path.join(data_folder, 'S2', 'ms')
##good_bad = os.path.join(data_folder, 'good_bad.csv')
##out_folder = """path/to/output"""
##try:
##    os.mkdir(out_folder)
##except:
##    pass
##images_L5 = get_images_for_analysis(in_folder_L5, good_bad, thresh=0.95)
##images_L7 = get_images_for_analysis(in_folder_L7, good_bad, thresh=0.95)
##images_L8 = get_images_for_analysis(in_folder_L8, good_bad, thresh=0.95)
##images_L9 = get_images_for_analysis(in_folder_L9, good_bad, thresh=0.95)
##images_S2 = get_images_for_analysis(in_folder_S2, good_bad, thresh=0.95)
##images = images_L5 + images_L7 + images_L8 + images_L9 + images_S2
##out_folder_otsu = os.path.join(out_folder, 'otsu')r'E:\ThresholdTest\otsu'
##out_folder_otsu_contours = os.path.join(out_folder, 'otsu_contours')
##out_folder_otsu_lines = os.path.join(out_folder, 'otsu_lines')
##merged_lines = os.path.join(out_folder, 'shorelines.geojson')
##batch_compute_otsu_threshold(images, out_folder_otsu)
##batch_binary_raster_to_vector(out_folder_otsu, out_folder_otsu_contours)
##batch_contours_to_lines(out_folder_otsu_contours,out_folder_otsu_lines)
##merge_lines(out_folder_otsu_lines,merged_lines)





