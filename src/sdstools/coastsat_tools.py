"""
Mark Lundine
This file can be used to download CoastSat data from
http://coastsat.wrl.unsw.edu.au/time-series/.

It will download the timeseries data, make plots, and then make a geojson showing the trends.
"""
import os
import geopandas as gpd
import glob
import urllib.request
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import shapely
from math import degrees, atan2, radians
from scipy import stats

def download_coastsat_data(transects_layer_path,
                           output_folder_path):
    """
    Downloads shoreline timeseries data from CoastSat web database
    You need to already have the trasnects layer downloaded
    inputs:
    transects_layer_path (str): path to the transects layer geojson
    output_folder_path (str): path to save timeseries csvs to
    outputs:
    output_folder_path (str): path to where the timeseries csvs were saved to
    """
    transects = gpd.read_file(transects_layer_path)
    url = r'http://coastsat.wrl.unsw.edu.au/time-series/'

    ##Checking if any were downloaded
    downloaded_csvs = glob.glob(output_folder_path+'/*.csv')
    already_downloaded = [None]*len(downloaded_csvs)
    for i in range(len(already_downloaded)):
        file_name = os.path.splitext(os.path.basename(downloaded_csvs[i]))[0]
        already_downloaded[i] = file_name

    ##Download all files that haven't been downloaded yet
    for i in range(len(transects)):
        transect_id = transects['TransectId'].iloc[i]
        if transect_id in already_downloaded:
            continue
        else:
            download_path = os.path.join(url, transect_id+r'/')
            print(transect_id)
            save_path = os.path.join(folder_save, transect_id+'.csv')
            try:
                urllib.request.urlretrieve(download_path, save_path)
            except:
                continue
            
    return output_folder_path

def gb(x1, y1, x2, y2):
    """
    Get's bearing of two points in utm coords
    """
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return angle

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

def make_coastsat_trends(transects_layer_path,
                         timeseries_data_path):
    """
    Makes the trend map file with CoastSat data
    inputs:
    transects_layer_path (str): path to the coastsat transects_layer geojson
    timeseries_data_path (str): path to where the timeseries csvs are saved to
    outputs:
    transects_trends_path (str): path to the new geojson
    """
    transects = gpd.read_file(transects_layer_path)
    transects_trends_path = os.path.splitext(transects_layer_path)[0]+'_trend_map.geojson'
    folder_save = timeseries_data_path
    downloaded_csvs = glob.glob(folder_save+'/*.csv')
    already_downloaded = [None]*len(downloaded_csvs)
    ##get all downloaded csvs
    for i in range(len(already_downloaded)):
        file_name = os.path.splitext(os.path.basename(downloaded_csvs[i]))[0]
        already_downloaded[i] = file_name

    org_crs = transects.crs
    slopes = transects['Trend'].astype(float)
    max_slope = np.max(np.abs(slopes))
    scaled_slopes = (np.array(slopes)/1)*100
    csv_paths = [None]*len(transects)
    png_paths = [None]*len(transects)
    new_lines = [None]*len(transects)

    ##loop over all transects
    ##adjust transect line
    ##make timeseries plot
    ##add timeseries plot path and timeseries csv path to geojson
    for i in range(len(transects)):
        print(str(i/len(transects)*100))
        transect_id = transects['TransectId'].iloc[i]
        transects_filter = transects[transects['TransectId']==transect_id].reset_index()
        utm_crs = transects_filter.estimate_utm_crs()
        transects_filter_utm = transects_filter.to_crs(utm_crs).iloc[0]
        first = transects_filter_utm.geometry.coords[0]
        last = transects_filter_utm.geometry.coords[1]
        midpoint = transects_filter_utm.geometry.centroid
        distance = scaled_slopes[i]
        if np.isnan(distance)==False:   
            if distance<0:
                angle = radians(gb(first[0], first[1], last[0], last[1])+180)
            else:
                angle = radians(gb(first[0], first[1], last[0], last[1]))
            northing = midpoint.y + abs(distance)*np.sin(angle)
            easting = midpoint.x + abs(distance)*np.cos(angle)
            line_arr = [(midpoint.x,midpoint.y),(easting,northing)]
            line = arr_to_LineString(line_arr)
            dummy_geodf = gpd.GeoDataFrame(pd.DataFrame({'id':[0]}),
                                           geometry=[line],
                                           crs=utm_crs).to_crs(org_crs)
            line = dummy_geodf.iloc[0].geometry
            new_lines[i] = line
        
        if transect_id in already_downloaded:
            save_path = os.path.join(folder_save, transect_id+'.csv')
            fig_save = os.path.splitext(save_path)[0]+'.png'
            csv_paths[i] = save_path
            png_paths[i] = fig_save
            if os.path.isfile(fig_save):
                continue
            else:
                df = pd.read_csv(save_path, header=None)
                x = pd.to_datetime(df.iloc[:,0], format='%Y-%m-%d %H:%M:%S')
                y = df.iloc[:,1]
                ##Plot timeseries
                plt.rcParams["figure.figsize"] = (12,4)
                plt.plot(x, y, c='k', markersize=1)
                plt.xlim(min(x), max(x))
                plt.ylim(np.nanmin(y), np.nanmax(y))
                plt.xlabel('Time (UTC)')
                plt.ylabel('Cross-Shore Position (m)')
                plt.tight_layout()
                plt.savefig(fig_save, dpi=300)
                plt.close()
        else:
            continue


    ##editing transects geodf    
    transects['timeseries_csvs'] = csv_paths
    transects['timeseries_pngs'] = png_paths
    transects['TransectId'] = transect_trends['TransectId'].astype(str)
    transects['SiteId'] = transect_trends['SiteId'].astype(str)
    transects['Orientation'] = transect_trends['Orientation'].astype(float)
    transects['Slope'] = transect_trends['Slope'].astype(float)
    transects['Trend'] = transect_trends['Trend'].astype(float)
    transects.geometry = new_lines
    transects.to_file(transects_trends_path)

    return transects_trends_path
