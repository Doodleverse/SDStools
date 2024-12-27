"""
Making a video of temporally colored shorelines with an evolving timeseries

Mark Lundine, USGS
"""


import os
import glob
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cv2
import shutil
import pandas as pd
from pathlib import Path
from scipy import stats


def make_shoreline_video_frames(shorelines_path,
                                config_gdf_path,
                                transect_timeseries_path,
                                transect_id,
                                sitename):
    """
    This function will construct video frames displaying historical shorelines and a select timeseries
    inputs:
    shorelines_path (str): path to the extracted_shorelines_lines.geojson file
    config_gdf_path (str): path to the config_gdf.geojson file
    transect_timeseries_path (str): path to the transect_timeseries.csv or transect_timeseries_tidally_corrected_matrix.csv
    transect_id (str): which transect to show
    sitename (str): name of site
    """
    ##Load shorelines, transect, timeseries in
    shorelines = gpd.read_file(shorelines_path)
    shorelines.rename({'date':'dates'},axis=1,inplace=True)
    shorelines['dates'] = pd.to_datetime(shorelines['dates'], format='%Y-%m-%dT%H:%M:%S')
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf[config_gdf['type']=='transect']
    transect = transects[transects['id']==transect_id]
    timeseries = pd.read_csv(transect_timeseries_path)
    timeseries['dates'] = pd.to_datetime(timeseries['dates'], format='%Y-%m-%d %H:%M:%S+00:00')

    ##Some simple timeseries processing
    dates = timeseries['dates']
    select_timeseries = np.array(timeseries[transect_id])
    data = pd.DataFrame({'pos':select_timeseries},
                             index=dates)
    ##do this so the timeseries dates match the shoreline dates
    data = data.loc[shorelines['dates']]
    ##90D running mean, forward-filling NANs 
    y = data['pos'].rolling('90D', min_periods=1).mean().fillna(method='ffill')
    ##de-meaning the timeseries
    meany = np.mean(y)
    y = y-meany
    ##de-trending the timeseries
    filter_df = pd.DataFrame({'pos':y},
                             index=data.index)
    datetimes_seconds = [None]*len(filter_df)
    initial_time = filter_df.index[0]
    for i in range(len(filter_df)):
        t = filter_df.index[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    X = datetimes_years
    Y = y
    lls_result = stats.linregress(X,Y)
    slope = lls_result.slope
    intercept = lls_result.intercept
    fitx = np.linspace(min(X),max(X),len(X))
    fity = slope*fitx + intercept
    detrended = Y - fity
    detrended[0] = Y[0]
    filter_df['detrended'] = detrended
    n = len(shorelines)
    filter_df['number'] = range(n)
    ##Make temp frame folders
    frame_folder = os.path.join(os.path.dirname(shorelines_path), 'frames')
    try:
        os.mkdir(frame_folder)
    except:
        pass

    ##Loop and plot each shoreline
    plt.rcParams["figure.figsize"] = (10,10)
    color = iter(plt.cm.viridis(np.linspace(0, 1, n)))
    fig, ax = plt.subplots(2,1)
    for i in range(n):
        ts_filter = filter_df[filter_df['number']<=i]
        ts = shorelines['dates'].iloc[i]
        c = next(color)
        filter_shoreline = shorelines[shorelines['dates']==ts]
        filter_shoreline.plot(color=c, ax=ax[0])
        transect.plot(color='k', ax=ax[0])
        ax[0].set_xticks([],[])
        ax[0].set_yticks([],[])
        plt.title(ts)
        plt.subplot(2,1,2)
        ax[1].plot(ts_filter.index, ts_filter['pos'],'--', c='k', label='Raw')
        ax[1].plot(ts_filter.index, ts_filter['detrended'], '--', c='b', label='Detrended')
        ax[1].set_xlim(min(filter_df.index), max(filter_df.index))
        ax[1].set_ylim(min(filter_df['detrended']), max(filter_df['pos']))
        #ax[1].legend()
        plt.minorticks_on()
        plt.ylabel('Cross-Shore Distance (m)')
        plt.xlabel('Time (UTC)')
        plt.tight_layout()
        plt.savefig(os.path.join(frame_folder,sitename+'_'+str(i)+'.png'), dpi=400)
            
    return frame_folder


def makeVideo(frame_folder, vid_path, frame_rate):
    """
    This takes the frames output from make_shoreline_video_frames() and puts them into a video
    It also deletes the frames
    inputs:
    frame_folder (str): path to the frames
    vid_path (str): path to save the video to
    frame_rate (int): frame rate for the video
    """
    path_in = frame_folder
    path_out = vid_path
    fps = frame_rate
    files = sorted(Path(frame_folder).iterdir(), key=os.path.getmtime)
    im = cv2.imread(os.path.join(frame_folder, files[0]))
    height,width,layers = im.shape
    size = (width,height)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(files)):
        print(i/len(files))
        filename=os.path.join(path_in, files[i])
        #reading each frame
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        out.write(img)
        img=None
    out.release()
    ##delete original frame folder
    #shutil.rmtree(frame_folder)
    

def make_shoreline_video(shorelines,
                         config_gdf,
                         transect_timeseries,
                         transect_id,
                         sitename,
                         frame_rate):
    """
    Makes a video of the shorelines from CoastSeg
    inputs:
    shorelines (str): path to the extracted_shorelines_lines.geojson
    config_gdf (str): path to the config_gdf.geojson
    transect_timeseries (str): path to the transect_timeseries.csv or transect_timeseries_tidally_corrected_matrix.csv
    transect_id (str): specifiy which transect to display
    sitename (str): provide a name for the site
    frame_rate (int): frame rate for the output video
    """
    frame_folder = make_shoreline_video_frames(shorelines,
                                               config_gdf,
                                               transect_timeseries,
                                               transect_id,
                                               sitename)
    vid_path = os.path.join(os.path.dirname(shorelines), os.path.splitext(os.path.basename(shorelines))[0]+'_video'+'.mp4')
    makeVideo(frame_folder,
              vid_path,
              frame_rate)

####Sample call
##make_shoreline_video(r'extracted_shorelines_lines.geojson',
##                     r'config_gdf.geojson',
##                     r'transect_time_series.csv',
##                     r'region_8_1557',
##                     'test',
##                     10)
