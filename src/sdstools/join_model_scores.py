"""
Joining model scores to extracted shoreline points, transect time series,
tidally corrected transect time series, extracted shoreline lines.
"""

import geopandas as gpd
import pandas as pd
import os


def join_model_scores_to_shorelines(good_bad_csv,
                                    good_bad_seg_csv,
                                    shorelines_path,
                                    img_type):
    """
    Joins model scores to shoreline points
    inputs:
    good_bad_csv (str): path to the image suitability output csv
    good_bad_seg_csv (str): path to the seg filter output csv
    shorelines_path (str): path to the extracted_shorelines_points.geojson or extracted_shorelines_lines.geojson
    img_type (str): 'RGB' 'MNDWI' or 'NDWI'.
    outputs:
    shorelines_path (str): path to the shoreline points with model scores joined
    """
    ##load files
    shorelines_gdf = gpd.read_file(shorelines_path)
    shorelines_gdf['date'] = pd.to_datetime(shorelines_gdf['date'], utc=True)

    good_bad_df = pd.read_csv(good_bad_csv)
    good_bad_seg_df = pd.read_csv(good_bad_seg_csv)

    ##get image dates
    dts = [None]*len(good_bad_df)
    for i in range(len(good_bad_df)):
        dt = os.path.basename(good_bad_df['im_paths'].iloc[i])
        idx = dt.find('_RGB')
        dt = dt[0:idx]
        dts[i] = dt
    good_bad_df['dates'] = dts
    good_bad_df['dates'] = pd.to_datetime(good_bad_df['dates'], utc=True,
                                          format='%Y-%m-%d-%H-%M-%S')

    ##get seg dates
    dts_seg = [None]*len(good_bad_seg_df)
    for i in range(len(good_bad_seg_df)):
        dt = os.path.basename(good_bad_seg_df['im_paths'].iloc[i])
        idx = dt.find('_'+img_type)
        dt = dt[0:idx]
        dts_seg[i] = dt
    good_bad_seg_df['dates'] = dts_seg
    good_bad_seg_df['dates'] = pd.to_datetime(good_bad_seg_df['dates'], utc=True,
                                          format='%Y-%m-%d-%H-%M-%S')

    ##merge image scores
    shorelines_gdf = shorelines_gdf.merge(good_bad_df,
                                               left_on='date',
                                               right_on='dates',
                                               suffixes=['', '_image']
                                                        )

    ##merge seg scores
    shorelines_gdf = shorelines_gdf.merge(good_bad_seg_df,
                                               left_on='date',
                                               right_on='dates',
                                               suffixes=['', '_seg']
                                                        )

    ##clean up columns
    cols = list(shorelines_gdf.columns)
    keep_cols = ['date', 'satname', 'geoaccuracy', 'cloud_cover',
                 'geometry','im_paths','model_scores','model_scores_seg']
    for col in cols:
        if col not in keep_cols:
            shorelines_gdf = shorelines_gdf.drop(columns=[col])
    shorelines_gdf.to_file(shorelines_path)

    return shorelines_path

def join_model_scores_to_time_series(good_bad_csv,
                                     good_bad_seg_csv,
                                     transect_time_series_merged_path,
                                     img_type):
    """
    Joins model scores to shoreline points
    inputs:
    good_bad (str): path to the image suitability output csv
    good_bad_seg (str): path to the seg filter output csv
    transect_time_series_merged_path (str): path to raw_transect_time_series_merged.csv
                                        or tidally_corrected_transect_time_series_merged.csv
    img_type (str): 'RGB' 'MNDWI' or 'NDWI'.
    outputs:
    shorelines_path (str): path to the transect_time_series_merged.csv with model scores joined
    """
    ##load csv
    transect_time_series_merged = pd.read_csv(transect_time_series_merged_path)
    transect_time_series_merged['dates'] = pd.to_datetime(transect_time_series_merged['dates'], utc=True)

    ##getting image dates
    good_bad = pd.read_csv(good_bad_csv)
    good_bad_seg = pd.read_csv(good_bad_seg_csv)
    dts = [None]*len(good_bad)
    for i in range(len(good_bad)):
        dt = os.path.basename(good_bad['im_paths'].iloc[i])
        idx = dt.find('_RGB')
        dt = dt[0:idx]
        dts[i] = dt
    good_bad['dates'] = dts
    good_bad['dates'] = pd.to_datetime(good_bad['dates'],
                                       utc=True,
                                       format='%Y-%m-%d-%H-%M-%S')
    try:
        good_bad = good_bad.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'])
        good_bad_seg = good_bad_seg.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'])
    except:
        pass

    ##gettting seg dates
    dts_seg = [None]*len(good_bad_seg)
    for i in range(len(good_bad_seg)):
        dt = os.path.basename(good_bad_seg['im_paths'].iloc[i])
        idx = dt.find('_'+img_type)
        dt = dt[0:idx]
        dts_seg[i] = dt
    good_bad_seg['dates'] = dts_seg
    good_bad_seg['dates'] = pd.to_datetime(good_bad_seg['dates'],
                                           utc=True,
                                           format='%Y-%m-%d-%H-%M-%S')
    

    ##join good_bad and good_bad_seg scores
    transect_time_series_merged = transect_time_series_merged.merge(good_bad,
                                                                    left_on='dates',
                                                                    right_on='dates',
                                                                    suffixes = ['_ts', '_image']
                                                                    )
    transect_time_series_merged = transect_time_series_merged.merge(good_bad_seg,
                                                                    left_on='dates',
                                                                    right_on='dates',
                                                                    suffixes = ['', '_seg']
                                                                    )
    ##get satellite names
    satnames = [None]*len(transect_time_series_merged)
    for i in range(len(transect_time_series_merged)):
        im_path = transect_time_series_merged['im_paths'].iloc[i]
        satname = os.path.splitext(os.path.basename(im_path))[0][-2:]
        satnames[i] = satname
    transect_time_series_merged['satname'] = satnames

    keep_cols = ['dates','x','y','transect_id','cross_distance',
                 'shore_x','shore_y','im_paths','model_scores',
                 'model_scores_seg','satname','tide']

    ##clean up columns
    cols = list(transect_time_series_merged.columns)
    for col in cols:
        if col not in keep_cols:
            transect_time_series_merged = transect_time_series_merged.drop(columns=[col])

    transect_time_series_merged.to_csv(transect_time_series_merged_path)
    
    return transect_time_series_merged_path

