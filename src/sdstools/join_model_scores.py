"""
Joining model scores to extracted shoreline points, transect time series,
tidally corrected transect time series, extracted shoreline lines.
"""

import geopandas as gpd
import pandas as pd
import os

def drop_columns_if_exist(df, columns):
    """
    Drop columns from a DataFrame if they exist.
    Parameters:
    df (pd.DataFrame): DataFrame to drop columns from.
    columns (list): List of column names to drop.

    Returns:
    pd.DataFrame: DataFrame with columns dropped.
    """
    for col in columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    return df


def join_model_scores_to_time_series(transect_time_series_merged_path,
                                     good_bad_csv=None,
                                     good_bad_seg_csv=None):
    """
    Joins model scores to time series transect data based on the provided CSVs.
    Parameters:
    transect_time_series_merged_path (str): Path to transect time series CSV.
    good_bad_csv (str, optional): Path to the image suitability output CSV.
    good_bad_seg_csv (str, optional): Path to the seg filter output CSV.

    Returns:
    str: Path to the updated transect time series CSV with model scores joined.
    """

    # Load transect time series data
    transect_time_series_merged = pd.read_csv(transect_time_series_merged_path)
    transect_time_series_merged['dates'] = pd.to_datetime(transect_time_series_merged['dates'], utc=True)

    if good_bad_csv:

        drop_columns_if_exist(transect_time_series_merged, ['classifier_model_score', 'classifier_threshold'])
        
        # Load and process good_bad CSV
        good_bad = pd.read_csv(good_bad_csv)
        good_bad['dates'] = good_bad['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        
        # Prepare merge and rename directly
        merge_columns = ['dates', 'model_scores']
        if 'threshold' in good_bad.columns:
            merge_columns.append('threshold')
        merged_df = good_bad[merge_columns]    
        
        transect_time_series_merged = transect_time_series_merged.merge(merged_df,
                                                                       on='dates',
                                                           how='left',
                                                                       suffixes=('_ts', '_image'))
        # Optional: drop additional unnamed columns if present
        transect_time_series_merged.drop(columns=[col for col in transect_time_series_merged if 'Unnamed:' in col], errors='ignore', inplace=True)
        # rename model_scores column to classifier_model_score
        transect_time_series_merged.rename(columns={'model_scores': 'classifier_model_score'}, inplace=True)
        if "threshold" in transect_time_series_merged:
            transect_time_series_merged.rename(columns={'threshold': 'classifier_threshold'}, inplace=True)

    if good_bad_seg_csv:

        drop_columns_if_exist(transect_time_series_merged, ['segmentation_model_score', 'segmentation_threshold'])

        # Load and process good_bad_seg CSV
        good_bad_seg = pd.read_csv(good_bad_seg_csv)
        good_bad_seg['dates'] = good_bad_seg['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        
        # Prepare merge and rename directly
        merge_columns = ['dates', 'model_scores']
        if 'threshold' in good_bad_seg.columns:
            merge_columns.append('threshold')
        merged_df = good_bad_seg[merge_columns]       
            
        transect_time_series_merged = transect_time_series_merged.merge(merged_df,
                                                                       on='dates',
                                                                       how='left',
                                                                       suffixes=('', '_seg'))
        # rename model_scores column to segmentation_model_score
        transect_time_series_merged.rename(columns={'model_scores': 'segmentation_model_score'}, inplace=True)
        if "threshold" in transect_time_series_merged:
            transect_time_series_merged.rename(columns={'threshold': 'segmentation_threshold'}, inplace=True)

    # Save updated DataFrame
    transect_time_series_merged.to_csv(transect_time_series_merged_path)

    return transect_time_series_merged_path

def join_model_scores_to_shorelines(shorelines_path,
                                    good_bad_csv=None,
                                    good_bad_seg_csv=None):
    """
    Joins model scores to shoreline points based on the provided CSVs.
    Parameters:
    shorelines_path (str): path to the extracted shorelines geojson.
    good_bad_csv (str, optional): path to the image suitability output CSV.
    good_bad_seg_csv (str, optional): path to the seg filter output CSV.

    Returns:
    str: path to the shoreline points with model scores joined.
    """
    # Load shorelines data
    shorelines_gdf = gpd.read_file(shorelines_path)
    shorelines_gdf['date'] = pd.to_datetime(shorelines_gdf['date'], utc=True)

    if good_bad_csv:
        good_bad_df = pd.read_csv(good_bad_csv)
        good_bad_df['dates'] = good_bad_df['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        
        # Drop old scores if they exist
        drop_columns_if_exist(shorelines_gdf, ['classifier_model_score', 'classifier_threshold'])

        # Prepare merge and rename directly
        merge_columns = ['dates', 'model_scores']
        if 'threshold' in good_bad_df.columns:
            merge_columns.append('threshold')
        merged_df = good_bad_df[merge_columns]
        
        shorelines_gdf = shorelines_gdf.merge(merged_df,
                                              left_on='date', right_on='dates',
                                              suffixes=('', '_image'))
        
        #rename model_scores to classifer_model_score
        shorelines_gdf.rename(columns={'model_scores': 'classifier_model_score'}, inplace=True)
        if "threshold" in shorelines_gdf:
            shorelines_gdf.rename(columns={'threshold': 'classifier_threshold'}, inplace=True)

    if good_bad_seg_csv:
        good_bad_seg_df = pd.read_csv(good_bad_seg_csv)
        good_bad_seg_df['dates'] = good_bad_seg_df['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        
        # Drop old scores if they exist
        drop_columns_if_exist(shorelines_gdf, ['segmentation_model_score', 'segmentation_threshold'])

        # Prepare merge and rename directly
        merge_columns = ['dates', 'model_scores']
        if 'threshold' in good_bad_seg_df.columns:
            merge_columns.append('threshold')
        merged_df = good_bad_seg_df[merge_columns]       
        
        shorelines_gdf = shorelines_gdf.merge(merged_df,
                                              left_on='date', right_on='dates',
                                              suffixes=('', '_seg'))
        # rename model_scores column to segmentation_model_score
        shorelines_gdf.rename(columns={'model_scores': 'segmentation_model_score'}, inplace=True)
        if "threshold" in shorelines_gdf:
            shorelines_gdf.rename(columns={'threshold': 'segmentation_threshold'}, inplace=True)

    # Save modified GeoDataFrame
    # drop any duplicate columns
    shorelines_gdf = shorelines_gdf.loc[:,~shorelines_gdf.columns.duplicated()]

    shorelines_gdf.to_file(shorelines_path)

    return shorelines_path


# # Example #1 : join_model_scores_to_shorelines
# good_bad_csv = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime03-04-24__03_43_01\jpg_files\preprocessed\RGB\image_classification_results.csv"
# good_bad_seg_csv = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\segmentation_classification_results.csv"
# shorelines_path = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\extracted_shorelines_points.geojson"
# join_model_scores_to_shorelines(shorelines_path,
#                                     good_bad_csv,
#                                     good_bad_seg_csv)


# # Example #2 : join_model_scores_to_time_series
# good_bad_csv = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime03-04-24__03_43_01\jpg_files\preprocessed\RGB\image_classification_results.csv"
# good_bad_seg_csv = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\segmentation_classification_results.csv"
# transect_time_series_merged_path = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\raw_transect_time_series_merged.csv"

# csv = pd.read_csv(transect_time_series_merged_path)
# # drop the columns if they exist
# columns = ['model_scores_seg','classifier_model_score','segmentation_model_score']
# for col in columns:
#     if col in csv.columns:
#         csv.drop(columns=col, inplace=True)
# # overwrite the old save
# csv.to_csv(transect_time_series_merged_path, index=False)

# join_model_scores_to_time_series(transect_time_series_merged_path,
#                                     good_bad_csv,
#                                     good_bad_seg_csv)