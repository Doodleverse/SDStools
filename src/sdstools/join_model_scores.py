
"""
Joining model scores to extracted shoreline points, transect time series,
tidally corrected transect time series, extracted shoreline lines.
"""

import geopandas as gpd
import pandas as pd
import os


def join_model_scores_to_time_series(transect_time_series_merged_path,
                                     img_type,
                                     good_bad_csv=None,
                                     good_bad_seg_csv=None):
    """
    Joins model scores to time series transect data based on the provided CSVs.
    Parameters:
    transect_time_series_merged_path (str): Path to transect time series CSV.
    img_type (str): Image type, 'RGB', 'MNDWI', or 'NDWI'.
    good_bad_csv (str, optional): Path to the image suitability output CSV.
    good_bad_seg_csv (str, optional): Path to the seg filter output CSV.

    Returns:
    str: Path to the updated transect time series CSV with model scores joined.
    """

    # Load transect time series data
    transect_time_series_merged = pd.read_csv(transect_time_series_merged_path)
    transect_time_series_merged['dates'] = pd.to_datetime(transect_time_series_merged['dates'], utc=True)

    if good_bad_csv:

        # if the 'classifier_model_score' already exists, then drop it
        if 'classifier_model_score' in transect_time_series_merged.columns:
            transect_time_series_merged.drop(columns='classifier_model_score', inplace=True)
        
        # Load and process good_bad CSV
        good_bad = pd.read_csv(good_bad_csv)
        good_bad['dates'] = good_bad['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        transect_time_series_merged = transect_time_series_merged.merge(good_bad[['dates', 'model_scores']],
                                                                       on='dates',
                                                                       how='left',
                                                                       suffixes=('_ts', '_image'))
        # Optional: drop additional unnamed columns if present
        transect_time_series_merged.drop(columns=[col for col in transect_time_series_merged if 'Unnamed:' in col], errors='ignore', inplace=True)
        # rename model_scores column to classifier_model_score
        transect_time_series_merged.rename(columns={'model_scores': 'classifier_model_score'}, inplace=True)

    if good_bad_seg_csv:

        # if the 'segmentation_model_score' already exists, then drop it
        if 'segmentation_model_score' in transect_time_series_merged.columns:
            transect_time_series_merged.drop(columns='segmentation_model_score', inplace=True)

        # Load and process good_bad_seg CSV
        good_bad_seg = pd.read_csv(good_bad_seg_csv)
        good_bad_seg['dates'] = good_bad_seg['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        transect_time_series_merged = transect_time_series_merged.merge(good_bad_seg[['dates', 'model_scores']],
                                                                       on='dates',
                                                                       how='left',
                                                                       suffixes=('', '_seg'))
        # rename model_scores column to segmentation_model_score
        transect_time_series_merged.rename(columns={'model_scores': 'segmentation_model_score'}, inplace=True)

    # Save updated DataFrame
    transect_time_series_merged.to_csv(transect_time_series_merged_path)

    return transect_time_series_merged_path




def join_model_scores_to_shorelines(shorelines_path,
                                    img_type,
                                    good_bad_csv=None,
                                    good_bad_seg_csv=None):
    """
    Joins model scores to shoreline points based on the provided CSVs.
    Parameters:
    shorelines_path (str): path to the extracted shorelines geojson.
    img_type (str): 'RGB', 'MNDWI', or 'NDWI'.
    good_bad_csv (str, optional): path to the image suitability output CSV.
    good_bad_seg_csv (str, optional): path to the seg filter output CSV.

    Returns:
    str: path to the shoreline points with model scores joined.
    """
    # Load shorelines data
    print(shorelines_path)
    shorelines_gdf = gpd.read_file(shorelines_path)
    shorelines_gdf['date'] = pd.to_datetime(shorelines_gdf['date'], utc=True)

    if good_bad_csv:
        good_bad_df = pd.read_csv(good_bad_csv)
        good_bad_df['dates'] = good_bad_df['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        shorelines_gdf = shorelines_gdf.merge(good_bad_df[['dates', 'model_scores']],
                                              left_on='date', right_on='dates',
                                              suffixes=('', '_image'))
        #rename model_scores to classifer_model_score
        shorelines_gdf.rename(columns={'model_scores': 'classifier_model_score'}, inplace=True)

    if good_bad_seg_csv:
        good_bad_seg_df = pd.read_csv(good_bad_seg_csv)
        good_bad_seg_df['dates'] = good_bad_seg_df['im_paths'].apply(lambda x: pd.to_datetime(os.path.basename(x).split('_')[0], utc=True, format='%Y-%m-%d-%H-%M-%S'))
        shorelines_gdf = shorelines_gdf.merge(good_bad_seg_df[['dates', 'model_scores']],
                                              left_on='date', right_on='dates',
                                              suffixes=('', '_seg'))
        # rename model_scores column to segmentation_model_score
        shorelines_gdf.rename(columns={'model_scores': 'segmentation_model_score'}, inplace=True)

    # Save modified GeoDataFrame
    shorelines_gdf.to_file(shorelines_path)

    return shorelines_path


# Example #1 : join_model_scores_to_shorelines
# good_bad_csv = ""
# good_bad_seg_csv = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\classification_results.csv"
# shorelines_path = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\extracted_shorelines_points.geojson"
# img_type = 'RGB'
# join_model_scores_to_shorelines(shorelines_path,
#                                     img_type,
#                                     good_bad_csv,
#                                     good_bad_seg_csv)


# Example #2 : join_model_scores_to_time_series
# good_bad_csv = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime03-04-24__03_43_01\jpg_files\preprocessed\RGB\classification_results.csv"
# good_bad_seg_csv = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\classification_results.csv"
# transect_time_series_merged_path = r"C:\development\doodleverse\coastseg\CoastSeg\sessions\sample_session_demo1\raw_transect_time_series_merged.csv"

# csv = pd.read_csv(transect_time_series_merged_path)
# # drop the columns if they exist
# columns = ['model_scores_seg','classifier_model_score','segmentation_model_score']
# for col in columns:
#     if col in csv.columns:
#         csv.drop(columns=col, inplace=True)
# # overwrite the old save
# csv.to_csv(transect_time_series_merged_path, index=False)

# img_type = 'RGB'
# join_model_scores_to_time_series(transect_time_series_merged_path,
#                                     img_type,
#                                     good_bad_csv,
#                                     good_bad_seg_csv)