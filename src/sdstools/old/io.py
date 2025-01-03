
import pandas as pd 
import numpy as np
from typing import Tuple, List

def read_merged_transect_time_series_file(transect_time_series_file: str) -> Tuple[np.ndarray, pd.Series, List[str]]:
    """
    Read and parse a CoastSeg/CoastSat output file in stacked column wise date and transects format.

    This function reads a CSV file, removes unnamed columns, and transforms the data into a matrix.
    It also extracts a vector of dates and a vector of transects from the data.

    Parameters:
    transect_time_series_file (str): The path to the CSV file to be read.

    Returns:
    Tuple[np.ndarray, pd.Series, List[str]]: A tuple containing the shoreline positions along the transects as a matrix (numpy array), 
    shoreline positions along the transects as a vector (pandas Series), and the transects vector (list of strings).
    """
    merged_transect_time_series = pd.read_csv(transect_time_series_file, index_col=False)
    merged_transect_time_series.reset_index(drop=True, inplace=True)

    # Removing unnamed columns using drop function
    merged_transect_time_series.drop(merged_transect_time_series.columns[merged_transect_time_series.columns.str.contains(
        'unnamed', case=False)], axis=1, inplace=True)
    
    # Extracting the shoreline positions along the transects for each date
    data_matrix = merged_transect_time_series.T.iloc[1:]
    data_matrix = np.array(data_matrix.values).astype('float')

    dates_vector = pd.to_datetime(merged_transect_time_series.dates)
    transects_vector = [t for t in merged_transect_time_series.T.index[1:] if 'date' not in t]

    return data_matrix, dates_vector, transects_vector


def pivot_df_distances_by_time_and_transect(input_matrix):
    "doc string here"
    df_distances_by_time_and_transect = input_matrix.pivot(index='dates',columns='transect_id', values='cross_distance')
    return df_distances_by_time_and_transect

def pivot_df_tides_by_time_and_transect(input_matrix):
    "doc string here"
    df_tides_by_time_and_transect = input_matrix.pivot(index='dates',columns='transect_id', values='tide')
    return df_tides_by_time_and_transect

def pivot_df_x_by_time_and_transect(input_matrix):
    "doc string here"
    df_x_by_time_and_transect = input_matrix.pivot(index='dates',columns='transect_id', values='x')
    return df_x_by_time_and_transect

def pivot_df_y_by_time_and_transect(input_matrix):
    "doc string here"
    df_y_by_time_and_transect = input_matrix.pivot(index='dates',columns='transect_id', values='y')
    return df_y_by_time_and_transect




# def strip_matrix(kmt):
#     "read CoastSat output in pandas datafrane and strip of shoreline data"
#     ind = [k for k in kmt.columns if k.startswith('Unnamed')]
#     for i in ind:
#         kmt = kmt.loc[:,kmt.columns != i ]

#     ind = [k for k in kmt.columns if k.startswith('date')]
#     for i in ind:
#         kmt = kmt.loc[:,kmt.columns != i ]

#     kmt = kmt.to_numpy()
#     return kmt

# def get_dates(kmt):
#     "read CoastSat output in pandas datafrane and strip of dates"
#     kmt = kmt.drop_duplicates()
#     kmt = kmt.loc[:,~kmt.columns.duplicated()]

#     keys = [k for k in kmt.keys() if k.startswith('dates')]
#     keys = [i for n, i in enumerate(keys) if i not in keys[:n]]

#     kmt_dt = [kmt[k] for k in keys]
#     kmt_dt_out = []
#     for kitem in kmt_dt:
#         kitem = [dt.strptime(k, '%Y-%m-%d %H:%M:%S+00:00') for k in kitem]
#         kmt_dt_out.append(kitem)
#     return kmt_dt_out

# def get_transects(kmt):
#     "read CoastSat output in pandas datafrane and strip of transects"
#     kmt = kmt.drop_duplicates()
#     kmt = kmt.loc[:,~kmt.columns.duplicated()]
#     return [k for k in kmt.keys() if k.startswith('kmt')]


# def read_individual_transect_time_series_file(transect_time_series_file):
#     "doc string here"
#     transect_time_series = pd.read_csv(transect_time_series_file)
#     return transect_time_series
