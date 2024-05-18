
## Takes a CSV file of SDS data (shorelines versus transects)
## and subtracts the starting value (or an average of N starting points)
## leaving each transect time-series relative to the starting position
## written by Dr Daniel Buscombe, May, 2024

## Example usage, from cmd:
## python detrend_relstart_transect_timeseries.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv -N 10

import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

def detrend_shoreline_rel_start(input_matrix, N=10, axis_to_average=0):
    "detrend a transect x time SDS matrix based on the first N values"
    axis_to_average = int(axis_to_average)
    if axis_to_average==0:
        vec_start = np.nanmean(input_matrix[:N,:],axis=0)
    else:
        vec_start = np.nanmean(input_matrix[:,:N],axis=1)

    shore_change = (input_matrix - vec_start).T
    return shore_change


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
    # get the transect IDs as a vector
    transects_vector = [t for t in merged_transect_time_series.T.index[1:] if 'date' not in t]

    return data_matrix, dates_vector, transects_vector


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the  Hampel filter to remove outliers in SDS data matrix script.
    Arguments and their defaults are defined within the function.
    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to ")

    parser.add_argument(
        "-f",
        "-F",
        dest="csv_file",
        type=str,
        required=True,
        help="Set the name of the CSV file.",
    )
    parser.add_argument(
        "-n",
        "-N",
        dest="num_start_points",
        type=int,
        required=True,
        help="Set the number of points to use to define the start.",
    )

    parser.add_argument(
        "-p",
        "-P",
        dest="doplot",
        type=int,
        required=False,
        default=0,
        help="1=make a plot, 0=no plot (default).",
    )

    return parser.parse_args()


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file
    num_start_points = args.num_start_points
    doplot = args.doplot
    print(f"File: {csv_file}, Number of starting points to average over: {num_start_points}")

    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = read_merged_transect_time_series_file(cs_file)

    axis_to_average = np.where([i==len(cs_transects_vector) for i in cs_data_matrix.shape])[0]

    ## detrend data
    cs_detrend = detrend_shoreline_rel_start(cs_data_matrix, N=num_start_points, axis_to_average = axis_to_average)

    ## write out new file
    df = pd.DataFrame(cs_detrend,columns=cs_transects_vector)
    df = df.set_index(cs_dates_vector)
    df.to_csv(cs_file.replace(".csv","_detrend.csv"))

    if doplot==1:
        ## make a plot
        outfile = cs_file.replace(".csv","_detrend.png")
        plt.figure(figsize=(12,8))
        plt.subplot(121)
        plt.imshow(cs_data_matrix)
        plt.axis('off'); plt.title("a) Original", loc='left')
        plt.subplot(122)
        plt.imshow(cs_detrend.T)
        plt.axis('off'); plt.title("b) Detrend", loc='left')
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()

