

## Takes a CSV file of SDS data (shorelines versus transects)
## written by Dr Daniel Buscombe, May 1, 2024

## Example usage, from cmd:
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv"
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv" -s 3

import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from src.SDStools.filter import hampel_filter_matlab
from src.SDStools import io 
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the  Hampel filter to remove outliers in SDS data matrix script.
    Arguments and their defaults are defined within the function.
    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to apply Hampel filter to remove outliers in SDS data matrix (columns=transects, rows=shoreline positions)")

    parser.add_argument(
        "-f",
        "-F",
        dest="csv_file",
        type=str,
        required=True,
        help="Set the name of the CSV file.",
    )

    parser.add_argument(
        "-S",
        "-s",
        dest="NoSTDsRemoved",
        type=int,
        required=False,
        default=3,
        help="Set the NoSTDsRemoved parameter.",
    )

    parser.add_argument(
        "-S",
        "-s",
        dest="iterations",
        type=int,
        required=False,
        default=5,
        help="Set the iterations parameter.",
    )

    parser.add_argument(
        "-S",
        "-s",
        dest="windowPerc",
        type=int,
        required=False,
        default=5,
        help="Set the windowPerc parameter.",
    )

    return parser.parse_args()


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file
    windowPerc = args.windowPerc
    iterations = args.iterations
    NoSTDsRemoved = args.NoSTDsRemoved

    # NoSTDsRemoved = 3, iterations   = 5, windowPerc   = .05
    # csv_file = '/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv'

    ### input files
    cs_file = os.path.normpath(os.getcwd()+csv_file)
    cs_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(cs_file)


    cs_data_matrix_outliers_removed = cs_data_matrix.copy()
    for k in range(cs_data_matrix.shape[0]):
        SDS_timeseries = cs_data_matrix[k,:]
        outliers = hampel_filter_matlab(SDS_timeseries, NoSTDsRemoved = NoSTDsRemoved, iterations   = iterations, windowPerc   = windowPerc)
        cs_data_matrix_outliers_removed[k,outliers] = np.nan 

    df = pd.DataFrame(cs_data_matrix_outliers_removed.T,columns=cs_transects_vector)
    df.set_index(cs_dates_vector)
    df.to_csv(csv_file.replace(".csv","_nooutliers.csv"))

    # plt.subplot(121)
    # plt.imshow(cs_data_matrix)
    # plt.subplot(122)
    # plt.imshow(cs_data_matrix_outliers_removed)
    # plt.show()


if __name__ == "__main__":
    main()

