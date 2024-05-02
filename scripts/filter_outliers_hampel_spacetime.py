

## Takes a CSV file of SDS data (shorelines versus transects)
## written by Dr Daniel Buscombe, May 1, 2024

## Example usage, from cmd:
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv"
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv" -s 3 -i 5 -w
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_zoo.csv"

import argparse, os
import matplotlib.pyplot as plt
import numpy as np
from sdstools import filter
from sdstools import io 
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
        default=2,
        help="Set the NoSTDsRemoved parameter.",
    )

    parser.add_argument(
        "-I",
        "-i",
        dest="iterations",
        type=int,
        required=False,
        default=3,
        help="Set the iterations parameter.",
    )

    parser.add_argument(
        "-W",
        "-w",
        dest="windowPerc",
        type=int,
        required=False,
        default=0.05,
        help="Set the windowPerc parameter.",
    )

    return parser.parse_args()

def implement_filter(cs_data_matrix, windowPerc, NoSTDsRemoved, iteration):

    orig = cs_data_matrix.copy()

    cs_data_matrix_outliers_removed = cs_data_matrix.copy()
    for k in range(orig.shape[0]):
        SDS_timeseries = orig[k,:]
        try:
            outliers = filter.hampel_filter(SDS_timeseries, window_size=int(windowPerc * len(SDS_timeseries)), n_sigma=NoSTDsRemoved) 
        except:
            outliers = filter.hampel_filter(SDS_timeseries, window_size=1+int(windowPerc * len(SDS_timeseries)), n_sigma=NoSTDsRemoved) 

        cs_data_matrix_outliers_removed[k,outliers] = np.nan 


    num_outliers_removed = np.sum(np.isnan(cs_data_matrix_outliers_removed)) -  np.sum(np.isnan(cs_data_matrix))
    print(f"Iteration: {iteration}")
    print(f"Outliers removed: {num_outliers_removed}")
    print(f"Outliers removed percent: {100*(num_outliers_removed/np.prod(np.shape(cs_data_matrix_outliers_removed)))}")

    return cs_data_matrix_outliers_removed


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file
    windowPerc = args.windowPerc
    iterations = args.iterations
    NoSTDsRemoved = args.NoSTDsRemoved

    print(f"Window as a percent of data length: {windowPerc}")
    print(f"Number of iterations: {iterations}")
    print(f"Number of stdev from mean: {NoSTDsRemoved}")

    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(cs_file)

    cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix, windowPerc, NoSTDsRemoved, iteration=1)
    if iterations>2:
        for k in range(iterations):
            cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix_outliers_removed, windowPerc, NoSTDsRemoved, iteration=k)
    elif iterations==2:
        cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix_outliers_removed, windowPerc, NoSTDsRemoved, iteration=2)

    df = pd.DataFrame(cs_data_matrix_outliers_removed.T,columns=cs_transects_vector)
    df.set_index(cs_dates_vector)
    df.to_csv(csv_file.replace(".csv","_nooutliers.csv"))


    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(cs_data_matrix)
    plt.axis('off'); plt.title("a) Original", loc='left')
    plt.subplot(122)
    plt.imshow(cs_data_matrix_outliers_removed)
    plt.axis('off'); plt.title("b) Outliers removed", loc='left')
    outfile = csv_file.replace(".csv","_nooutliers.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

