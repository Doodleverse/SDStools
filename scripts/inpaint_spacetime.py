

## Takes a CSV file of SDS data (shorelines versus transects)
## written by Dr Daniel Buscombe, May 1, 2024

## Example usage, from cmd:
## python inpaint_spacetime.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/elwha_mainROI_df_distances_by_time_and_transect_CoastSat_nooutliers.csv

import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
from skimage.restoration import inpaint

def inpaint_spacetime_matrix(input_matrix):
    mask = np.isnan(input_matrix)
    return inpaint.inpaint_biharmonic(input_matrix, mask)

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
    Parse command-line arguments for inpainting SDS data matrix.
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
    doplot = args.doplot
    
    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = read_merged_transect_time_series_file(cs_file)

    cs_data_matrix_nooutliers_nonans = inpaint_spacetime_matrix(cs_data_matrix)

    df = pd.DataFrame(cs_data_matrix_nooutliers_nonans.T,columns=cs_transects_vector)
    df = df.set_index(cs_dates_vector)
    df.to_csv(csv_file.replace(".csv","_inpainted.csv"))
    print(f"Saved inpainted data to {csv_file.replace('.csv','_inpainted.csv')}")

    if doplot==1:
        plt.figure(figsize=(12,8))
        plt.subplot(121)
        plt.imshow(cs_data_matrix)
        plt.axis('off'); plt.title("a) Original", loc='left')
        plt.subplot(122)
        plt.imshow(cs_data_matrix_nooutliers_nonans)
        plt.axis('off'); plt.title("b) Inpainted", loc='left')
        outfile = csv_file.replace(".csv","_inpainted.png")
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()

