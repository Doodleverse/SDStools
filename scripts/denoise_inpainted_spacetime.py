

## Takes a CSV file of SDS data (shorelines versus transects)
## written by Dr Daniel Buscombe, May 1, 2024

## Example usage, from cmd:
## python denoise_inpainted_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv"

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
    parser = argparse.ArgumentParser(description="Script to ")

    parser.add_argument(
        "-f",
        "-F",
        dest="csv_file",
        type=str,
        required=True,
        help="Set the name of the CSV file.",
    )


    return parser.parse_args()


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file

    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(cs_file)

    cs_data_matrix_nonans_denoised = filter.filter_wavelet_auto(cs_data_matrix)

    df = pd.DataFrame(cs_data_matrix_nonans_denoised.T,columns=cs_transects_vector)
    df.set_index(cs_dates_vector)
    df.to_csv(csv_file.replace(".csv","_denoised.csv"))


    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(cs_data_matrix)
    plt.axis('off'); plt.title("a) Original", loc='left')
    plt.subplot(122)
    plt.imshow(cs_data_matrix_nonans_denoised)
    plt.axis('off'); plt.title("b) Denoised", loc='left')
    outfile = csv_file.replace(".csv","_nooutliers.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

