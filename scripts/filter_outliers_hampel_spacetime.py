

## Takes a CSV file of SDS data (shorelines versus transects) and filters outliers
## written by Dr Daniel Buscombe, May, 2024

## Example usage, from cmd:
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv"
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv" -s 3 -i 5 -w 0.05
## python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_zoo.csv"

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Union, List, Tuple


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

class HampelFilter:
    """
    HampelFilter class for providing additional functionality such as checking the upper/lower boundaries for paramter tuning.
    """

    def __init__(self, window_size: int = 5, n_sigma: int = 3, c: float = 1.4826):
        """ Initialize HampelFilter object. Rolling median and rolling sigma are calculated here.

        :param window_size: length of the sliding window, a positive odd integer.
            (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
        :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
        :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
        :return: the outlier indices
        """

        if not (type(window_size) == int and window_size % 2 == 1 and window_size > 2):
            raise ValueError("window_size must be an odd integer greater than 2")
        if not (type(n_sigma) == int and n_sigma >= 0):
            raise ValueError("n_sigma must be a positive integer greater than or equal to 0.")

        self.window_size = window_size
        self.n_sigma = n_sigma
        self.c = c

        # These values will be set after executing apply()
        self._outlier_indices = None
        self._upper_bound = None
        self._lower_bound = None

    def apply(self, x: Union[List, pd.Series, np.ndarray]):
        """ Return the indices of the detected outliers by the filter.

        :param x: timeseries values of type List, numpy.ndarray, or pandas.Series

        :return: indices of the outliers
        """
        # Check given arguments
        if not (type(x) == list or type(x) == np.ndarray or type(x) == pd.Series):
            raise ValueError("x must be either of type List, numpy.ndarray, or pandas.Series.")

        # calculate rolling_median and rolling_sigma using the given parameters.
        x_window_view = sliding_window_view(np.array(x), window_shape=self.window_size)
        rolling_median = np.median(x_window_view, axis=1)
        rolling_sigma = self.c * np.median(np.abs(x_window_view - rolling_median.reshape(-1, 1)), axis=1)

        self._upper_bound = rolling_median + (self.n_sigma * rolling_sigma)
        self._lower_bound = rolling_median - (self.n_sigma * rolling_sigma)

        outlier_indices = np.nonzero(
            np.abs(np.array(x)[(self.window_size - 1) // 2:-(self.window_size - 1) // 2] - rolling_median)
            >= (self.n_sigma * rolling_sigma)
        )[0] + (self.window_size - 1) // 2

        if type(x) == list:
            # When x is of List[float | int], return the indices in List.
            self._outlier_indices = list(outlier_indices)
        elif type(x) == pd.Series:
            # When x is of pd.Series, return the indices of the Series object.
            self._outlier_indices = x.index[outlier_indices]
        else:
            self._outlier_indices = outlier_indices

        return self

    def get_indices(self) -> Union[List, pd.Series, np.ndarray]:
        """
        """
        if self._outlier_indices is None:
            raise AttributeError("Outlier indices have not been set. Execute hampel_filter_object.apply(x) first.")
        return self._outlier_indices

    def get_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the upper and lower boundaries of the filter. Note that the values are `window_size - 1` shorter than the given timeseries x.

        :return: a tuple of the lower bound values and the upper bound values. i.e. (lower_bound_values, upper_bound_values)
        """
        if self._upper_bound is None or self._lower_bound is None:
            raise AttributeError("Boundary values have not been set. Execute hampel_filter_object.apply() first.")

        return self._lower_bound, self._upper_bound


def hampel_filter(x: Union[List, pd.Series, np.ndarray], window_size: int = 5, n_sigma: int = 3, c: float = 1.4826) \
        -> Union[List, pd.Series, np.ndarray]:
    """ Outlier detection using the Hampel identifier

    :param x: timeseries values of type List, numpy.ndarray, or pandas.Series
    :param window_size: length of the sliding window, a positive odd integer.
        (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
    :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
    :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
    :return: the outlier indices
    """
    return HampelFilter(window_size=window_size, n_sigma=n_sigma, c=c).apply(x).get_indices()


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
        type=float,
        required=False,
        default=0.05,
        help="Set the windowPerc parameter.",
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

def implement_filter(cs_data_matrix, windowPerc, NoSTDsRemoved, iteration):

    orig = cs_data_matrix.copy()

    cs_data_matrix_outliers_removed = cs_data_matrix.copy()
    for k in range(orig.shape[0]):
        SDS_timeseries = orig[k,:]
        if len(SDS_timeseries)<2:
            continue
        else:
            window_size=int(windowPerc * len(SDS_timeseries))

        if window_size<=2:
            window_size = 3

        if (window_size % 2) == 0: 
            window_size = window_size+1

        # Remove outliers using the Hampel filter
        outliers = hampel_filter(SDS_timeseries, window_size=window_size, n_sigma=NoSTDsRemoved) 
        
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
    doplot = args.doplot

    print(f"Window as a percent of data length: {windowPerc}")
    print(f"Number of iterations: {iterations}")
    print(f"Number of stdev from mean: {NoSTDsRemoved}")

    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = read_merged_transect_time_series_file(cs_file)

    cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix, windowPerc, NoSTDsRemoved, iteration=1)
    if iterations>2:
        for k in range(iterations):
            cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix_outliers_removed, windowPerc, NoSTDsRemoved, iteration=k)
    elif iterations==2:
        cs_data_matrix_outliers_removed = implement_filter(cs_data_matrix_outliers_removed, windowPerc, NoSTDsRemoved, iteration=2)

    df = pd.DataFrame(cs_data_matrix_outliers_removed.T,columns=cs_transects_vector)
    df = df.set_index(cs_dates_vector)
    df.to_csv(csv_file.replace(".csv","_nooutliers.csv"))
    print(f"Output written to {os.path.abspath(csv_file.replace('.csv','_nooutliers.csv'))}")


    if doplot==1:
        plt.figure(figsize=(12,8))
        plt.subplot(121)
        plt.imshow(cs_data_matrix)
        plt.title("a) Original", loc='left') #plt.axis('off'); 
        plt.xlabel('Time'); plt.ylabel('Transect')
        plt.subplot(122)
        plt.imshow(cs_data_matrix_outliers_removed)
        plt.title("b) Outliers removed", loc='left') #plt.axis('off'); 
        plt.xlabel('Time'); plt.ylabel('Transect')
        outfile = csv_file.replace(".csv","_nooutliers.png")
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Figure save saved to  {os.path.abspath(outfile)}")
        plt.close()


if __name__ == "__main__":
    main()

