
## Takes a CSV file of SDS data (shorelines versus transects)
## and computes statistics per transect
## written by Dr Daniel Buscombe and Dr Mark Lundine, May 2024

## Example usage, from cmd:
## python analyze_transects.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import os
import scipy
from typing import List, Tuple
import datetime
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the stats script
    Arguments and their defaults are defined within the function.
    Returns:
    - argparse.Namespace: A namespace containing the script's command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Script to compute statistics per transect in an SDS matrix")

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

def adf_test(timeseries):
    """
    Checks for stationarity (lack of trend) in timeseries
    significance value here is going to be 0.05
    If p-value > 0.05 then we are interpeting the tiemseries as stationary
    otherwise it's interpreted as non-stationary
    inputs:
    timeseries:
    outputs:
    stationary_bool: if p-value > 0.05, return True, if p-value < 0.05 return False
    """
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if dfoutput['p-value'] < 0.05:
        stationary_bool = True
    else:
        stationary_bool = False
    return stationary_bool


def get_linear_trend(df):
    """
    LLS on single transect timeseries
    inputs:
    df (pandas DataFrame): two columns, dates and cross-shore positions
    trend_plot_path (str): path to save plot to
    outputs:
    lls_result: all the lls results (slope, intercept, stderr, intercept_stderr, rvalue)
    x: datetimes in years
    """
    
    datetimes = np.array(df.index)
    shore_pos = np.array(df.values)
    datetimes_seconds = [None]*len(datetimes)
    initial_time = datetimes[0]
    for i in range(len(df)):
        t = df.index[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    x = datetimes_years
    y = shore_pos
    lls_result = scipy.stats.linregress(x,y)
    return lls_result.slope, lls_result.intercept,lls_result.rvalue, lls_result.pvalue, lls_result.stderr, lls_result.intercept_stderr #, x


def compute_approximate_entropy(U, m, r):
    """Compute Aproximate entropy, from https://en.wikipedia.org/wiki/Approximate_entropy
    If this value is high, then the timeseries is probably unpredictable.
    If this value is low, then the timeseries is probably predictable.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))


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


def detrend_shoreline_rel_mean(input_matrix):
    "subtract a stable (N-average) initial position from shoreline time-series"
    shore_change = (input_matrix - input_matrix.mean(axis=0)).T
    return shore_change


def compute_time_delta(df, which_timedelta):
    """
    Computes average and max time delta for timeseries rounded to days
    Need to drop the nan rows to compute these
    returns average and maximum timedeltas
    """
    df = df.dropna()
    datetimes = df.index
    timedeltas = [datetimes[i-1]-datetimes[i] for i in range(1, len(datetimes))]

    if which_timedelta == 'minimum':
        return_timedelta = min(abs(np.array(timedeltas)))
    elif which_timedelta == 'average':
        avg_timedelta = sum(timedeltas, datetime.timedelta(0)) / len(timedeltas)
        return_timedelta = abs(avg_timedelta)
    else:
        return_timedelta = max(abs(np.array(timedeltas)))

    return return_timedelta


def resample_timeseries(df, timedelta):
    """
    Resamples the timeseries according to the provided timedelta
    """
    df2 = df.loc[~df.index.duplicated(), :]
    oidx = df2.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq=timedelta)
    new_df = df2.reindex(nidx, method='nearest').interpolate()

    # new_df = df2.resample(timedelta).mean()
    return new_df

def compute_autocorrelation(df):
    """
    This computes and plots the autocorrelation
    Autocorrelation tells you how much a timeseries is correlated with a lagged version of itself.
    Useful for distinguishing timeseries with patterns vs random timeseries
    For example, for a timeseries sampled every 1 month,
    if you find a strong positive correlation (close to +1) at a lag of 12,
    then your timeseries is showing repeating yearly patterns.

    Alternatively, if it's a spatial series, sampled every 50m,
    if you find a strong negative correlation (close to -1) at a lag of 20,
    then you might interpret this as evidence for something like the presence of littoral drift.

    If the autocorrelation is zero for all lags, the series is random--good luck finding any meaning from it!
    """
    
    x = pd.plotting.autocorrelation_plot(df)
    plt.close('all')
    lags = x.lines[-1].get_xdata()
    autocorr = x.lines[-1].get_ydata()

    idx = autocorr.argmax()
    idx2 = autocorr.argmin()
    autocorr_max = np.max(autocorr)
    autocorr_min = np.min(autocorr)
    # lag_max = lags[idx]
    lag_min = lags[idx2]
    
    return autocorr_min, lag_min, autocorr, lags


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file
    doplot = args.doplot

    print(f"Analyzing each transect in file: {csv_file}")

    ##which_timedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is what the timeseries is resampled at
    which_timedelta = 'average'

    ### input files
    cs_file = os.path.normpath(csv_file)
    ### read in data and column/row vectors
    cs_data_matrix, cs_dates_vector, cs_transects_vector = read_merged_transect_time_series_file(cs_file)
    cs_data_matrix_demeaned = detrend_shoreline_rel_mean(cs_data_matrix)

    df_demean = pd.DataFrame(cs_data_matrix_demeaned,columns=cs_transects_vector)
    df_demean = df_demean.set_index(cs_dates_vector)

    ##Step 2: Compute time delta
    new_timedelta = compute_time_delta(df_demean, which_timedelta)

    df = pd.DataFrame(cs_data_matrix.T,columns=cs_transects_vector)
    df = df.set_index(cs_dates_vector)

    ##Step 4: Resample timeseries to the new timedelta
    df_resampled = resample_timeseries(df, new_timedelta)

    ## re-allocate empty lists for outputs
    stationarity = [] 
    autocorr_mins = []
    lag_mins = []
    entropy = []
    linear_trend_slopes = []
    linear_trend_intercepts = []
    linear_trend_rvalues = []
    linear_trend_pvalues = []
    linear_trend_stderr = []
    linear_trend_intercept_stderr = []

    trend_mat = []
    season_mat = []
    autocorr_mat = []
    weights_mat = []
    ## cycle through each transect time-series and make stats
    for k in tqdm(list(df_resampled.columns.values)):
        stationary_bool = adf_test(df_resampled[k])
        stationarity.append(stationary_bool)

        sl, intercept, rvalue, pvalue, stderr, int_stderr = get_linear_trend(df_resampled[k])
        linear_trend_slopes.append(sl)
        linear_trend_intercepts.append(intercept)
        linear_trend_rvalues.append(rvalue)
        linear_trend_pvalues.append(pvalue)
        linear_trend_stderr.append(stderr)
        linear_trend_intercept_stderr.append(int_stderr)

        autocorr_min, lag_min, autocorr, lags = compute_autocorrelation(df_resampled[k])
        approx_entropy = compute_approximate_entropy(df_demean[k].values,2,np.std(df_demean[k].values))

        stl = STL(df_demean[k], period=12, robust=True)
        res_robust = stl.fit()
        trend_mat.append(res_robust.trend)
        season_mat.append(res_robust.seasonal)
        autocorr_mat.append(autocorr)
        weights_mat.append(res_robust.weights)

        autocorr_mins.append(autocorr_min)
        lag_mins.append(lag_min)
        entropy.append(approx_entropy)


    ### create dictionary for output to csv
    out_dict = {}
    out_dict['stationarity'] = np.array(stationarity,dtype='int')
    out_dict['autocorr_min'] = np.array(autocorr_mins,dtype='float')
    out_dict['lag_min'] = np.array(lag_mins,dtype='int')
    out_dict['entropy'] = np.array(entropy,dtype='float')

    out_dict['linear_trend_slopes'] = np.array(linear_trend_slopes,dtype='float')
    out_dict['linear_trend_intercepts'] = np.array(linear_trend_intercepts,dtype='float')
    out_dict['linear_trend_rvalues'] = np.array(linear_trend_rvalues,dtype='float')
    out_dict['linear_trend_pvalues'] = np.array(linear_trend_pvalues,dtype='float')
    out_dict['linear_trend_stderr'] = np.array(linear_trend_stderr,dtype='float')
    out_dict['linear_trend_intercept_stderr'] = np.array(linear_trend_intercept_stderr,dtype='float')

    ## make output dataframe and export
    output_df = pd.DataFrame.from_dict(out_dict).T
    output_df.columns = cs_transects_vector
    output_df = output_df.T
    output_df.to_csv(cs_file.replace(".csv","_stats_per_transect.csv"))

    trend2d = np.dstack(trend_mat).squeeze().T
    season2d = np.dstack(season_mat).squeeze().T
    auto2d = np.dstack(autocorr_mat).squeeze().T
    weights2d = np.dstack(autocorr_mat).squeeze().T

    np.savez(cs_file.replace(".csv","_stats_timeseries.npz"), trend2d=trend2d, season2d=season2d, auto2d=auto2d, weights2d=weights2d, cs_transects_vector=cs_transects_vector, cs_dates_vector=cs_dates_vector, cs_data_matrix_demeaned=cs_data_matrix_demeaned, df_resampled=df_resampled)

    if doplot==1:
        ## make a plot
        plt.figure(figsize=(12,8))
        plt.subplot(131); plt.imshow(trend2d.T,vmin=np.percentile(trend2d,2),vmax=np.percentile(trend2d,98))
        cb=plt.colorbar(); cb.set_label('Trend (m)')
        plt.xlabel('Transect'); plt.ylabel('Time')
        plt.subplot(132); plt.imshow(season2d.T, vmin=np.percentile(season2d,2),vmax=np.percentile(season2d,98))
        cb=plt.colorbar(); cb.set_label('Seasonality (m)')
        plt.subplot(133); plt.imshow(auto2d.T,vmin=0,vmax=1)
        cb=plt.colorbar(); cb.set_label('Autocorrelation (-)')
        plt.savefig(cs_file.replace(".csv","_stats_timeseries.png"), dpi=200, bbox_inches='tight')
        plt.close()
        


if __name__ == "__main__":
    main()

