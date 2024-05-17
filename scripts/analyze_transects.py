
## Takes a CSV file of SDS data (shorelines versus transects)
## and computes statistics per transect
## written by Dr Daniel Buscombe, May, 2024

## Example usage, from cmd:
## python analyze_transects.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import scipy
from typing import List, Tuple
import datetime
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import STL

def fit_sine(t, y, lag, timedelta):
    """
    Fitting a sine wave to data
    adapted from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    inputs:
    t (array of datetimes)
    y (array of shoreline poisition)
    lag (int): number of lags
    timedelta (datetime.TimeDelta): time spacing of trace
    output_folder (path): path to output figure to
    outputs:
    result_dict: sine wave fit params
    """
    # fig_save_path = os.path.join(output_folder, 'sin_fit.png')
    tt = np.arange(0, len(t), 1)
    yy = np.array(y)
    guess_period = lag*2
    guess_freq = 1./guess_period
    guess_amp = (np.max(yy)-np.min(yy))/2
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0.])

    def sinfunc(t, A, w, p):  return A * np.sin(w*t + p)
    
    popt, pcov = scipy.optimize.curve_fit(sinfunc,
                                          tt,
                                          yy,
                                          p0=guess,
                                          maxfev=5000,
                                          bounds = ((0, 2*np.pi*guess_freq/2, 0),
                                                    (np.max(yy), 2*np.pi*guess_freq*2, 2*np.pi)
                                                    )
                                          )
    A, w, p = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p)
    period = 1./f
    period = period*timedelta
    rmse = np.sqrt((np.square(fitfunc(tt) - yy)).mean(axis=0))
    error_max = max(np.abs(fitfunc(tt)-yy))
    result_dict = {"amp": A,
                   "phase": p,
                   "period": period,
                   "rmse":rmse,
                   "error_max":error_max}

    return result_dict

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
    shore_pos = np.array(df['position'])
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
    return lls_result, x



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

    return parser.parse_args()


def detrend_shoreline_rel_mean(input_matrix):
    "doc string here"
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
    lags = x.lines[-1].get_xdata()
    autocorr = x.lines[-1].get_ydata()

    idx = autocorr.argmax()
    idx2 = autocorr.argmin()
    autocorr_max = np.max(autocorr)
    autocorr_min = np.min(autocorr)
    lag_max = lags[idx]
    lag_min = lags[idx2]
    
    return autocorr_max, lag_max, autocorr_min, lag_min, autocorr, lags


##==========================================
def main():
    args = parse_arguments()
    csv_file = args.csv_file
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


    stationarity = [] 
    autocorr_maxs = []
    lag_maxs = []
    autocorr_mins = []
    lag_mins = []

    sine_periods = []
    sine_amplitudes = []
    sine_phases = []
    sine_rmse = []
    sine_errormax = []

    for k in list(df_resampled.columns.values):
        stationary_bool = adf_test(df_resampled[k])
        stationarity.append(stationary_bool)

        autocorr_max, lag_max, autocorr_min, lag_min, autocorr, lags = compute_autocorrelation(df_resampled[k])

        # res = STL(df_resampled[k].values).fit()

        # res = statsmodels.tsa.seasonal.seasonal_decompose(df_resampled[k].values, period = 14)

        autocorr_maxs.append(autocorr_max)
        lag_maxs.append(lag_max)
        autocorr_mins.append(autocorr_min)
        lag_mins.append(lag_min)

        try:
            sin_result = fit_sine(df_demean.index,
                                    df_demean[k].values,
                                    lag_min,
                                    pd.Timedelta(new_timedelta))
            sine_periods.append(sin_result.period)
            sine_amplitudes.append(sin_result.amp)
            sine_phases.append(sin_result.phase)
            sine_rmse.append(sin_result.rmse)
            sine_errormax.append(sin_result.error_max)
        except:
            sine_periods.append(np.nan)
            sine_amplitudes.append(np.nan)
            sine_phases.append(np.nan)
            sine_rmse.append(np.nan)
            sine_errormax.append(np.nan)

    out_dict = {}
    out_dict['stationarity'] = np.array(stationary_bool,dtype='int')








            
    #     approximate_entropy = compute_approximate_entropy(df_de_meaned['position'],
    #                                                         2,
    #                                                         np.std(df_de_meaned['position']))

    #     slope = np.nan
    #     intercept = np.nan
    #     stderr = np.nan
    #     intercept_stderr = np.nan
    #     r_sq = np.nan

    # else:

    #     trend_result, x = get_linear_trend(df_med_filt)
    #     df_de_trend, df_trend = de_trend_timeseries(df_med_filt, trend_result, x)
        
    #     ##Step 5: De-mean the timeseries
    #     df_de_meaned = de_mean_timeseries(df_de_trend)
    #     autocorr_max, lag_max, autocorr_min, lag_min, autocorr, lags = plot_autocorrelation(output_folder,
    #                                                                                         name,
    #                                                                                         df_de_meaned)
    #     try:
    #         sin_result = fit_sine(df_de_meaned.index,
    #                               df_de_meaned['position'],
    #                               lag_min,
    #                               pd.Timedelta(new_timedelta),
    #                               output_folder)
    #     except:
    #         sin_result = {'period':np.nan,
    #                       'amp':np.nan,
    #                       'phase':np.nan,
    #                       'rmse':np.nan,
    #                       'error_max':np.nan
    #                       }
    #     approximate_entropy = compute_approximate_entropy(df_de_meaned['position'],
    #                                                       2,
    #                                                       np.std(df_de_meaned['position']))


    #     slope = trend_result.slope
    #     intercept = trend_result.intercept
    #     stderr = trend_result.stderr
    #     intercept_stderr = trend_result.intercept_stderr
    #     r_sq = trend_result.rvalue**2
        

    # ##Put results into dictionary
    # timeseries_analysis_result = {'stationary_bool':stationary_bool,
    #                               'computed_trend':slope,
    #                               'computed_intercept':intercept,
    #                               'trend_unc':stderr,
    #                               'intercept_unc':intercept_stderr,
    #                               'r_sq':r_sq,
    #                               'autocorr_max':autocorr_max,
    #                               'lag_max':str(lag_max*new_timedelta),
    #                               'autocorr_min':autocorr_min,
    #                               'lag_min':str(lag_min*new_timedelta),
    #                               'new_timedelta':str(new_timedelta),
    #                               'snr_no_nans':snr_no_nans,
    #                               'snr_median_filter':snr_median_filter,
    #                               'approx_entropy':approximate_entropy,
    #                               'period':sin_result['period'],
    #                               'amplitude':sin_result['amp'],
    #                               'phase':sin_result['phase'],
    #                               'sin_rmse':sin_result['rmse'],
    #                               'sin_error_max':sin_result['error_max']}


    # output_df = pd.DataFrame(timeseries_analysis_result)
    # output_path = os.path.join(output_folder, name+'_resampled.csv')
    # output_df.to_csv(output_path)


if __name__ == "__main__":
    main()

