
"""
Mark Lundine
This is a script with tools for
processing 1D shoreline timeseries data in the time domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats#, signal
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os


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
    lls_result = stats.linregress(x,y)
    return lls_result, x


def plot_autocorrelation(output_folder,
                         name,
                         df):
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
    fig_save = os.path.join(output_folder, name+'autocorrelation.png')
    # Creating Autocorrelation plot
    
    x = pd.plotting.autocorrelation_plot(df['position'])
    lags = x.lines[-1].get_xdata()
    autocorr = x.lines[-1].get_ydata()

    idx = autocorr.argmax()
    idx2 = autocorr.argmin()
    autocorr_max = np.max(autocorr)
    autocorr_min = np.min(autocorr)
    lag_max = lags[idx]
    lag_min = lags[idx2]
    
    # plotting the Curve
    x.plot()

    # Display
    plt.savefig(fig_save, dpi=300)
    plt.close()
    return autocorr_max, lag_max, autocorr_min, lag_min, autocorr, lags


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
