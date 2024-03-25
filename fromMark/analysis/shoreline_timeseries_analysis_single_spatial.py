"""
Mark Lundine
This is an in-progress script with tools for processing 1D shoreline timeseries data.

Updates needed!!
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import random
from scipy import stats
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import csv


def adf_test(series):
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
    dftest = adfuller(series, autolag="AIC")
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
    LLS on single longshore transect
    ##todo
    inputs:
    df (pandas DataFrame): two columns, dates and cross-shore positions
    trend_plot_path (str): path to save plot to
    outputs:
    lls_result: all the lls results (slope, intercept, stderr, intercept_stderr, rvalue)
    x: datetimes in years
    """
    
    longshore = np.array(df.index)
    shore_pos = np.array(df['position'])
    x = longshore
    y = shore_pos
    lls_result = stats.linregress(x,y)
    return lls_result, x

def de_trend_timeseries(df, lls_result, x):
    """
    de-trends the shoreline timeseries
    """
    slope = lls_result.slope
    intercept = lls_result.intercept
    y = df['position']
    
    fitx = np.linspace(min(x),max(x),len(x))
    fity = slope*fitx + intercept

    detrend = y - fity
    new_df = pd.DataFrame({'position':detrend},
                          index=df.index)
    trend_df = pd.DataFrame({'position':fity},
                            index=df.index)
    return new_df, trend_df

def de_mean_timeseries(df):
    """
    de-means the shoreline timeseries
    """
    mean_pos = np.mean(df['position'])
    new_pos = df['position']-mean_pos
    new_df = pd.DataFrame({'position':new_pos.values},
                          index=df.index)
    return new_df
    
def get_shoreline_data(csv_path, transect_spacing):
    """
    Reads and reformats the timeseries into a pandas dataframe with cross-shore position and longshore position
    """
    df = pd.read_csv(csv_path)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    transect_ids = df['transect_id']
    longshore_position = np.arange(0, len(transect_ids)*transect_spacing, transect_spacing)

    new_df = pd.DataFrame({'position':df['position'].values},
                          index=longshore_position)
    return new_df

def get_shoreline_data_df(df, transect_spacing):
    """
    Reads and reformats the timeseries into a pandas dataframe with datetime index
    """
    df = df.replace(r'^\s*$', np.nan, regex=True)
    transect_ids = df['transect_id']
    longshore_position = np.arange(0, len(transect_ids)*transect_spacing, transect_spacing)

    new_df = pd.DataFrame({'position':df['position'].values},
                          index=longshore_position)
    return new_df

def compute_time_delta(df, which_spacedelta):
    """
    Computes average and max time delta for timeseries rounded to days
    Need to drop the nan rows to compute these
    returns average and maximum timedeltas
    """
    df = df.dropna()
    longshore_position = df.index
    spacedeltas = [longshore_position[i-1]-longshore_position[i] for i in range(1, len(longshore_position))]
    avg_spacedelta = sum(spacedeltas, spacedeltas[0])/len(spacedeltas)
    avg_spacedelta = abs(avg_spacedelta)
    min_spacedelta = min(abs(np.array(spacedeltas)))
    max_spacedelta = max(abs(np.array(spacedeltas)))

    if which_spacedelta == 'minimum':
        return_spacedelta = min_spacedelta
    elif which_spacedelta == 'average':
        return_spacedelta = avg_spacedelta
    else:
        return_spacedelta = max_spacedelta
    return return_spacedelta

def resample_timeseries(df, spacedelta):
    """
    todo
    Resamples the timeseries according to the provided spacedelta
    """
    new_df = df[df.index%spacedelta==0]
    return new_df

def fill_nans(df):
    """
    Fills nans in timeseries with linear interpolation, keep it simple student
    """
    new_df = df.interpolate(method='linear', limit=None, limit_direction='both')
    return new_df

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

    autocorr = np.abs(autocorr)
    idx = autocorr.argmax()
    autocorr_max = np.max(autocorr)
    lag_max = lags[idx]
    
    # plotting the Curve
    x.plot()

    # Display
    plt.savefig(fig_save, dpi=300)
    plt.close()
    return autocorr_max, lag_max

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
    
def make_plots(output_folder,
               name,
               df,
               df_resampled,
               df_no_nans,
               df_de_meaned,
               df_de_trend_bool=False,
               df_de_trend=None,
               df_trend=None):
    """
    Making longshore plots of data, vertically stacked
    """
    fig_save = os.path.join(output_folder, name+'timeseries.png')
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 1
    if df_de_trend_bool == False:
        plt.rcParams["figure.figsize"] = (16,12)
        ##Raw 
        plt.subplot(4,1,1)
        plt.suptitle(name)
        plt.plot(df.index, df['position'], '--o', color='k', label='Raw')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df['position']), np.nanmax(df['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        ##Resampled
        plt.subplot(4,1,2)
        plt.plot(df_resampled.index, df_resampled['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df_resampled.index), max(df_resampled.index))
        plt.ylim(np.nanmin(df_resampled['position']), np.nanmax(df_resampled['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        ##Interpolated
        plt.subplot(4,1,3)
        plt.plot(df_no_nans.index, df_no_nans['position'], '--o', color='k', label='Interpolated')
        plt.xlim(min(df_no_nans.index), max(df_no_nans.index))
        plt.ylim(np.nanmin(df_no_nans['position']), np.nanmax(df_no_nans['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        ##De-meaned
        plt.subplot(4,1,4)
        plt.plot(df_de_meaned.index, df_de_meaned['position'], '--o', color='k', label='De-Meaned')
        plt.xlim(min(df_de_meaned.index), max(df_de_meaned.index))
        plt.ylim(np.nanmin(df_de_meaned['position']), np.nanmax(df_de_meaned['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlabel('Longhsore Distance (m)')
        plt.minorticks_on()
        plt.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(fig_save, dpi=300)
        plt.close()
    else:
        plt.rcParams["figure.figsize"] = (16,16)
        ##Raw 
        plt.subplot(5,1,1)
        plt.suptitle(name)
        plt.plot(df.index, df['position'], '--o', color='k', label='Raw')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df['position']), np.nanmax(df['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.legend()
        ##Resampled
        plt.subplot(5,1,2)
        plt.plot(df_resampled.index, df_resampled['position'], '--o', color='k', label='Resampled')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df_resampled['position']), np.nanmax(df_resampled['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        ##Interpolated
        plt.subplot(5,1,3)
        plt.plot(df_no_nans.index, df_no_nans['position'], '--o', color='k', label='Interpolated')
        plt.plot(df_trend.index, df_trend['position'], '--', color='blue', label='Linear Trend')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df_no_nans['position']), np.nanmax(df_no_nans['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xticks([],[])
        plt.minorticks_on()
        plt.legend()
        ##De-trended
        plt.subplot(5,1,4)
        plt.plot(df_de_trend.index, df_de_trend['position'], '--o', color='k', label='De-Trended')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df_de_trend['position']), np.nanmax(df_de_trend['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.minorticks_on()
        plt.legend()
        ##De-meaned
        plt.subplot(5,1,5)
        plt.plot(df_de_meaned.index, df_de_meaned['position'], '--o', color='k', label='De-Meaned')
        plt.xlim(min(df.index), max(df.index))
        plt.ylim(np.nanmin(df_de_meaned['position']), np.nanmax(df_de_meaned['position']))
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlabel('Longshore Distance (m)')
        plt.xticks([],[])
        plt.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(fig_save, dpi=300)
        plt.close()
        
def main_df(df,
            output_folder,
            name,
            transect_spacing,
            which_spacedelta,
            spacedelta=None):
    """
    Spatial analysis for satellite shoreline data
    inputs:
    csv_path (str): path to the shoreline timeseries csv
    should have columns 'transect_id' and 'position'
    where transect_id contains the transect id, transects should be evenly spaced!!
    position is the cross-shore position of the shoreline (in m)
    output_folder (str): path to save outputs to
    name (str): a site name
    transect_spacing (int): transect spacing in meters
    which_spacedelta (str): 'minimum' 'average' or 'maximum' or 'custom this is the new longshore spacing to sample at
    spacedelta (int, optional): if custom specify new spacedelta, do not put finer spacing than input!!
    outputs:
    spatial_series_analysis_result (dict): results of this cookbook
    """
    ##Step 1: Load in data
    df = get_shoreline_data_df(df, transect_spacing)
    
    ##Step 2: Compute average and max time delta
    if which_spacedelta != 'custom':
        new_spacedelta = compute_space_delta(df, which_spacedelta)
    else:
        new_spacedelta=spacedelta


    ##Step 3: Resample timeseries to the maximum timedelta
    df_resampled = resample_timeseries(df, new_spacedelta)

    ##Step 4: Fill NaNs
    df_no_nans = fill_nans(df_resampled)
    snr_no_nans = np.abs(np.mean(df_no_nans['position']))/np.std(df_no_nans['position'])

    ##Step 5: Check for stationarity with ADF test
    stationary_bool = adf_test(df_no_nans['position'])

    ##Step 6a: If timeseries stationary, de-mean, compute autocorrelation and approximate entropy
    ##Then make plots
    if stationary_bool == True:
        df_de_meaned = de_mean_timeseries(df_no_nans)
        autocorr_max, lag_max = plot_autocorrelation(output_folder,
                                                     name,
                                                     df_de_meaned)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'].values,
                                                          2,
                                                          np.std(df_de_meaned['position']))
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend_bool=False)
        slope = np.nan
        intercept = np.nan
        stderr = np.nan
        intercept_stderr = np.nan
        r_sq = np.nan

        
    ##Step 6b: If timeseries non-stationary, compute trend, de-trend, de-mean, compute autocorrelation and approximate entropy
    ##Then make plots
    else:
        trend_result, x = get_linear_trend(df_no_nans)
        df_de_trend, df_trend = de_trend_timeseries(df_no_nans, trend_result, x)
        ##Step 5: De-mean the timeseries
        df_de_meaned = de_mean_timeseries(df_de_trend)
        autocorr_max, lag_max = plot_autocorrelation(output_folder,
                                                     name,
                                                     df_de_meaned)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'].values,
                                                          2,
                                                          np.std(df_de_meaned['position']))
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend_bool=True,
                   df_de_trend=df_de_trend,
                   df_trend=df_trend)
        slope = trend_result.slope
        intercept = trend_result.intercept
        stderr = trend_result.stderr
        intercept_stderr = trend_result.intercept_stderr
        r_sq = trend_result.rvalue**2

    ##Put results into dictionary
    spatial_series_analysis_result = {'stationary_bool':stationary_bool,
                                      'computed_trend':slope,
                                      'computed_intercept':intercept,
                                      'trend_unc':stderr,
                                      'intercept_unc':intercept_stderr,
                                      'r_sq':r_sq,
                                      'autocorr_max':autocorr_max,
                                      'lag_max':str(lag_max*new_spacedelta),
                                      'new_spacedelta':str(new_spacedelta),
                                      'snr_no_nans':snr_no_nans,
                                      'approx_entropy':approximate_entropy}

    result = os.path.join(output_folder, name+'ssa_result.csv')
    with open(result,'w') as f:
        w = csv.writer(f)
        w.writerow(spatial_series_analysis_result.keys())
        w.writerow(spatial_series_analysis_result.values())
        
    output_df = pd.DataFrame({'longshore_position':df_no_nans.index,
                              'position':df_no_nans['position']})
    output_path = os.path.join(output_folder, name+'_resampled.csv')
    output_df.to_csv(output_path)  
    return spatial_series_analysis_result, output_df, new_spacedelta

def main(csv_path,
         output_folder,
         name,
         transect_spacing,
         which_spacedelta,
         spacedelta=None):
    """
    Spatial analysis for satellite shoreline data
    inputs:
    csv_path (str): path to the shoreline timeseries csv
    should have columns 'transect_id' and 'position'
    where transect_id contains the transect id, transects should be evenly spaced!!
    position is the cross-shore position of the shoreline (in m)
    output_folder (str): path to save outputs to
    name (str): a site name
    transect_spacing (int): transect spacing in meters
    which_spacedelta (str): 'minimum' 'average' or 'maximum' or 'custom this is the new longshore spacing to sample at
    spacedelta (int, optional): if custom specify new spacedelta, do not put finer spacing than input!!
    outputs:
    spatial_series_analysis_result (dict): results of this cookbook
    """
    ##Step 1: Load in data
    df = pd.read_csv(csv_path)
    df = get_shoreline_data(csv_path, transect_spacing)
    
    ##Step 2: Compute average and max time delta
    if which_spacedelta != 'custom':
        new_spacedelta = compute_space_delta(df, which_spacedelta)
    else:
        new_spacedelta=spacedelta


    ##Step 3: Resample timeseries to the maximum timedelta
    df_resampled = resample_timeseries(df, new_spacedelta)

    ##Step 4: Fill NaNs
    df_no_nans = fill_nans(df_resampled)
    snr_no_nans = np.abs(np.mean(df_no_nans['position']))/np.std(df_no_nans['position'])

    ##Step 5: Check for stationarity with ADF test
    stationary_bool = adf_test(df_no_nans['position'])
    
    ##Step 6a: If timeseries stationary, de-mean, compute autocorrelation and approximate entropy
    ##Then make plots
    if stationary_bool == True:
        df_de_meaned = de_mean_timeseries(df_no_nans)
        autocorr_max, lag_max = plot_autocorrelation(output_folder,
                                                     name,
                                                     df_de_meaned)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'],
                                                          2,
                                                          np.std(df_de_meaned['position']))
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend_bool=False)
        slope = np.nan
        intercept = np.nan
        stderr = np.nan
        intercept_stderr = np.nan
        r_sq = np.nan

        
    ##Step 6b: If timeseries non-stationary, compute trend, de-trend, de-mean, compute autocorrelation and approximate entropy
    ##Then make plots
    else:
        trend_result, x = get_linear_trend(df_no_nans)
        df_de_trend, df_trend = de_trend_timeseries(df_no_nans, trend_result, x)
        ##Step 5: De-mean the timeseries
        df_de_meaned = de_mean_timeseries(df_de_trend)
        autocorr_max, lag_max = plot_autocorrelation(output_folder,
                                                     name,
                                                     df_de_meaned)
        approximate_entropy = compute_approximate_entropy(df_de_meaned['position'].values,
                                                          2,
                                                          np.std(df_de_meaned['position']))
        make_plots(output_folder,
                   name,
                   df,
                   df_resampled,
                   df_no_nans,
                   df_de_meaned,
                   df_de_trend_bool=True,
                   df_de_trend=df_de_trend,
                   df_trend=df_trend)
        slope = trend_result.slope
        intercept = trend_result.intercept
        stderr = trend_result.stderr
        intercept_stderr = trend_result.intercept_stderr
        r_sq = trend_result.rvalue**2

    ##Put results into dictionary
    spatial_series_analysis_result = {'stationary_bool':stationary_bool,
                                      'computed_trend':slope,
                                      'computed_intercept':intercept,
                                      'trend_unc':stderr,
                                      'intercept_unc':intercept_stderr,
                                      'r_sq':r_sq,
                                      'autocorr_max':autocorr_max,
                                      'lag_max':str(lag_max*new_spacedelta),
                                      'new_spacedelta':str(new_spacedelta),
                                      'snr_no_nans':snr_no_nans,
                                      'approx_entropy':approximate_entropy}

    result = os.path.join(output_folder, name+'ssa_result.csv')
    with open(result,'w') as f:
        w = csv.writer(f)
        w.writerow(spatial_series_analysis_result.keys())
        w.writerow(spatial_series_analysis_result.values())
        
    output_df = pd.DataFrame({'longshore_position':df_no_nans.index,
                              'position':df_no_nans['position']})
    output_path = os.path.join(output_folder, name+'_resampled.csv')
    output_df.to_csv(output_path)      
    return spatial_series_analysis_result




    

