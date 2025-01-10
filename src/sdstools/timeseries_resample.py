import pandas as pd

def resample_timeseries(df, timedelta):
    """
    Resamples the timeseries according to the provided timedelta
    """
    old_df = df
    old.index = df['dates']
    new_df = old_df.resample(timedelta).mean()
    return new_df

def fill_nans(df):
    """
    Fills nans in timeseries with linear interpolation
    """
    old_df = df
    old.index = df['dates']
    new_df['position'] = old_df['position'].interpolate(method='linear', limit=None, limit_direction='both')
    return new_df

def moving_average(df, window):
    """
    Applying a moving average to a timeseries of specified window length
    inputs:
    df (pandas DataFrame): index is datetimes, only column is 'position'
    window (timedelta str): the window length for the moving average
    returns:
    df (pandas DataFrame): the timeseries with moving average applied
    """
    old_df = df
    old_df.index = df['dates']
    new_df['position'] = old_df['position'].rolling(window).mean()
    return new_df
