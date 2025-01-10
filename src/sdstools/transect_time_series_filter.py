import pandas as pd
import numpy as np
import HampelFilter.hampel_filter as hampel_filter



def hampel_filter_df(df, hampel_window=10, hampel_sigma=2):
    """
    Applies a Hampel Filter
    """
  
    vals = df['cross_distance'].values
    outlier_idxes = hampel_filter(vals, hampel_window, hampel_sigma)
    vals[outlier_idxes] = np.nan
    df['cross_distance'] = vals
    return df

def hampel_filter_loop(df, hampel_window=5, hampel_sigma=3):
    df['date'] = df.index
    num_nans = df['cross_distance'].isna().sum()
    new_num_nans = None
    h=0
    while (num_nans != new_num_nans) and (len(df)>hampel_window):
        num_nans = df['cross_distance'].isna().sum()
        df = hampel_filter_df(df, hampel_window=hampel_window, hampel_sigma=hampel_sigma)
        new_num_nans = df['cross_distance'].isna().sum()
        df = df.dropna()
        h=h+1
    print('hampel iterations: '+str(h))
    return df

def change_filter(df, q=0.75):
    """
    Applies a filter on shoreline change
    """
    vals = df['cross_distance'].values
    time = df['dates'] - df['dates'].iloc[0]
    time = time.dt.days
    change_y = np.abs(np.diff(vals))
    change_t = np.diff(time)
    dy_dt = change_y/change_t
    dy_dt = np.concat([[0],dy_dt])
    max_val = np.nanquantile(dy_dt,q)
    outlier_idxes = dy_dt>max_val
    vals[outlier_idxes] = np.nan
    df['cross_distance'] = vals
    return df

def change_filter_loop(df, iterations=1, q=0.75):
    h=0
    for i in range(iterations):
        df = change_filter(df, q=q)
        df = df.dropna()
        h=h+1
    return df
