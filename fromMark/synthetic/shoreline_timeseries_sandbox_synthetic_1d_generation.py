"""
Mark Lundine
This is an experimental script to investigate how much information can be pulled
from satellite shoreline timeseries.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import random
import pandas as pd
import os

def time_array_to_years(datetimes):
    """
    Converts array of datetimes to years since earliest datetime
    inputs:
    datetimes (array): array of datetimes
    returns:
    datetimes_years (array): the time array but in years from the earliest datetime
    """
    initial_time = datetimes[0]
    datetimes_seconds = [None]*len(datetimes)
    for i in range(len(datetimes)):
        t = datetimes[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    return datetimes_years

def linear_trend(t, trend_val):
    """
    Applies linear trend to trace
    y = mx+b
    inputs:
    t (array or float): array of time (in years) or just a single float of time in years
    trend_val (float): trend value in m/year
    returns:
    y (float or array): the results of this linear trend function (in m)
    """
    y = trend_val*t
    return y

def sine_pattern(t, A, period_years):
    """
    Applies seasonal pattern of random magnitude to trace
    y = A*sin((2pi/L)*x)
    inputs:
    t (array or float): array of time (in years) or just a single float of time in years
    A (float): amplitude of sine wave (in m)
    period_years (float): period of sine wave in years
    returns:
    y (array or float): result of this sine function (in m)
    """
    y = A*np.sin(((2*np.pi*t)/period_years))
    return y

def enso_pattern(t, amplitudes, periods):
    """
    Enso-esque pattern mean of a bunch of sine waves with periods between 3 and 7 years
    Lenght of amplitudes should be equal to length of periods
    inputs:
    t (float): single time in years or array of times in years
    amplitudes (array): array of amplitudes (in m), ex: np.random.uniform(low,high,size=100)
    periods (array): array of periods in years, ex: np.arange(3, 7, 100)
    outputs:
    mean_y (float): mean of the distribution of sine waves (in m)
    """
    y = [None]*len(amplitudes)
    for i in range(len(y)):
        y[i] = amplitudes[i]*np.sin(2*np.pi*t/periods[i])
    y = np.array(y)
    mean_y = np.mean(y)
    return mean_y

def noise(y, noise_val):
    """
    Applies random noise to trace
    inputs:
    y (float): shoreline_position at a specific time (in m)
    noise_val (float): amount of noise to add (in m)
    outputs:
    y (float): shoreline position with noise added (in m)
    """
    ##Let's make the noise between -20 and 20 m to sort of simulate the uncertainty of satellite shorelines
    noise = np.random.normal(-1*noise_val,noise_val,1)
    y = y+noise
    return y

def apply_NANs(y,
               nan_idxes):
    """
    Randomly throws NANs into a shoreline trace
    inputs:
    y (array): the shoreline positions over time 
    nan_idxes (array): the indices to thrown nans in
    outputs:
    y (array): the shoreline positions with nans thrown in at specified indices
    """
    y[nan_idxes] = np.nan
    return y
    
def make_matrix(dt):
    """
    Just hardcoding a start and end datetime with a timestep of dt days
    Start = Jan 1st 1984
    End = Jan 1st 2024
    Make a matrix to hold cross-shore position values
    inputs:
    dt (int): time spacing in days
    outputs:
    shoreline_matrix (array): an array of zeroes with length of the datetimes
    datetimes (array): the datetimes
    """
    datetimes = np.arange(datetime.datetime(1984,1,1),
                          datetime.datetime(2024,1,1),
                          datetime.timedelta(days=int(dt))
                          ).astype(datetime.datetime)
    num_transects = len(datetimes)
    shoreline_matrix = np.zeros((len(datetimes)))
    return shoreline_matrix, datetimes


def make_data(noise_val,
              trend_val,
              yearly_amplitude,
              dt,
              t_gap_frac,
              save_name):
    """
    Makes synthetic shoreline data, will save a csv and png plot of data
    Csv will have columns 'date' and 'position',
    corresponding to the timestamp and cross-shore position, respectively
    y(t) = trend(t) + yearly_pattern(t) + noise(t) + nan(t)
    inputs:
    noise_val (float): amount of noise to add to each position in m
    trend_val (float): linear trend value in m/year
    yearly_amplitude (float): amplitude for sine wave with period of 1 year in m
    dt (int): time spacing in days
    t_gap_frac (float): fraction of timesteps to throw NaNs in
    save_name (str): name to give this
    outputs:
    save_name (str): the save name
    """
    ##Initialize stuff
    #random.seed(0) #uncomment if you want to keep the randomness the same and play with hard-coded values
    matrix, datetimes = make_matrix(dt)
    num_timesteps = matrix.shape[0]
    t = time_array_to_years(datetimes)

    ##randomly selecting a percent of the time periods to throw gaps in
    max_nans = int(t_gap_frac*len(t))
    num_nans = random.randint(0, max_nans)
    nan_idxes = random.sample(range(len(t)), num_nans)

    ##Building matrix
    for i in range(num_timesteps):
         ##Linear trend + yearly cycle + noise
        matrix[i] = sum([linear_trend(t[i], trend_val),
                         sine_pattern(t[i], yearly_amplitude, 1),
                         noise(matrix[i], noise_val)
                         ]
                        )

    matrix = apply_NANs(matrix, nan_idxes)

    df = pd.DataFrame({'date':datetimes,
                       'position':matrix})
    df.to_csv(save_name+'.csv', index=False)

    plt.rcParams["figure.figsize"] = (16,4)
    plt.title(r'Trend value = ' +str(np.round(trend_val,2)) +
              r'm/year    Yearly Amplitude = ' +
              str(np.round(yearly_amplitude,2)) + r'm    dt = ' +
              str(dt) + r' days    Missing timestamps = ' +
              str(t_gap_frac*100)+'%'+
              '    Noise amount = ' + str(noise_val) + 'm')
    plt.plot(df['date'], df['position'], '--o', color='k', markersize=1, linewidth=1)
    plt.xlim(min(df['date']), max(df['date']))
    plt.ylim(min(df['position']), max(df['position']))
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(save_name+'.png', dpi=300)
    return save_name


##Example call, setting the random seed
random.seed(0)
##Noise value in meters
noise_val = 20
##Trend value in m/year
trend_val = random.uniform(-3, 3)
##Amplitude for yearly pattern in m
yearly_amplitude = random.uniform(0, 5)
##Revisit time in days
dt = 12
##Fraction of missing time periods
t_gap_frac = 0.15
##give it a save_name
save_name = 'test1'

make_data(noise_val,
          trend_val,
          yearly_amplitude,
          dt,
          t_gap_frac,
          save_name)


