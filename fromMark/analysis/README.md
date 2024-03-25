# SDStols/fromMark/analysis

This will contain scripts for analysis tools for CoastSeg shoreline data.

# shoreline_timeseries_analysis_single.py

Libraries required (Python 3.7, numpy, matplotlib, datetime, random, scipy, pandas, statsmodels, os, csv)

	main(csv_path,
             output_folder,
             name,
             which_timedelta):
		"""
		Timeseries analysis for satellite shoreline data
		Will save timeseries plot (raw, resampled, de-trended, de-meaned) and autocorrelation plot.
		Will also output analysis results to a csv (result.csv)
		inputs:
		csv_path (str): path to the shoreline timeseries csv
		should have columns 'date' and 'position'
		where date contains UTC datetimes in the format YYYY-mm-dd HH:MM:SS
		position is the cross-shore position of the shoreline
		output_folder (str): path to save outputs to
		name (str): name to give this analysis run
		which_timedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is what the timeseries is resampled at
		timedelta (str, optional): the custom time spacing (e.g., '30D' is 30 days)
		beware of choosing minimum, with a mix of satellites, the minimum time spacing can be so 
		low that you run into fourier transform problems
		outputs:
		timeseries_analysis_result (dict): results of this cookbook
		"""
		
1. Resample timeseries to minimum, average, or maximum time delta (temporal spacing of timeseries). My gut is to go with the maximum so we aren't creating data. If we go with minimum or average then linearly interpolate the values to get rid of NaNs.

2. Check if timeseries is stationary with ADF test. We'll use a p-value of 0.05. If we get a p-value greater than this then we are interpreting
the timeseries as non-stationary (there is a temporal trend). 

3. 
	a) If the timeseries is stationary then de-mean it, compute and plot autocorrelation, compute approximate entropy (measure of how predictable 		the timeseries is, values towards 0 indicate predictability, values towards 1 indicate random).

	b) If the timeseries is non-stationary then compute the trend with linear least squares,
	and then de-trend the timeseries. Then de-mean, do autocorrelation, and approximate entropy.

4. This will return a dictionary with the following keys:

	* 'stationary_bool': True or False, whether or not the input timeseries was stationary according to the ADF test.
	* 'computed_trend': a computed linear trend via linear least squares, m/year
	* 'computed_intercept': the computed intercept via linear least squares, m
	* 'trend_unc': the standard error of the linear trend estimate, m/year
	* 'intercept_unc': the standard of error of the intercept estimate, m
	* 'r_sq': the r^2 value from the linear trend estimation, unitless
	* 'autocorr_max': the maximum value from the autocorrelation estimation, this code computes the maximum of the absolute value of the 			autocorrelation
	* 'lag_max': the lag that corresponds to the maximum of the autocorrelation, something of note here: if you are computing autocorrelation on
	a signal with a period of 1 year, then here the lag_max will be half a year. Autocorrelation in this case should be -1 at a half-year lag and 		+1 at a year lag. Since I do the max calculation on the absolute value of the autocorrelation, you get lag_max at the maximum negative 			correlation.
	* 'new_timedelta': this is the new time-spacing for the resampled timeseries
	* 'snr_no_nans': a crude estimate of signal-to-noise ratio, here I just did the mean of the timeseries divided by the standard deviation
	* 'approx_entropy': entropy estimate, values closer to 0 indicate predicatibility, values closer to 1 indicate disorder

main_df will take as input a pandas dataframe instead of a path to a csv. 
It will output the result dictionary, the resampled pandas dataframe, and the new timedelta.

# shoreline_timeseries_analysis_single_spatial.py

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
		
1. Resample spatial series to minimum, average, or maximum spatial delta (longshore spacing of spatial series). My gut is to go with the maximum so we aren't creating data. If we go with minimum or average then linearly interpolate the values to get rid of NaNs.

2. Check if spatial series is stationary with ADF test. We'll use a p-value of 0.05. If we get a p-value greater than this then we are interpreting
the timeseries as non-stationary (there is a temporal trend). 

3. 
	a) If the spatial series is stationary then de-mean it, compute and plot autocorrelation, compute approximate entropy (measure of how 	predictable the timeseries is, values towards 0 indicate predictability, values towards 1 indicate random).

	b) If the spatial series is non-stationary then compute the trend with linear least squares,
	and then de-trend the spatial series. Then de-mean, do autocorrelation, and approximate entropy.

4. This will return a dictionary with the following keys:

	* 'stationary_bool': True or False, whether or not the input timeseries was stationary according to the ADF test.
	* 'computed_trend': a computed linear trend via linear least squares, m/m
	* 'computed_intercept': the computed intercept via linear least squares, m
	* 'trend_unc': the standard error of the linear trend estimate, m/m
	* 'intercept_unc': the standard of error of the intercept estimate, m
	* 'r_sq': the r^2 value from the linear trend estimation, unitless
	* 'autocorr_max': the maximum value from the autocorrelation estimation, this code computes the maximum of the absolute value of the 			autocorrelation
	* 'lag_max': the lag that corresponds to the maximum of the autocorrelation, something of note here: if you are computing autocorrelation on
	a signal with a period of 1000m, then here the lag_max will be 500m. Autocorrelation in this case should be -1 at a 500m lag and 			+1 at a 1000m lag. Since I do the max calculation on the absolute value of the autocorrelation, you get lag_max at the maximum negative 		correlation.
	* 'new_spacedelta': this is the new longshore-spacing for the resampled timeseries (in m)
	* 'snr_no_nans': a crude estimate of signal-to-noise ratio, here I just did the mean of the spatial series divided by the standard deviation
	* 'approx_entropy': entropy estimate, values closer to 0 indicate predicatibility, values closer to 1 indicate disorder
	
main_df will take as input a pandas dataframe instead of a path to a csv. 
It will output the result dictionary, the resampled pandas dataframe, and the new spacedelta.

# coastseg_time_and_space_analysis_matrix.py

	def main(transect_timeseries_path,
			 config_gdf,
			 output_folder,
			 transect_spacing,
			 which_timedelta,
			 which_spacedelta,
			 timedelta=None,
			 spacedelta=None):
		"""
		Performs timeseries and spatial series analysis cookbook on each
		transect in the transect_time_series matrix from CoastSeg
		inputs:
		transect_timeseries_path (str): path to the transect_time_series.csv
		config_gdf_path (str): path to the config_gdf.geojson
		output_folder (str): path to save outputs to
		which_timedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is what the timeseries is resampled at
		which_spacedelta (str): 'minimum' 'average' or 'maximum' or 'custom', this is the matrix is sampled at in the longshore direction
		timedelta (str, optional): the custom time spacing (e.g., '30D' is 30 days)
		beware of choosing minimum, with a mix of satellites, the minimum time spacing can be so low that you run into fourier transform problems
		spacedelta (int, optional): custom longshore spacing, do not make this finer than the input transect spacing!!!!
		outputs:
		new_matrix_path (str): path to the output matrix csv
		"""

Performs time and space analysis on entire matrix of shoreine data from CoastSeg. Will save a resampled matrix.