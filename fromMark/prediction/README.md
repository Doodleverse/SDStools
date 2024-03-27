# SDStools/fromMark/prediction

Requirements: python>=3.7, tensorflow<=1.14.0, pandas, matplotlib, scikit-learn, numpy, shapely, geopandas, statsmodels

Some tools for building data-driven predictive models from satellite shoreline data.

# Before you use

Generate tidally corrected shoreline data with CoastSeg.

# Step 1: lstm_parallel_coastseg.py

Run data through SDSTools/fromMark/analysis/coastseg_time_and_space_analysis_matrix.py.

This ensures that you get an evenly sampled matrix in the time and space domain. There is a ton more information this script outputs too.

We will work with the file 'timeseries_mat_resample_time_space.csv'.

This file is indexed by time, with column names corresponding to transect ids, 
and the matrix filled with tidally corrected cross-shore positions. 
These positions are referenced to a position along a transect from the config_gdf.geojson.

So gather 'timeseries_mat_resample_time_space.csv' and the 'config_gdf.geojson' you used in CoastSeg.

The script to use is lstm_parallel_coastseg.py.

	def main(sitename,
			 coastseg_matrix_path,
			 bootstrap=30,
			 num_prediction=30,
			 num_epochs=2000,
			 units=64,
			 batch_size=32,
			 look_back=3,
			 split_percent=0.80,
			 freq='30D'):
		"""
		Trains parallel LSTM model on shoreline data
		inputs:
		sitename (str): name of the sitename/study area
		coastseg_matrix_path (str): path to the matrix output from coastseg_time_and_space_analysis_matrix.py
		bootstrap (int): number of times to train the model
		num_prediction (int): number of timesteps to project model
		num_epochs (int): number of epochs to train the model
		units (int): number of units for LSTM layers
		batch_size (int): batch size for training
		look_back (int): number of previous timesteps to use to predict next value
		split_percent (int): fraction of timeseries to use as training data
		freq (str): timestep, this should match the timedelta of the coastseg_matrix
		"""

Try running this on your data with the default values for
bootstrap, num_prediction, num_epochs, units, batch_size, look_back, and split_percent.

Make sure freq matches the time-spacing you used in SDSTools/fromMark/analysis/coastseg_time_and_space_analysis_matrix.py.

This will output a two plots and two csvs for each transect (one for the observed temporal range and one for the forecasted temporal range).

It will also output csvs for each training run and a final plot showing the loss curves over each training run.

It will also output two csvs (observed period and forecast period)
where all of the data is stacked (with an additional column 'transect_id' indicating which transect the data is from).
 
 'forecast_stacked.csv' and 'predict_stacked.csv'
 These two files will be used in the next step.
 
 # Step 2: lstm_2D_projection_coastseg.py
 
 This script takes the outputs from lstm_parallel_coastseg.py and makes GIS outputs:
 mean shorelines and confidence interval polygons.
 
	 def main(sitename,
			 coastseg_matrix_path,
			 forecast_stacked_df_path,
			 predict_stacked_df_path,
			 save_folder,
			 config_gdf_path,
			 switch_dir=False):
		"""
		Takes projected cross-shore positions and uncertainties and constructs 2D projected shorelines/uncertainties
		Saves these to two shapefiles (mean shorelines and confidence intervals)
		inputs:
		sitename: Name of site (str)
		coastseg_matrix_path: path to the resampled coastseg matrix (str)
		model_folder: path ot the folder containing lstm results (str)
		save_folder: folder to save projected shoreline shapefiles to (str)
		config_gdf_path: path to config_gdf.geojson containing transects (str)
		switch_dir: Optional, if True, then transect direction is reversed
		"""

So for this script, you need the path to the 'timeseries_mat_resample_time_space.csv' which gets made
from SDSTools/fromMark/analysis/coastseg_time_and_space_analysis_matrix.py.

You also need the path to the stacked forecasts and predictions which are output from lstm_parallel_coastseg.py:
 'forecast_stacked.csv' and 'predict_stacked.csv'
 
 You will also need the path to the config_gdf made during CoastSeg:
 'config_gdf.geojson'
 
 Give it a folder to save everything to (save_folder). 
 
 If your transects are in the wrong direction you can set switch_dir to True.
 
 This will output new csvs for the observed temporal period and forecasted temporal period, and for each transect with the additional columns:
 
 eastings_mean_utm, northings_mean_utm, eastings_upper_utm, northings_upper_utm, eastings_lower_utm, northings_lower_utm
 
 eastings_mean_wgs84, northings_mean_wgs84, eastings_upper_wgs84, northings_upper_wgs84, eastings_lower_wgs84, northings_lower_wgs84
 
 So basically all the model outputs' coordinates in utm and wgs84.
 
 It will also stack these csvs into two dataframes (forecast_stacked.csv and predict_stacked.csv).
 
 Finally, it will output four geojsons:
 * One will have the mean shorelines during the observed temporal period.
 * One will have the confidence interval polygons during the observed temporal period.
 * One will have the mean shorelines during the forecasted temporal period.
 * One will have the confidence interval polygons for the forecasted temporal period.
 
 These geojson will have the timestamps and year as columns that may be useful for constructing maps.
 
 # Future Work
 
 Will potentially add in independent variables (e.g., waves).
 


