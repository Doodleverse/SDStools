# SDStools/fromMark/visualization

This will contain visualization scripts for shoreline data obtained through CoastSeg.

Requirements: Python>=3.7, numpy, matplotlib, cv2, pandas, scipy, geopandas, shapely

# shoreline_video.py

	def make_shoreline_video(shorelines,
							 config_gdf,
							 transect_timeseries,
							 transect_id,
							 sitename,
							 frame_rate):
		"""
		Makes a video of the shorelines from CoastSeg with a select transect timeseries.
		inputs:
		shorelines (str): path to the extracted_shorelines_lines.geojson
		config_gdf (str): path to the config_gdf.geojson
		transect_timeseries (str): path to the transect_timeseries.csv or transect_timeseries_tidally_corrected_matrix.csv
		transect_id (str): specifiy which transect to display
		sitename (str): provide a name for the site
		frame_rate (int): frame rate for the output video
		"""

# trend_maps.py

	def get_trends(transect_timeseries_path,
				   config_gdf_path,
				   t_min,
				   t_max):
		"""
		Computes linear trends with LLS on each transect's timeseries data
		Saves geojson linking transect id's to trend values
		Transect length is scaled to trend magnitude, direction corresponds to sign of trend value
		inputs:
		transect_timeseries (str): path to the transect_timeseries csv (or transect_timeseries_tidally_corrected_matrix.csv)
		config_gdf_path (str): path to the config_gdf (.geojson), it's assumed these are in WGS84
		outputs:
		save_path (str): path to geojson with adjusted transects (in WGS84), trends, csv path, timeseries plot path, trend plot path
		"""







