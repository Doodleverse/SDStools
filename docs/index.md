# SDS Tools

Welcome to SDS tools!

SDS tools is a comprehensive suite of tools designed for the robust analysis of time series shoreline data. It provides functionalities for filtering outliers, conducting in-depth data analysis, and identifying trends, thereby enabling more accurate and reliable shoreline data interpretation.

This website and the code is currently under active development.

## SDS Scripts Available for Use

1. `filter_outliers_hampel_spacetime.py`: Removes outliers from time series data using the Hampel filter applicable for `raw_transect_time_series.csv` & `tidally_corrected_transect_time_series.csv`. This takes a csv file of time (rows) versus transects (columns) and outputs a new csv file in the same format.

2. `inpaint_spacetime.py`: Use this script to fill in the missing values after using the script `filter_outliers_hampel_spacetime.py` to remove outliers from the time series data. This takes a csv file of time (rows) versus transects (columns) and outputs a new csv file in the same format.

3. `detrend_relstart_transect_timeseries.py`: Use this script to detrend each transect relative to a stable initial value.  This takes a csv file of time (rows) versus transects (columns) and outputs a new csv file in the same format.

4. `analyze_transects.py`: Use this script to statistically analyze data after using the script `inpaint_spacetime.py` to analyze the time series data. This generates one csv file that contains statistics per transect, and an npz file that contains 2d arrays of linear trend, autocorrelation, etc.  This takes a csv file of time (rows) versus transects (columns) and outputs a new csv file in the same format.

5. `denoise_inpainted_spacetime.py`: Use this script to denoise an inpainted SDS matrix using a Morlet wavelet (experimental). This results in a smoother dataset, but carries out filtering in both space and time.


## Auxiliary Scripts Available for Use

1. `download_era5_dataframe_singlelocation.py`: Use this script to download a time-series of bulk wave statistics at a single point, using a geoJSON polygon as input

2. `download_era5_dataframe_grid.py`: Use this script to download a time-series of bulk wave statistics over a 2d grid, using a geoJSON polygon as input

3. `download_topobathymap_geojson.py`: Use this script to download a topobathymetric map at approx ~3m resolution, and store as a geoTIFF. Takes geoJSON polygon as input

4. `download_topobathymap_geotiff.py`: Use this script to download a topobathymetric map at approx ~3m resolution, and store as a geoTIFF. Takes geoTIFF, e.g. a CoastSeg image, as input


## Test script

`test_scripts.sh` is a utility script for testing all scripts using example datasets