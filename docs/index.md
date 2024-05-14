# SDS Tools

Welcome to SDS tools!

SDS tools is a comprehensive suite of tools designed for the robust analysis of time series shoreline data. It provides functionalities for filtering outliers, conducting in-depth data analysis, and identifying trends, thereby enabling more accurate and reliable shoreline data interpretation.

This website and the code is currently under active development.

## Scripts Available for Use

1. `filter_outliers_hampel_spacetime.py`: Removes outliers from time series data using the Hampel filter applicable for `raw_transect_time_series.csv` & `tidally_corrected_transect_time_series.csv`

2. `inpaint_spacetime.py`: Use this script to fill in the missing values after using the script `filter_outliers_hampel_spacetime.py` to remove outliers from the time series data.
