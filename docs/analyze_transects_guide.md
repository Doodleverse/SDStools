# Usage Guide for analyze_transects.py

This script analyzes each transect in a csv file of SDS data and computes a number of statistics for each transect. These statistics summarize a variety of quantities that are outlined below.

Each variable in the output CSV file, and their meaning:

* `stationarity`: 1=statistically significant stationarity (i.e., a lack of trend. Otherwise, 0). This uses the Augmented Dickey-Fuller test [docs here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)
* `autocorr_min`: minimum autocorrelation
* `lag_min`: lag associated with minimum autocorrelation
* `entropy`: If this value is high, then the timeseries is probably unpredictable. [details here](from https://en.wikipedia.org/wiki/Approximate_entropy)
* `linear_trend_slopes`: slope of linear fit
* `linear_trend_intercepts`: intercept of linear fit
* `linear_trend_rvalues`: correlation of linear fit with underlying data
* `linear_trend_pvalues`: significance of linear fit (probability of no trend)
* `linear_trend_stderr`: standard error in estimate of linear trend slope
* `linear_trend_intercept_stderr`: standard error of linear trend intercept

Each variable in the output NPZ file, and their meaning:

* `trend2d`: 2d matrix of shoreline trend (decomposed using the STL technique)
* `season2d`: 2d matrix of shoreline seasonality (decomposed using the STL technique)
* `auto2d`: 2d matrix of autocorrelation 
* `weights2d`: 2d matrix of autocorrelation (decomposed using the STL technique)
* `cs_transects_vector`: vector of transect names
* `cs_dates_vector`: vector of shoreline dates
* `cs_data_matrix_demeaned`: the demeaned version of the data used to compute statistics
* `df_resampled`: the regular-in-time (resampled) version of the data used to compute statistics



## Need Help

To view the help documentation for the script, use the following command:

```bash
python analyze_transects.py --help
```

## Command line arguments
- `-f`: Sets the file (csv) to be analyzed

<details>
<summary>More details</summary>
The csv format file shoule contain shoreline positions in each cell, with rows as time and columns as transects
</details>

- `-p`: If 1, make a plot

<details>
<summary>More details</summary>
A flag to make (or suppress) a plot
</details>



## Examples

# Example #1: Basic Usage

This example analyzes the `transect_time_series_coastsat_nooutliers_inpainted.csv` file using the default parameters:

```python
python analyze_transects.py -f /path/to/SDStools/example_data/transect_time_series_coastsat_nooutliers_inpainted.csv -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 12-10-29](https://github.com/Doodleverse/SDStools/assets/3596509/872906bc-1af2-4f7c-b431-2b0ecbefae5b)

The (optional) plot is created. This shows the data as a 2d matrix of a) trend in shoreline position, b) seasonal shoreline excursion, and c) autocorrelation as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:

![transect_time_series_coastsat_nooutliers_inpainted_stats_timeseries](https://github.com/Doodleverse/SDStools/assets/3596509/908503b7-ff50-47ab-9a0c-b5cb82e56e2b)

This shows the contents of the compressed numpy archive (.npz) file.

![Screenshot from 2024-05-22 12-11-11](https://github.com/Doodleverse/SDStools/assets/3596509/7555c7ae-5c22-4321-a394-5a40dea37d63)


## Future work

1. add https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.kpss.html option
2. add https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.levinson_durbin.html for autoregressive test