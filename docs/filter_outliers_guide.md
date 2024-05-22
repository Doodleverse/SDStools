# Usage Guide for filter_outliers_hampel_spacetime.py

This script removes outliers from time series data produced by CoastSeg using the Hampel filter, which identifies outliers by comparing each data point to the median of a sliding window centered around it. If a data point deviates from the median by more than a specified threshold, it is considered an outlier and can be replaced with the median value. This process helps in cleaning and smoothing time series files, such as `raw_transect_time_series.csv` & `tidally_corrected_transect_time_series.csv`, based on user-defined parameters for threshold, iterations, and window size.

## Need Help

To view the help documentation for the script, use the following command:

```bash
python filter_outliers_hampel_spacetime.py --help
```

## Command line arguments

- `-f`: Sets the file (csv) to be analyzed

<details>
<summary>More details</summary>
The csv format file shoule contain shoreline positions in each cell, with rows as time and columns as transects
</details>

- `-s`: Sets the integer threshold for outlier detection. Here it is set to 3.

<details>
<summary>More details</summary>
The threshold determines how many standard deviations a data point must deviate from the median within a sliding window to be considered an outlier.
If a data point's deviation exceeds this threshold, it is flagged as an outlier and can be replaced by the median value of the window.
</details>

- `-i`: Sets the number of iterations for applying the Hampel filter. Here it is set to 5, meaning the filter will be applied for 5 iterations.

<details>
<summary>More details</summary>
The number of iterations in the Hampel filter determines how many times the filter is applied to the data. Multiple iterations can enhance the effectiveness of outlier removal by progressively refining the data and eliminating any residual outliers that may not have been detected in earlier passes.
</details>

- `-w`: Sets the window size as a percentage of the data length. Here it is set to 20% of the data length.
<details>
<summary>More details</summary>
The window size in the Hampel filter specifies the span of data points (as a percentage of the total data length) around each target point that are used to calculate the median and median absolute deviation. This sliding window determines the local context for outlier detection, with a larger window capturing more data points and a smaller window providing a more localized analysis.
</details>

- `-p`: If 1, make a plot
<details>
<summary>More details</summary>
A flag to make (or suppress) a plot
</details>


## Examples

### Example #1: Basic Usage

This example removes the outliers from the `raw_transect_time_series.csv` file using the default parameters:

```python
python filter_outliers_hampel_spacetime.py -f "/path/to/SDStools/example_data/raw_transect_time_series.csv" -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 11-37-37](https://github.com/Doodleverse/SDStools/assets/3596509/ffc31fa4-bc4a-4aa5-b7e1-0c4f4489db14)

The (optional) plot is created. This shows the data as a 2d matrix of shoreline positions as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:
![Screenshot from 2024-05-22 11-37-54](https://github.com/Doodleverse/SDStools/assets/3596509/097a8be6-b372-4b27-a263-82b25f4a5069)


### Example #2: Custom Parameters

This removes outliers from the `raw_transect_time_series.csv` using the following parameters :

```python
python filter_outliers_hampel_spacetime.py -f "/path/to/SDStools/example_data/raw_transect_time_series.csv" -s 3 -i 5 -w 0.20
```

Here's an example screenshot:

![Screenshot from 2024-05-22 11-42-49](https://github.com/Doodleverse/SDStools/assets/3596509/551d4f8d-f724-48bd-a2d8-ce457c288595)


The (optional) plot is created. This shows the data as a 2d matrix of shoreline positions as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:
![Screenshot from 2024-05-22 11-44-00](https://github.com/Doodleverse/SDStools/assets/3596509/6fccc786-a8f0-40a3-896c-86fbd2abae2b)