# Usage Guide for detrend_relstart_transect_timeseries.py

Use this script to detrend each transect relative to the initial condition. The initial condition is defined by the number, `N`, or data points to use at the start. Must be 1 or more. Typically this number would be less than 20.

The script reads a CSV file containing shoreline data and detrends each based on the mean of the first `N` data points on that transect. The processed data is saved to a new CSV file, and a comparison plot of the original and filtered data matrices is generated and saved.

## Need Help

To view the help documentation for the script, use the following command:

```bash
python detrend_relstart_transect_timeseries.py --help
```

## Command line arguments
- `-f`: Sets the file (csv) to be analyzed

<details>
<summary>More details</summary>
The csv format file should contain shoreline positions in each cell, with rows as time and columns as transects
</details>

- `-p`: If 1, make a plot

<details>
<summary>More details</summary>
A flag to make (or suppress) a plot
</details>


## Examples

## Example #1: Basic Usage

This example detrends the values from the `transect_time_series_coastsat_inpainted.csv` file using the default parameters:

```python
python detrend_relstart_transect_timeseries.py -f "/path/to/SDStools/example_data/transect_time_series_coastsat_inpainted.csv" -p 1 
```


Here's an example screenshot
![Screenshot from 2024-05-22 12-00-47](https://github.com/Doodleverse/SDStools/assets/3596509/2a285ef7-fda6-41ca-838f-2cd190c63e9e)

The (optional) plot is created. This shows the data as a 2d matrix of shoreline positions as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:

![transect_time_series_coastsat_nooutliers_inpainted_detrend](https://github.com/Doodleverse/SDStools/assets/3596509/2d94c747-6560-422f-9eb9-6d3828d05d54)














