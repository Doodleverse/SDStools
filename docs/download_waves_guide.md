# Usage Guide for download_era5_dataframe_singlelocation.py and download_era5_dataframe_grid.py

This script downloads wave data from ERA5-reanalysis

## Need Help

To view the help documentation for the script, use the following command:

```bash
python download_era5_dataframe_singlelocation.py --help
```

```bash
python download_era5_dataframe_grid.py --help
```

# Examples

## Example #1: Basic Usage

This example downloads waves at a single point

```python
python download_era5_dataframe_singlelocation.py -f 
```

or, for 2D measurements

```python
python download_era5_dataframe_grid.py -f 
```

<!-- 

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
</details> -->
