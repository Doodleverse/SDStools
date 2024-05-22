# Usage Guide for denoise_inpainted_spacetime.py

Use this script to filter values, after using the script `inpaint_spacetime.py`, by applying a calibrated wavelet denoising to the space-time shoreline data.

The script reads a CSV file containing shoreline data and applies a calibrated wavelet filter to the data to filter out unstructured noise. The processed data is saved to a new CSV file, and a comparison plot of the original and filtered data matrices is generated and saved.

## Need Help

To view the help documentation for the script, use the following command:

```bash
python denoise_inpainted_spacetime.py --help
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

## Example #1: Basic Usage

This example filters the values from the `transect_time_series_coastsat_inpainted.csv` file using the default parameters:

```python
python denoise_inpainted_spacetime.py -f "/path/to/SDStools/example_data/transect_time_series_coastsat_inpainted.csv" -p 1 
```


Here's an example screenshot
![Screenshot from 2024-05-22 11-57-23](https://github.com/Doodleverse/SDStools/assets/3596509/ebb59bca-2824-4472-bf51-9c74f8c43466)

The (optional) plot is created. This shows the data as a 2d matrix of shoreline positions as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:

![Screenshot from 2024-05-22 11-57-57](https://github.com/Doodleverse/SDStools/assets/3596509/60d3c27d-3113-40e6-9c73-0bbd5ebd07c8)
