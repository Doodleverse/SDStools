# Usage Guide for inpaint_spacetime.py

Use this script to fill in the missing values after using the script `filter_outliers_hampel_spacetime.py` to impute (inpaint) missing data from time series data.

The script `inpaint_spacetime.py` reads a CSV file containing shoreline data and applies biharmonic inpainting to fill missing values in the data matrix. The processed data is saved to a new CSV file, and a comparison plot of the original and inpainted data matrices is generated and saved.

## Need Help

To view the help documentation for the script, use the following command:

```bash
python inpaint_spacetime.py --help
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

This example inpaints the missing values from the `raw_transect_time_series_nooutliers.csv` file using the default parameters:

```python
python inpaint_spacetime.py -f "/path/to/SDStools/example_data/raw_transect_time_series_nooutliers.csv" -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 11-47-09](https://github.com/Doodleverse/SDStools/assets/3596509/4e4dda3d-4dae-411f-a737-54078e172dff)

The (optional) plot is created. This shows the data as a 2d matrix of shoreline positions as a function of time and transect. This plot is purely for QA/QC purposes and is not intended to be a publication ready figure. This merely shows the data, as a convenience:
![Screenshot from 2024-05-22 11-47-28](https://github.com/Doodleverse/SDStools/assets/3596509/eed00123-bc8f-4e72-9604-dd5839a7d9bc)

