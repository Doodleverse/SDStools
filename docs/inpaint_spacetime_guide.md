# Usage Guide for inpaint_spacetime.py

Use this script to fill in the missing values after using the script `filter_outliers_hampel_spacetime.py` to remove outliers from the time series data.

The script `inpaint_spacetime.py` reads a CSV file containing shoreline data and applies biharmonic inpainting to fill missing values in the data matrix. It supports two inpainting methods: a default method and a "smart" method that also removes small objects and holes before inpainting. The processed data is saved to a new CSV file, and a comparison plot of the original and inpainted data matrices is generated and saved.

## Need Help

To view the help documentation for the script, use the following command:

```bash
python inpaint_spacetime.py --help
```

# Examples

## Example #1: Basic Usage

This example inpaints the missing values from the `raw_transect_time_series_nooutliers.csv` file using the default parameters:

```python
python inpaint_spacetime.py -f "C:\development\doodleverse\coastseg\CoastSeg\sessions\pls\ID_rpu1_datetime04-26-24__04_25_54\raw_transect_time_series_nooutliers.csv"
```

## Example #2: Custom Parameters

This removes outliers from the `raw_transect_time_series_nooutliers.csv` using the following parameters :

```python
python inpaint_spacetime.py -f "C:\development\doodleverse\coastseg\CoastSeg\sessions\pls\ID_rpu1_datetime04-26-24__04_25_54\raw_transect_time_series_nooutliers.csv" -m "smart"
```

- `-m`: Sets the inpainting method to use. Uses the default method by default.

<details>
<summary>More details</summary>
Setting the method to "smart" in the script applies biharmonic inpainting to the data matrix after removing small objects and holes from the mask of missing values.
</details>
