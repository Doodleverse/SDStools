# Usage Guide for download_era5_dataframe_singlelocation.py and download_era5_dataframe_grid.py

This script downloads wave data from ERA5-reanalysis

## Need Help

To view the help documentation for the script, use the following command:

```bash
python download_era5_dataframe_singlelocation.py --help
```

or 

```bash
python download_era5_dataframe_grid.py --help
```

## Common command line arguments

- `-i`: Sets the file prefix to use for output data files
<details>
<summary>More details</summary>
A string that typically represents the name of the place where the data come from
</details>

- `-p`: If 1, make a plot
<details>
<summary>More details</summary>
A flag to make (or suppress) a plot
</details>

- `-a`: start year

- `-b`: end year


## Examples

### Example #1: single point

Additional command line arguments:

- `-x`: longitude
- `-y`: latitude
- `-w`: water depth, either 'deep' or 'intermediate'
<details>
<summary>More details</summary>
Intermediate water is defined as water depths where d/L is between 0.05 and 0.5, where d = water depth in meters, and L is wavelength in meters
</details>


This example downloads waves at a single point

```python
python download_era5_dataframe_singlelocation.py  -i "my_location" -a 1984 -b 2023 -x -160.8052  -y 64.446 -w intermediate -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 13-34-51](https://github.com/Doodleverse/SDStools/assets/3596509/a68a60b6-3af5-468c-b8f2-9cec188fffcb)

The (optional) plots created. This is not intended to be a publication ready figures. This merely shows the data, as a convenience:

![slapton_reanalysis-era5-single-levels_timeseries_Hs](https://github.com/Doodleverse/SDStools/assets/3596509/9807cc65-ab4e-4a8f-a0b8-2927a609cf9d)

[slapton_reanalysis-era5-single-levels_joint_and_marginal_distributions_Hs_Tpeak](https://github.com/Doodleverse/SDStools/assets/3596509/39bbecca-04e4-4378-ada5-8134d3dc9638)
![slapton_reanalysis-era5-single-levels_joint_and_marginal_distributions_Hs_Tmean](https://github.com/Doodleverse/SDStools/assets/3596509/0117b4bf-10c3-430b-8fab-9760a996f91b)
![slapton_reanalysis-era5-single-levels_joint_and_marginal_distributions_Hs_Dmean](https://github.com/Doodleverse/SDStools/assets/3596509/1163ba4f-99b8-4c3e-97f3-638c8d14af0f)


### Example #2: grid

Additional command line arguments:

- `-f`: Sets the file (geojson) to be analyzed

<details>
<summary>More details</summary>
The geoJSON format file should contain a polygon
</details>

For 2D measurements, at 'my_location', from 1984 to 2023 inclusive, according to a geoJSON file region of interest

```python
python download_era5_dataframe_grid.py -i "my_location" -a 1984 -b 2023 -f /path/to/your/geoJSON file -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 13-23-48](https://github.com/Doodleverse/SDStools/assets/3596509/e525ebea-e394-4981-ad29-d744a2197ec2)

The (optional) plots created. This is not intended to be a publication ready figures. This merely shows the data, as a convenience:

![AK_mean_2d](https://github.com/Doodleverse/SDStools/assets/3596509/54f24f5a-c600-451d-99e7-63eda32c313e)