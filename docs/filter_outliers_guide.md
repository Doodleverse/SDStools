# Examples

## Need Help

    python filter_outliers_hampel_spacetime.py --help

## Example #1

- Removes the outliers from the `transect_time_series_coastsat.csv` from CoastSeg using the default parameters

  python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv"

## Example #2

This removes outliers from the `transect_time_series_coastsat.csv` using the following parameters :

- `-s`: controls the integer threshold for outlier detection and here its set to 3
- `-i`: controls the number iterations the hampel is applied for. Here its set to 5 meaning the filter will be applied for 5 iterations
- `-w`: control the window size as a percent of data length. Here we set the window size to be 20% of the length of the data

  python filter_outliers_hampel_spacetime.py -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv" -s 3 -i 5 -w 0.20
