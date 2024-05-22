############# SDS analyses

### filter outliers
python filter_outliers_hampel_spacetime.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat.csv -p 1

### inpaint missing values
python inpaint_spacetime.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat_nooutliers.csv -p 1 

### inpaint missing values
python denoise_inpainted_spacetime.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat_inpainted.csv -p 1 

### detrend each transect
python detrend_relstart_transect_timeseries.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat_nooutliers_inpainted.csv -N 10 -p 1

### compute time-series stats
python analyze_transects.py -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/transect_time_series_coastsat_nooutliers_inpainted.csv -p 1

############# Auxiliary analyses

#### download wave data over a grid from ERA5
python download_era5_dataframe_grid.py -i "AK" -a 1984 -b 2023 -f /media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/ak_example.geojson -p 1

#### download wave data at a single location from ERA5
python download_era5_dataframe_singlelocation.py -f "AK" -a 1984 -b 2023 -x -160.8052  -y 64.446 -w intermediate -p 1

### download a topobathy map based on geojson polygon
python download_topobathymap_geojson.py -s "LongIsland" -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/longisland_example.geojson" -p 1

### download a topobathy map based on geojson polygon
python download_topobathymap_geotiff.py -s "Klamath" -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/2017-10-02-18-57-29_L8_GREATER_KLAMATH_ms.tif" -p 1