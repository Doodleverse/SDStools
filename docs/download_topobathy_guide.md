# Usage Guide for download_topobathymap_geojson.py and download_topobathymap_geotiff.py

This script downloads topobathymetric data using the `bathyreq` [package](https://github.com/NeptuneProjects/BathyReq). Currently only digital elevation model (DEM) data from the NOAA National Centers for Environmental Information (NCEI) database is supported.

There are two scripts, one based on defining the spatial extent with a geoJSON file, and the other defines the spatial region using a geoTIFF, for example one downloaded by CoastSeg.

## Need Help?

To view the help documentation for the script, use the following command:

```bash
python download_topobathymap_geojson.py --help
```

or 

```bash
python download_topobathymap_geotiff.py --help
```


## Common command line arguments

- `-f`: Sets the file that defines spatial extents

<details>
<summary>More details</summary>
The file should be geotiff when using `download_topobathymap_geotiff.py` and a geojson polygon when using `download_topobathymap_geojson.py`
</details>

- `-s`: Sets the file prefix to use for output data files
<details>
<summary>More details</summary>
A string that typically represents the name of the place where the data come from
</details>

- `-p`: If 1, make a plot
<details>
<summary>More details</summary>
A flag to make (or suppress) a plot
</details>

## Examples

### Example #1: Basic Usage with geojson

This example downloads topobathy data according to a geoJSON file

```python
python download_topobathymap_geojson.py -s "LongIsland" -f "/path/to/SDStools/example_data/longisland_example.geojson" -p 1
```

Here's an example screenshot

![Screenshot from 2024-05-22 13-53-56](https://github.com/Doodleverse/SDStools/assets/3596509/c99de21d-4ebb-487b-9a8a-6ed6d46d866b)

The (optional) plot is created. This is not intended to be a publication ready figure. This merely shows the data, as a convenience:

![LongIsland_topobathy_-74 29673896077358_-71 775261823626_40 2062276698526_41 06288754297591](https://github.com/Doodleverse/SDStools/assets/3596509/1b47d251-fe91-4cdd-bed5-b3b2a02d4aac)

### Example #2: Basic Usage with geotiff

This example downloads topobathy data according to a geoTIFF file

```python
python download_topobathymap_geotiff.py -s "Klamath" -f "/path/to/SDStools/example_data/2017-10-02-18-57-29_L8_GREATER_KLAMATH_ms.tif" -p 1 
```

The (optional) plot is created. This is not intended to be a publication ready figure. This merely shows the data, as a convenience:

![Klamath_topobathy_-124 10012980366487_-124 04540285887134_41 28140338603512_41 55394321487515](https://github.com/Doodleverse/SDStools/assets/3596509/6efe4718-8a93-4633-98f7-4fbdf0ad78ae)