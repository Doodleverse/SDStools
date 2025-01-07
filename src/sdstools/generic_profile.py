"""
Profiling tool with xarrays I guess?
"""

import rioxarray
import geopandas
import pandas
import os

def extract_along_line(xarr, line, raster_res):
    """
    gets profile data
    """
    line_length = line.length
    n_samples = int(line.length/raster_res)
    profile = [None]*n_samples
    for i in range(n_samples):
        # get next point on the line
        point = line.interpolate(i / n_samples - 1., normalized=True)
        # access the nearest pixel in the xarray
        value = xarr.sel(x=point.x, y=point.y, method="nearest").data
        profile[i] = value

    profile_df = pd.DataFrame({'x':range(n_samples)*raster_res,
                               'z':profile})
    return profile_df

def get_profile_csv(raster_path,
                    raster_res,
                    transects_path,
                    output_folder):
    """
    Profiles raster with transects
    saves csv for each transect

    inputs:
    raster_path (str): path to raster
    raster_res (float): horizontal resolution of raster
    transects_path (str): path to transects
    output_folder (str): path to save profiles to
    
    """
    ##sample kde
    raster = rioxarray.open_rasterio(raster_path).squeeze()
    transects = gpd.read_file(transects_path)
    profiles [None]*len(transects)
    for idx,row in transects.iterrows():
        transect = row['geometry']
        profile = extract_along_line(raster, transect, raster_res)
        profile.to_csv(os.path.join(output_folder, 'profile_'+str(idx)+'.csv'))










        
