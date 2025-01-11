import os
import numpy as np
import glob
from coastsat import SDS_preprocess, SDS_tools

from osgeo import gdal
from skimage import morphology
import re


def preprocess_single(
    fn, satname, cloud_mask_issue, pan_off, collection='C02', do_cloud_mask=True, s2cloudless_prob=60
):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image.
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018
    Modified by Sharon Fitzpatrick Batiste 2025

    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. For L7, L8 and S2 this
        is a list of filenames, one filename for each band at different
        resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images
    pan_off : boolean
        if True, disable panchromatic sharpening and ignore pan band
    collection: str
        Landsat collection 'C02'
    do_cloud_mask: boolean
        if True, apply the cloud mask to the image. If False, the cloud mask is not applied.
    s2cloudless_prob: float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask

    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """
    if isinstance(fn, list):
        fn_to_split = fn[0]
    elif isinstance(fn, str):
        fn_to_split = fn
    # split by os.sep and only get the filename at the end then split again to remove file extension
    fn_to_split = fn_to_split.split(os.sep)[-1].split(".")[0]
    # search for the year the tif was taken with regex and convert to int
    year = int(re.search("[0-9]+", fn_to_split).group(0))
    # after 2022 everything is automatically from Collection 2
    if collection == "C01" and year >= 2022:
        collection = "C02"

    # =============================================================================================#
    # L5 images
    # =============================================================================================#
    if satname == "L5":
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_mask = fn[1]
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the same as the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []
    # =============================================================================================#
    # L7, L8 and L9 images
    # =============================================================================================#
    elif satname in ["L7", "L8", "L9"]:
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_pan = fn[1]
        fn_mask = fn[2]
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask and get the QA from the first band
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]

        if not do_cloud_mask:
            cloud_mask = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # if panchromatic sharpening is turned off
        if pan_off:
            # ms bands are untouched and the extra image is empty
            im_extra = []

        # otherwise perform panchromatic sharpening
        else:
            # read panchromatic band
            data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            bands = [
                data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)
            ]
            im_pan = bands[0]

            # pansharpen all of the landsat 7 bands
            if satname == "L7":
                try:
                    im_ms_ps = SDS_preprocess.pansharpen(im_ms[:, :, [0,1,2,3,4]], im_pan, cloud_mask)
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [1, 2, 3]]
                    # add downsampled Blue and SWIR1 bands
                    im_ms_ps = np.append(im_ms[:, :, [0]], im_ms_ps, axis=2)
                    im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [4]], axis=2)

                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

            # pansharpen Blue, Green, Red for Landsat 8 and 9
            elif satname in ["L8", "L9"]:
                try:
                    im_ms_ps = SDS_preprocess.pansharpen(im_ms[:, :, [0, 1, 2,3,4]], im_pan, cloud_mask)
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [0, 1, 2]]
                    # add downsampled NIR and SWIR1 bands
                    im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [3, 4]], axis=2)
                
                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

    # =============================================================================================#
    # S2 images
    # =============================================================================================#
    if satname == "S2":
        # read 10m bands (R,G,B,NIR)
        fn_ms = fn[0]
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = SDS_preprocess.read_bands(fn_ms, satname)
        im_ms = np.stack(bands, 2)
        im_ms = im_ms / 10000  # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((nrows, ncols)).astype("bool")
            return im_ms, georef, cloud_mask, [], [], []

        # read 20m band (SWIR1) from the first band
        fn_swir = fn[1]
        im_swir = SDS_preprocess.read_bands(fn_swir)[0] / 10000  # TOA scaled to 10000
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im_ms, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn_mask = fn[2]
        im_QA = SDS_preprocess.read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask_QA60 = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = SDS_preprocess.create_s2cloudless_mask(cloud_prob, s2cloudless_prob)
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60,cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = SDS_preprocess.pad_edges(im_swir, im_nodata)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, morphology.square(5))
            # update cloud mask with all the nodata pixels
            # v0.1.40 change : might be bug
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
        else:  # apply the cloud mask
            # compute cloud mask using QA60 band
            cloud_mask_QA60 = SDS_preprocess.create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = SDS_preprocess.create_s2cloudless_mask(cloud_prob, s2cloudless_prob)
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60,cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = SDS_preprocess.get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = SDS_preprocess.pad_edges(im_swir, im_nodata)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = SDS_preprocess.get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, morphology.square(5))
            # move cloud mask to above if statement to avoid bug in v0.1.40

        # no extra image
        im_extra = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata

def save_pansharpened_jpg(
    filename,
    tif_paths,
    satname,
    sitename,
    cloud_thresh,
    cloud_mask_issue,
    filepath_data,
    collection,
    **kwargs,
):
    """
    Save a jpg for a single set of tifs

    KV WRL 2018

    Arguments:
    -----------
    cloud_thresh: float
        value between 0 and 1 indicating the maximum cloud fraction in
        the cropped image that is accepted
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels
        are erroneously being masked on the images

    Returns:
    -----------
    Creates RGB, NIR, SWIR as .jpg in:
    RGB saved under data/preprocessed/RGB
    NIR saved under data/preprocessed/NIR
    SWIR saved under data/preprocessed/SWIR
    """
    if "apply_cloud_mask" in kwargs:
        do_cloud_mask = kwargs["apply_cloud_mask"]
    else:
        do_cloud_mask = True
    # create subfolder to store the jpg files
    jpg_directory = os.path.join(filepath_data, sitename, "jpg_files", "preprocessed")
    os.makedirs(jpg_directory, exist_ok=True)
    # get locations to each tif file for ms, pan, mask, swir (if applicable for satname)
    fn = SDS_tools.get_filenames(filename, tif_paths, satname)
    # preprocess the image and perform pansharpening
    im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(
        fn, satname, cloud_mask_issue, False, collection, do_cloud_mask
    )
    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = np.divide(
        sum(sum(cloud_mask.astype(int))), (cloud_mask.shape[0] * cloud_mask.shape[1])
    )
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        return
    # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)

    # compute updated cloud cover percentage (without no data pixels)
    cloud_cover = np.divide(
        sum(sum(cloud_mask_adv.astype(int))), (sum(sum((~im_nodata).astype(int))))
    )
    # skip image if cloud cover is above threshold
    if cloud_cover > cloud_thresh or cloud_cover == 1:
        return
    # save .jpg with date and satellite in the title
    date = filename[:19]
    # get cloud mask parameter

    SDS_preprocess.create_jpg(
        im_ms, cloud_mask, date, satname, jpg_directory, im_nodata=im_nodata, **kwargs
    )


### give it a config.json file and it will do the rest
##
##config_json_path = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime01-10-25__03_58_44\config.json'
##
### read the config
##import json
##with open(config_json_path) as json_file:
##    config = json.load(json_file)
##    roi_ids = config['roi_ids']
##
##    # read the settings needed to create jpgs
##    settings = config['settings']
##    cloud_threshold = settings.get('cloud_thresh', 0.5)
##    cloud_mask_issue = settings.get('cloud_mask_issue', False)
##    apply_cloud_mask = settings.get('apply_cloud_mask', True)
##
##
##    # load the inputs for each ROI 
##    for roi_id in roi_ids:
##        inputs = config[roi_id]
##        print(inputs)
##        satlist = inputs['sat_list']
##        for sat in satlist:
##            print(sat)
##            tif_paths = SDS_tools.get_filepath(inputs, sat)
##
##            # get a list of all the ms files for the given satellite
##            ms_folder = os.path.join(inputs["filepath"],inputs["sitename"], sat, "ms")
##            if not os.path.exists(ms_folder):
##                raise Exception(f"Folder {ms_folder} does not exist")
##
##            ms_files = glob.glob(os.path.join(ms_folder, "*.tif"))
##            for ms_file in ms_files:
##                print(ms_file)
##                save_pansharpened_jpg(
##                    filename=os.path.basename(ms_file),
##                    tif_paths=tif_paths,
##                    satname=sat,
##                    sitename=inputs["sitename"],
##                    cloud_thresh=cloud_threshold,
##                    cloud_mask_issue=cloud_mask_issue,
##                    filepath_data=inputs["filepath"],
##                    collection=inputs["landsat_collection"],
##                    apply_cloud_mask=apply_cloud_mask,
##                )
