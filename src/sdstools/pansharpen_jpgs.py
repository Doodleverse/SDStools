import os
import numpy as np
import imageio
import glob
from coastsat import SDS_preprocess, SDS_tools
import skimage
from typing import List, Set, Tuple

from osgeo import gdal
from skimage import morphology
import re

def scale(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """returns resized matrix with shape(rows,cols)
        for 2d discrete labels
        for resizing 2d integer arrays
    Args:
        im (np.ndarray): 2d matrix to resize
        nR (int): number of rows to resize 2d matrix to
        nC (int): number of columns to resize 2d matrix to

    Returns:
        np.ndarray: resized matrix with shape(rows,cols)
    """
    src_rows = len(matrix)  # source number of rows
    src_cols = len(matrix[0])  # source number of columns
    tmp = [
        [
            matrix[int(src_rows * r / rows)][int(src_cols * c / cols)]
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return np.array(tmp).reshape((rows, cols))

def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn

def matching_datetimes_files(dir1: str, dir2: str) -> Set[str]:
    """
    Get the matching datetimes from the filenames in two directories.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        Set[str]: A set of strings representing the common datetimes.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Create sets of the date-time parts of the filenames in each directory
    files1_dates = {
        re.search(pattern, filename).group(0)
        for filename in files1
        if re.search(pattern, filename)
    }
    files2_dates = {
        re.search(pattern, filename).group(0)
        for filename in files2
        if re.search(pattern, filename)
    }

    # Find the intersection of the two sets
    matching_files = files1_dates & files2_dates

    return matching_files

def get_full_paths(
    dir1: str, dir2: str, common_dates: Set[str]
) -> Tuple[List[str], List[str]]:
    """
    Get the full paths of the files with matching datetimes.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        common_dates (Set[str]): A set of strings representing the common datetimes.

    Returns:
        Tuple[List[str], List[str]]: Two lists of strings representing the full paths of the matching files in dir1 and dir2.
    """
    # Get the filenames in each directory
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    # Define a pattern to match the date-time part of the filenames
    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    )  # Matches YYYY-MM-DD-HH-MM-SS

    # Find the full paths of the files with the matching date-times
    matching_files_dir1 = [
        os.path.join(dir1, filename)
        for filename in files1
        if re.search(pattern, filename)
        and re.search(pattern, filename).group(0) in common_dates
    ]
    matching_files_dir2 = [
        os.path.join(dir2, filename)
        for filename in files2
        if re.search(pattern, filename)
        and re.search(pattern, filename).group(0) in common_dates
    ]

    return matching_files_dir1, matching_files_dir2

def get_files(RGB_dir_path: str, img_dir_path: str) -> np.ndarray:
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    Returns:
        np.ndarray: A matrix of matching files, shape (bands, number of samples).
    """
    if not os.path.exists(RGB_dir_path):
        raise FileNotFoundError(f"{RGB_dir_path} not found")
    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(f"{img_dir_path} not found")

    # get the dates in both directories
    common_dates = matching_datetimes_files(RGB_dir_path, img_dir_path)
    # get the full paths to the dates that exist in each directory
    matching_files_RGB_dir, matching_files_img_dir = get_full_paths(
        RGB_dir_path, img_dir_path, common_dates
    )
    # the order must be RGB dir then not RGB dir for other code to work
    # matching_files = sorted(matching_files_RGB_dir) + sorted(matching_files_img_dir)
    files = []
    files.append(sorted(matching_files_RGB_dir))
    files.append(sorted(matching_files_img_dir))
    # creates matrix:  matrix: RGB x number of samples in img_dir_path
    matching_files = np.vstack(files).T
    return matching_files

def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> None:
    """Converts two directories of RGB and (NIR/SWIR) imagery to (NDWI/MNDWI) imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix: RGB files x NIR files
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path = os.path.join(output_path, output_type.upper())

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.nan
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = (green_band-infrared) / (green_band + infrared )
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        # skimage.io.imsave(
        #     output_path + os.sep + new_filename,
        #     output_img.astype("uint8"),
        #     check_contrast=False,
        #     quality=100,
        # )
        # save output_img (MNDWI/NDWI) as .jpg in output directory
        imageio.imwrite(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            quality=100
        )

    return output_path

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


# give it a config.json file and it will do the rest

config_json_path = r'C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime01-10-25__03_58_44\config.json'

# read the config
import json
with open(config_json_path) as json_file:
    config = json.load(json_file)
    roi_ids = config['roi_ids']

    # read the settings needed to create jpgs
    settings = config['settings']
    cloud_threshold = settings.get('cloud_thresh', 0.5)
    cloud_mask_issue = settings.get('cloud_mask_issue', False)
    apply_cloud_mask = settings.get('apply_cloud_mask', True)


    # load the inputs for each ROI 
    for roi_id in roi_ids:
        inputs = config[roi_id]
        print(inputs)
        satlist = inputs['sat_list']
        for sat in satlist:
            jpg_files_folder = os.path.join(inputs["filepath"],inputs["sitename"], "jpg_files", "preprocessed")
            RGB_path = os.path.join(jpg_files_folder, "RGB")
            print(sat)
            tif_paths = SDS_tools.get_filepath(inputs, sat)

            # get a list of all the ms files for the given satellite
            ms_folder = os.path.join(inputs["filepath"],inputs["sitename"], sat, "ms")
            if not os.path.exists(ms_folder):
                raise Exception(f"Folder {ms_folder} does not exist")

            ms_files = glob.glob(os.path.join(ms_folder, "*.tif"))
            for ms_file in ms_files:
                print(ms_file)
                save_pansharpened_jpg(
                    filename=os.path.basename(ms_file),
                    tif_paths=tif_paths,
                    satname=sat,
                    sitename=inputs["sitename"],
                    cloud_thresh=cloud_threshold,
                    cloud_mask_issue=cloud_mask_issue,
                    filepath_data=inputs["filepath"],
                    collection=inputs["landsat_collection"],
                    apply_cloud_mask=apply_cloud_mask,
                )
            # Recreate the MNDWI and NDWI folder with the new pansharpened images if they exist
            NIR_path = os.path.join(jpg_files_folder, "NIR")
            NDWI_path = RGB_to_infrared(RGB_path, NIR_path, jpg_files_folder, "NDWI")
            SWIR_path = os.path.join(jpg_files_folder, "SWIR")
            MNDWI_path = RGB_to_infrared(RGB_path, SWIR_path, jpg_files_folder, "MNDWI")