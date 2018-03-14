"""
generate_score.py

Purpose: generate texture score(s) using tile(s) from basemap.

Created: 11 January 2018

Python version: 2.7.13
Requires GDAL >= 2.1
"""
import logging
import os
import platform
import sys
import math
import time
import subprocess
import uuid
import numpy as np
from osgeo import gdal, osr, gdalconst
from skimage.measure import compare_ssim as ssim

# TODO: find way to incorporate MODIS pole hole filler

# create logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)


def read_image(img_in):
    """
    Read image as GDAL object.

    :param img_in: Path to image.
    :return: <osgeo.gdal.Dataset>
    """
    logger.info("Reading image {0}".format(img_in))
    logger.debug("img_in type: {0}".format(type(img_in)))

    img = gdal.Open(img_in, gdal.GA_ReadOnly)

    return img


def read_band(gdataset_in, band_num=1):
    """
    Read specific band from raster.

    :param gdataset_in: <osgeo.gdal.Dataset>
    :param band_num: <int> specific band to be read.
    :return:
    """
    band = gdataset_in.GetRasterBand(band_num)

    return band


def img_to_array(gdal_ds, band_num=1):
    """
    Read GDAL object as numpy array. Defaults to band 1, which is usually red. If None, reads all bands and returns
    their average.

    :param gdal_ds: <osgeo.gdal.Dataset>
    :param band_num: <int> specific band to be read. If 0, read multiband array.
    :return: <numpy.nd.array> array of uint8 values
    """
    if band_num == 0:
        arr = gdal_ds.ReadAsArray()
        arr_out = np.round(np.mean(arr, axis=0)).astype('uint8')
    else:
        arr_out = np.array(read_band(gdal_ds, band_num).ReadAsArray())

    return arr_out


def get_bbox(gdal_object_in):
    """

    :param gdal_object_in: <osgeo.gdal.Dataset>
    :return: <list>
    """
    ulx, xres, xskew, uly, yskew, yres = gdal_object_in.GetGeoTransform()
    lrx = ulx + (gdal_object_in.RasterXSize * xres)
    lry = uly + (gdal_object_in.RasterYSize * yres)

    return [ulx, uly, lrx, lry]


def sint2str(int_value, pad_value=7):
    """
    Turn integer into string and pad with zeros; if int is positive, return with '+' at front. Assumes 0 is positive.

    :param int_value: <int> integer value
    :param pad_value: <int> number of digits to consider for padding (default = 7, which is what LIMA uses)
    :return: <str>
    """
    if int_value >= 0:
        str_value = '+{}'.format(str(int_value).zfill(pad_value))
    else:
        str_value = '-{}'.format(str(-1*int_value).zfill(pad_value))

    return str_value


def get_lima_tile(scene_in):
    """
    Build the name of the tile(s) based upon the extent of the input scene. Assumes LIMA RGBREF tiles as input.

    :param scene_in: <str> path to image file

    :return: <list> full path and name of LIMA tile(s)
    """
    # TODO: add path for MODIS pole hole filler
    if platform.system() == 'Windows':
        lima_base = r'V:\pgc\data\sat\prod\lima\rgbref'
    else:
        lima_base = '/mnt/pgc/data/sat/prod/lima/rgbref'

    # get boundaries of scene
    scene_bounds = get_bbox(read_image(scene_in))

    def determine_coord(bbox, interval, grid_crd):
        """
        Determine bbox extents within a specific interval.

        :param bbox: <list> list of coordinates, as [ulx, uly, lrx, lry]
        :param interval: <int> grid spacing interval
        :param grid_crd: <str> define whether 'x' or 'y' boundaries are to be checked
        :return: <list> ints of outer bounding coordinates
        """
        if grid_crd.lower() == "x":
            ld = bbox[0]
            ud = bbox[2]
        elif grid_crd.lower() == "y":
            ld = bbox[3]
            ud = bbox[1]
        else:
            logger.error("Incorrect grid coordinate {0} supplied, should be 'x' or 'y'".format(grid_crd))
            sys.exit(-1)

        ll = int(interval * math.floor(ld / interval))
        uu = int(interval * round(ud / interval))

        out = [sint2str(ll)]

        if ll != uu:  # assuming scene spans multiple tiles
            out.append(sint2str(uu))

        return out

    x_out = determine_coord(scene_bounds, 150000, 'x')
    y_out = determine_coord(scene_bounds, 150000, 'y')

    # build tile name & verify existence
    tile_name = []
    for xx in x_out:
        for yy in y_out:
            tile_name.append(os.path.join(lima_base, "RGBREF_x{0}y{1}.tif".format(xx, yy)))
            if not os.path.isfile(tile_name[-1]):
                # attempt to find under alternative naming scheme
                logger.warning("Tile with prefix 'RGBREF*' not available, trying prefix 'rgbref*'")
                tile_name[-1] = os.path.join(lima_base, "rgbref_x{0}y{1}.tif".format(xx, yy))

                if not os.path.isfile(tile_name[-1]):
                    logger.warning("Unable to find tile {0}, setting tile path to None".format(tile_name[-1]))
                    tile_name[-1] = None
                    # logger.error("Unable to find tile {0}".format(tile_name[-1]))
                    # sys.exit(-1)

    logger.debug("Input LIMA tile(s): {0}".format(tile_name))

    return tile_name


def merge(img_list, fn_out, nd_value=0):
    """
    Merge rasters into single raster.

    :param img_list: <list> Path to rasters.
    :param fn_out: <str> Path to output directory.
    :param nd_value: <int> (default = 0)
    :return:
    """
    logger.info("Merging {0} tiles ({1}) to file {2}".format(len(img_list), img_list, fn_out))

    subprocess.call(subprocess.list2cmdline(['gdal_merge.py', '-o', fn_out, '-of', 'GTiff', '-n', str(nd_value)] +
                                            img_list), shell=True)

    if not os.path.isfile(fn_out):
        logger.error("Output file {0} does not exist".format(fn_out))
        logger.error("Command used: {0}".format(['gdal_merge.py', '-o', fn_out, '-of', 'GTiff', '-n', str(nd_value),
                                                 img_list[0], img_list[1]]))
        sys.exit(-1)


def build_lima_tiles(scene_in, dir_out):
    """
    If > 1 LIMA tile, merge together, else return single path.

    :param scene_in: <str> Path to input scene (extent used to determine LIMA tile(s) to be used)
    :param dir_out: <str> Path to output directory
    :return: <list>,<bool>,<str> [Tile path]; True if merge, False if no merge; unique_id
    """
    # build tile filename(s)
    tiles = get_lima_tile(scene_in)

    # remove None values, if any
    tiles = [i for i in tiles if i is not None]

    # generate unique id for ouptut tile
    unique_id = str(uuid.uuid4())

    # if more than one tile, build mosaic
    if len(tiles) > 1:
        # create new tile
        mosaic_path = os.path.join(dir_out, unique_id + '.tif')
        merge(tiles, mosaic_path)

        logger.debug("Output mosaic path: {0}".format(mosaic_path))

        return [mosaic_path], True, unique_id

    elif len(tiles) == 0:
        # return None
        return None, False, unique_id

    else:
        # return tile as-is
        return tiles, False, unique_id


def write_score(scene_in, final_score):
    """
    Write score to file as plain text.

    :param scene_in: <str> Path to input scene (will have extension stripped for final .score file>
    :param final_score: <int or float> Score to be written to file
    :return:
    """
    # write score to .score file
    score_out = os.path.splitext(scene_in)[0] + '.score'
    with open(score_out, 'w') as f:
        f.write('{0}'.format(final_score))

    logger.debug('ssim score {0} written to {1}'.format(final_score, score_out))


def generate_score(scene, pct_thresh=0.95, water_mask=None, tile_path=None, not_tiled=False):
    """
    Generate correlation score between input scene and LIMA tile; write .score file to dir_out.

    :param scene: <str> Path to input scene.
    :param pct_thresh: <int or float> Percent of NoData allowed between scene and tile (range=0.0-1.0; default=0.95)
    :param water_mask: <str> Path to water mask (default=None)
    :param tile_path: <str> Path to GeoTIFF tile(s), not necessary if LIMA
    :param not_tiled: <bool> False if mosaic is split up between files; True if mosaic in single image
    :return:
    """
    if water_mask:
        if not os.path.isfile(water_mask):
            logger.error("water_mask path {0} is not a valid file".format(water_mask))
            sys.exit(-1)

    # open hires scene
    scene_gdal = read_image(scene)

    # identify LIMA tile(s) (merge if necessary)
    if not tile_path and not not_tiled:
        tile_path, merged_tiles, uid = build_lima_tiles(scene, os.path.dirname(scene))

    # if a single, non-LIMA mosaic file is used
    if tile_path and not_tiled:
        tile_path = [tile_path]
        uid = str(uuid.uuid4())

    # if tiled, non-LIMA mosaic is used
    if tile_path and not not_tiled:
        logger.error("Non-LIMA mosaic tiles are currently not supported. The get_lima_tile() function must be modified "
                     "to accept a new tile naming convention and dimensions.")
        sys.exit(-1)

    # if no valid tile, set score to -9999 and continue
    if tile_path is None:
        logger.warning("No tile_path provided, returning score of -9999 for scene {0}".format(scene))
        write_score(scene, -9999)
        return

    # open tile as GDAL object
    tile_gdal = read_image(tile_path[0])

    # get spatial resolution
    scene_gt = scene_gdal.GetGeoTransform()
    scene_res = scene_gt[1]

    tile_gt = tile_gdal.GetGeoTransform()
    tile_res = tile_gt[1]

    # calculate scale factor (resample scene to tile)
    scale_factor = float(tile_res) / scene_res

    # create file names for resampled scene and clipped tile
    scene_resample = os.path.join(os.path.dirname(scene), os.path.splitext(os.path.basename(scene))[0] +
                                  '_{0}m-resample.tif'.format(int(tile_res)))
    tile_clip = os.path.join(os.path.dirname(tile_path[0]),
                             os.path.splitext(os.path.basename(tile_path[0]))[0] + '_clip.tif')
    if water_mask:
        if not uid:
            uid = str(uuid.uuid4())
        water_mask_gdal = read_image(water_mask)
        water_gt = water_mask_gdal.GetGeoTransform()
        water_res = water_gt[1]

        water_mask_clip = os.path.join(os.path.dirname(scene), os.path.splitext(os.path.basename(water_mask))[0] +
                                       '{0}_{1}m-wmask_clip.tif'.format(uid, int(tile_res)))

    # get projection
    scene_proj = scene_gdal.GetProjection()
    tile_proj = tile_gdal.GetProjection()

    # make sure the projections are the same using IsSame method from SpatialReference class
    if not osr.SpatialReference(scene_proj).IsSame(osr.SpatialReference(tile_proj)):
        logger.warning("Scene {0} and tile {1} do not have equivalent projections; the tile will be warped to the "
                       "scene's SRS.".format(os.path.basename(scene), os.path.basename(tile_path[0])))

    # edit GeoTransform for new output band
    output_gt = (scene_gt[0], scene_gt[1] * scale_factor, scene_gt[2], scene_gt[3], scene_gt[4],
                 scene_gt[5] * scale_factor)

    # create output image for resampling
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(scene_resample,
                           int(np.ceil(scene_gdal.RasterXSize / scale_factor)),
                           int(np.ceil(scene_gdal.RasterYSize / scale_factor)),
                           1,
                           gdal.GDT_Byte)
    output.SetGeoTransform(output_gt)
    output.SetProjection(scene_proj)
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(0)

    # resample image
    tr0 = time.time()
    logger.debug("Resampling {0} to file {1}...".format(os.path.basename(scene), os.path.basename(scene_resample)))
    gdal.ReprojectImage(scene_gdal, output, scene_proj, scene_proj, gdalconst.GRA_Cubic)
    del output

    total = time.time() - tr0
    logger.debug("Total resampling time: {0} minutes".format(round(total/60, 3)))

    # get extent of resampled tile
    scene_res_gdal = read_image(scene_resample)

    ulx, xres, xskew, uly, yskew, yres = scene_res_gdal.GetGeoTransform()
    lrx = ulx + (scene_res_gdal.RasterXSize * xres)
    lry = uly + (scene_res_gdal.RasterYSize * yres)

    # clip tile
    gdal.Translate(tile_clip, tile_gdal, projWin=[ulx, uly, lrx, lry])

    # resample and clip water mask
    if water_mask:
        # calculate scale factor (resample water_mask to tile)
        scale_factor = float(tile_res) / water_res

        # edit GeoTransform for new output band
        output_gt = (water_gt[0], water_gt[1] * scale_factor, water_gt[2], water_gt[3], water_gt[4],
                     water_gt[5] * scale_factor)

        # create output image for resampling
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(scene_resample,
                               int(np.ceil(water_mask_gdal.RasterXSize / scale_factor)),
                               int(np.ceil(water_mask_gdal.RasterYSize / scale_factor)),
                               1,
                               gdal.GDT_Byte)
        output.SetGeoTransform(output_gt)
        output.SetProjection(scene_proj)
        output_band = output.GetRasterBand(1)
        output_band.SetNoDataValue(255)
        gdal.Translate(water_mask_clip, water_mask_gdal, projWin=[ulx, uly, lrx, lry])

    # read images as arrays
    scene_res_arr = img_to_array(scene_res_gdal)

    tile_clip_gdal = read_image(tile_clip)
    num_bands = tile_clip_gdal.RasterCount
    if num_bands > 1:
        logger.info("{0} bands in tile_clip_gdal, will use mean of all bands".format(num_bands))
        tile_clip_arr = img_to_array(tile_clip_gdal, band_num=0)
    else:
        logger.debug("tile_clip_gdal has one band, reading 'band 1'")
        tile_clip_arr = img_to_array(tile_clip_gdal, band_num=1)

    # mask arrays to common extent (both use NoData value of 0)
    clip_mask = np.ma.masked_values(scene_res_arr, 0)
    tile_mask = np.ma.masked_values(tile_clip_arr, 0)

    if water_mask:
        water_clip_arr = img_to_array(water_mask_clip)
        water_mask_mask = np.ma.masked_values(water_clip_arr, 255)

        # apply 'water_mask' to clip_mask and tile_mask
        clip_mask = np.ma.masked_values(water_mask_mask.mask, clip_mask)
        tile_mask = np.ma.masked_values(water_mask_mask.mask, tile_mask)

    # make sure tile NoData does not exceed a certain percentage, otherwise the metric isn't accurate
    pct_data = np.float(np.sum(tile_mask.mask)) / np.product(tile_mask.shape)

    # check for image extent equivalency
    c_size = clip_mask.shape
    t_size = tile_mask.shape
    logger.debug(c_size)
    logger.debug(t_size)
    if c_size != t_size:
        logger.error("arrays are not same dimensions. clip_mask: {0}; tile_mask {1}".format(c_size, t_size))
        sys.exit(-1)

    # TODO: add check for MODIS pole hole (<= -82 deg. lat.)

    if pct_data >= pct_thresh:
        logger.warning("The tile clipped to the scene extent contains {0}% NoData, which exceeds the threshold of {1}%;"
                       " a score of -9999 is being returned.".format(round(pct_data, 4) * 100,
                                                                     round(pct_thresh, 4) * 100))
        s_gws9 = -9999

    else:
        # get ssim score
        s_gws9 = ssim(clip_mask, tile_mask, gaussian_weights=True, sigma=9)

    # write score to .score file
    write_score(scene, s_gws9)

    # clean up intermediate files
    del scene_res_gdal
    os.remove(scene_resample)
    del tile_clip_gdal
    os.remove(tile_clip)

    # clean up merged tile
    if merged_tiles:
        del tile_gdal
        logger.debug("Removing merged tile {0}".format(tile_path[0]))
        os.remove(tile_path[0])

    if water_mask:
        del water_mask_gdal
        logger.debug("Removing clipped water mask tile {0}".format(water_mask_clip))
        os.remove(water_mask_clip)
