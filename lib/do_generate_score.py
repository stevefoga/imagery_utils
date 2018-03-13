"""
do_generate_score.py

Purpose: wrapper script for generate_score.py. Use when an orthorectifed image exists, but does not have .score, or if
    existing .score is incorrect.

Created: 13 March 2018

Python version: 2.7.13
Requires GDAL >= 2.1
"""
import logging
import os
import glob
import generate_score

# create logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)


def main(src, pct_thresh, overwrite_invalid, overwrite_all, dryrun, is_tiled=True, water_mask=None, tile_path=None):

    # if dir, get files, else assume single file was supplied
    if os.path.isdir(src):
        fns_in = glob.glob(src + "*.tif")
    else:
        fns_in = [src]

    for fn_in in fns_in:

        # if all files are not to be modified, check for invalid values
        if not overwrite_all:
            score_path = fn_in.replace('.tif', '.score')

            if os.path.isfile(score_path) and overwrite_invalid:
                # check if score is invalid
                with open(score_path, 'r') as f:
                    score = float(f.read())
                if score != -9999:
                    logger.debug('score is valid, excluding {0}'.format(fn_in))
                    continue

            elif os.path.isfile(score_path) and not overwrite_invalid:
                logger.debug('excluding {0}'.format(fn_in))
                continue

        logger.info('.score being calculated for {0}'.format(fn_in))

        if not dryrun:
            generate_score.generate_score(fn_in, pct_thresh=pct_thresh, water_mask=water_mask, tile_path=tile_path,
                                          is_tiled=is_tiled)

    if dryrun:
        logger.info('--dryrun used, no .score files generated')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create mosaic correlation score from existing ortho images.")

    parser.add_argument("src", help="Input file or directory (.tif only)", type=str)
    parser.add_argument("--pct-thresh", help="Percent nodata threshold to accept (default=0.95)", type=float,
                        default=0.95, required=False)
    parser.add_argument("--water-mask", help="Path to water mask (default=None)", required=False)
    parser.add_argument("--tile-path", help="Path to tile mosaic(s) (required if not LIMA)", required=False)
    parser.add_argument("--overwrite-invalid", help="Overwrite invalid (-9999) scores", action="store_true",
                        required=False)
    parser.add_argument("--overwrite-all", help="Overwrite all .score files", action="store_true", required=False)
    parser.add_argument("--is-tiled", help="True if mosaic split into multiple files, False if single mosaic image",
                        action="store_true")

    parser.add_argument("--dryrun", help="Print cmd, do not submit job", dest="dryrun", action="store_true",
                        required=False)

    args = parser.parse_args()

    main(**vars(args))
