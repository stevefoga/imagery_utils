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
import sys
import glob
import generate_score
import taskhandler

# create logger
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)


def main(src, pct_thresh, overwrite_invalid, overwrite_all, not_tiled, dryrun, water_mask=None, tile_path=None):

    logfile = os.path.join(os.path.dirname(src), 'generate_score.log')
    lfh = logging.FileHandler(logfile)
    lfh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s- %(message)s', '%m-%d-%Y %H:%M:%S')
    lfh.setFormatter(formatter)
    logger.addHandler(lfh)

    lsh = logging.StreamHandler()
    lsh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s- %(message)s','%m-%d-%Y %H:%M:%S')
    lsh.setFormatter(formatter)
    logger.addHandler(lsh)

    # if dir, get files, else assume single file was supplied
    if os.path.isdir(src):
        src = os.path.join(src, '')
        fns_in = glob.glob(src + "*.tif")
    else:
        fns_in = [src]

    for fn_in in fns_in:

        # if all files are not to be modified, check for invalid values
        if not overwrite_all:
            score_path = fn_in.replace('.tif', '.score')
            logger.debug("score_path: {0}".format(score_path))

            if os.path.isfile(score_path) and overwrite_invalid:
                # check if score is invalid
                with open(score_path, 'r') as f:
                    score = float(f.read())

                logger.debug("score: {0}".format(score))
                if score != -9999:
                    logger.debug('score is valid, excluding {0}'.format(fn_in))
                    continue

            elif os.path.isfile(score_path) and not overwrite_invalid:
                logger.debug('excluding {0}'.format(fn_in))
                continue

        logger.info('.score being calculated for {0}'.format(fn_in))

        if not dryrun:
            generate_score.generate_score(fn_in, pct_thresh=pct_thresh, water_mask=water_mask, tile_path=tile_path,
                                          not_tiled=not_tiled)

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
    parser.add_argument("--not-tiled", help="Use if mosaic is single image (as opposed to being split into chunks)",
                        action="store_true")
    parser.add_argument("--parallel-processes", help="number of processes to spawn (default=1)", default=1)

    parser.add_argument("--dryrun", help="Print cmd, do not submit job", dest="dryrun", action="store_true",
                        required=False)

    args = parser.parse_args()

    if args.parallel_processes > 1:
        # build script path
        scriptpath = os.path.abspath(sys.argv[0])
        gen_score_script = os.path.join(os.path.dirname(scriptpath), 'generate_score.py')

        # convert args
        pos_arg_keys = ["src"]
        arg_keys_to_remove = (
            "pct_thresh",
            "water_mask",
            "tile_path",
            "overwrite_invalid",
            "overwrite_all",
            "not_tiled",
            "parallel_processes",
            "dryrun"
        )
        gscore_arg_str = taskhandler.convert_optional_args_to_string(args, pos_arg_keys, arg_keys_to_remove)

        # get all files
        if os.path.isdir(args.src):
            src = os.path.join(args.src, '')
            fns_in = glob.glob(args.src + "*.tif")
        else:
            fns_in = [args.src]

        task_queue = []
        it = 0
        for fn in fns_in:
            it += 1
            task = taskhandler.Task(
                fn,
                'Gscore{:04g}'.format(it),
                'python',
                '{} {} {}'.format(gen_score_script, fn, args.pct_thresh)
            )

            task_queue.append(task)

        try:
            task_handler = taskhandler.ParallelTaskHandler(args.parallel_processes)
        except RuntimeError as e:
            logger.error(e)
        else:
            logger.info("Number of child processes to spawn: {0}".format(task_handler.num_processes))
            task_handler.run_tasks(task_queue)

    else:
        main(**vars(args))
