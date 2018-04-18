#! /usr/bin/env python3

"""
This is a script to pre-process the scanning data, it assumes
that the input is multiple 2D slice images and creates a 3D image
from the input.

The output is normalized (mean = 0 and std = 1) and only the slice,
defined by the mask bounding box, is extracted and then resized to
a 64x64x64 3D image.

To do so it uses the utils.ScanNormalizer class so it this file only
parses the arguments and then calls the previous class.
"""

import os
import argparse

import data.image_data
import settings
import multiprocessing

import data


def main(arguments):
    """
    Main function. This function is called if this file is executed
    as a script
    :param arguments: Arguments provided by argparse.ArgumentParser 
    """
    input_dir = settings.DATA_RAW
    output_dir = settings.DATA_PROCESSED

    # Check if folder exists
    if not os.path.exists(input_dir):
        print("ERROR: The {} folder does not exist".format(input_dir))
        exit(1)

    print("We have {} cores".format(multiprocessing.cpu_count()))

    print("Root dir: " + settings.APP_ROOT)
    print("Overwrite: " + str(arguments.overwrite))
    print("Data raw dir: " + input_dir)
    print("Data preprocessed dir: " + output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pre_processed = data.image_data.PreProcessedData()
    pre_processed.store(overwrite=arguments.overwrite)


if __name__ == "__main__":
    # We are being called and not as a module
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pre-process all the images by normalizing, "
                                                 "slicing and rotating them")
    parser.add_argument(
        "--overwrite",
        help="Overwrite destination directory",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        # This avoids an ugly print on the terminal
        print()
        print("----------------------------------")
        print("Stopping due to keyboard interrupt")
        print("THANKS FOR THE RIDE 😀")
