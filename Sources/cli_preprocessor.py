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
import shutil
from typing import Iterator

import utils


def is_valid_dir(test_dir: os.DirEntry) -> bool:
    """
    Only some directories are valid, the ones that start with FHBO
    (which is our original Head and Neck dataset)
    and then only the only ones that have two subdirs:
     - The scan directory
     - The mask directory
    :param test_dir: Directory to be tested
    :return: True or False depending on the folder conditions
    """
    if not test_dir.is_dir():
        return False
    # Check that the name is the one that we need, only a part of the dataset is valid, the ones starting with FHBO
    name = str(test_dir.name)

    if not name.startswith("FHBO"):
        return False

    # Check if it has two sub dirs that start with the same name
    sub_dirs = list(filter(lambda x: x.is_dir() and name in x.name, os.scandir(test_dir.path)))
    return len(sub_dirs) >= 2


def get_folders(input_dir: str) -> Iterator[os.DirEntry]:
    """
    Get all the folders that contain all the data, each folder is a different patient
    :param input_dir: Directory to where search the folders
    :return:
    """
    return filter(is_valid_dir, os.scandir(input_dir))


def main(arguments):
    """
    Main function. This function is called if this file is executed
    as a script
    :param arguments: Arguments provided by argparse.ArgumentParser 
    """
    datasets_dir = os.path.expandvars(arguments.datasets_dir)
    input_dir = os.path.join(datasets_dir, arguments.input_dir)
    output_dir = os.path.join(datasets_dir, arguments.output_dir)

    # Check if folder exists
    if not all(map(os.path.exists, [input_dir, datasets_dir])):
        print("ERROR: Not all the folder exist")
        exit(1)

    print("Datasets: " + datasets_dir)
    print("Input: " + input_dir)
    print("Output: " + output_dir)

    if arguments.overwrite and os.path.exists(output_dir):
        # To overwrite remove the path then it will be created again
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = list(get_folders(input_dir))
    print("{} valid dirs have been found".format(len(dirs)))

    normalizer = utils.ScanNormalizer(dirs, output_dir, os.path.join(input_dir, "clinical_info.csv"))
    normalizer.process_data()


if __name__ == "__main__":
    # We are being called and not as a module
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pre-process all the images by normalizing, "
                                                 "slicing and rotating them")
    parser.add_argument(
        "--datasets_dir",
        help="Directory where all the datasets can be found",
        default="$HOME/Documents/Datasets"
    )
    parser.add_argument(
        "--input_dir",
        help="Directory inside datasets_dir where the desired dataset is found",
        default="HNK"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory inside datasets_dir where the processed images will be saved ",
        default="HNK_processed"
    )
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

