#! /usr/bin/env python3

"""
This file is used to normalize and scale the data
"""

import os
import argparse
import shutil

import utils


def is_valid_dir(test_dir: os.DirEntry) -> bool:
    if not test_dir.is_dir():
        return False
    # Check that the name is the one that we need, only a part of the dataset is valid
    name = str(test_dir.name)
    if not name.startswith("FHBO"):
        return False

    # Check if it has two sub dirs
    sub_dirs = list(filter(lambda x: x.is_dir() and name in x.name, os.scandir(test_dir.path)))
    return len(sub_dirs) >= 2


def get_folders(input_dir):
    return filter(is_valid_dir, os.scandir(input_dir))


def main(arguments: argparse.Namespace):
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

    # Remove the output and create it again to make sure it does not have anything
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    dirs = list(get_folders(input_dir))
    print("{} valid dirs have been found".format(len(dirs)))

    normalizer = utils.ScanNormalizer(dirs, output_dir)
    normalizer.process_images()


if __name__ == "__main__":
    # We are being called and not as a module
    # Parse arguments
    parser = argparse.ArgumentParser(description="Normalize and slice images")
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

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print()
        print("----------------------------------")
        print("Stopping due to keyboard interrupt")
        print("THANKS FOR THE RIDE 😀")

