import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit the data with a Tensorflow model")
    parser.add_argument(
        "--datasets_dir",
        help="Directory where all the datasets can be found",
        default="$HOME/Documents/Datasets"
    )
    parser.add_argument(
        "--input_dir",
        help="Directory inside datasets_dir where the desired dataset is found",
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