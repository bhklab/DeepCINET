#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd

from pytorch_src.models.lightning import ImageSiamese
from pytorch_src.data.datareader import PairProcessor
import numpy as np

import torch
import torch.nn as nn

import pytorch_src.config as config

from pytorch_lightning import Trainer

################
#     MAIN     #
################
def deepCinet(num_epochs: int = 1,
              batch_size: int = 20,
              radiomics_path: str = "",
              clinical_path: str = "",
              image_path: str = ""):
    pairProcessor = PairProcessor(clinical_path)
    train_ids, test_ids = pairProcessor.train_test_split(
        test_ratio = 0.2,
        split_model = PairProcessor.TIME_SPLIT,
        random_seed = 520)
    siamese_model = ImageSiamese(train_ids, test_ids, batch_size,
                                 image_path, radiomics_path, clinical_path,
                                 use_images=config.USE_IMAGES,
                                 use_radiomics=config.USE_RADIOMICS)
    trainer = Trainer(min_epochs = num_epochs, max_epochs=num_epochs, gpus=1)
    trainer.fit(siamese_model)

def main() -> None:
    """
    Main function
    :param args: Command Line Arguments
    """
    batch_size = config.BATCH_SIZE
    num_epochs = config.EPOCHS
    radiomics_path = config.RADIOMICS_PATH
    clinical_path = config.CLINICAL_PATH
    image_path = config.IMAGE_PATH
    deepCinet(num_epochs=num_epochs,
              batch_size=batch_size,
              radiomics_path=radiomics_path,
              clinical_path=clinical_path,
              image_path=image_path)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\r----------------------------------")
        logger.info("\rStopping due to keyboard interrupt")
        logger.info("\rTHANKS FOR THE RIDE 😀")
