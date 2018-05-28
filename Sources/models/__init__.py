"""
The Train package contains different classes designed to be trained to fit data
"""

from .siameses import \
    SimpleImageSiamese, \
    ImageScalarSiamese, \
    ScalarOnlySiamese, \
    ScalarOnlyDropoutSiamese, \
    ImageSiamese, \
    ResidualImageScalarSiamese, \
    ScalarOnlyDropoutSiameseDistance, \
    VolumeOnlySiamese
from .basics import BasicModel, BasicSiamese, BasicImageSiamese
