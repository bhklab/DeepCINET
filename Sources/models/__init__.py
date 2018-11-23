"""
The Train package contains different classes designed to be trained to fit data

.. inheritance-diagram:: models.basics models.siameses
   :parts: 1

"""

from .siameses import \
    SimpleImageSiamese, \
    ImageScalarSiamese, \
    ScalarOnlySiamese, \
    ScalarOnlyDropoutSiamese, \
    ImageSiamese, \
    ResidualImageScalarSiamese, \
    VolumeOnlySiamese
from .basics import BasicModel, BasicSiamese, BasicImageSiamese
