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
    VolumeOnlySiamese,\
    ClinicalOnlySiamese,\
    ClinicalOnlySiamese2,\
    ClinicalOnlySiamese3,\
    ClinicalVolumeSiamese, \
    ClinicalVolumeSiamese2, \
    ClinicalVolumeSiamese3,\
    ScalarOnlyInceptionSiamese

from .basics import BasicModel, BasicSiamese, BasicImageSiamese
