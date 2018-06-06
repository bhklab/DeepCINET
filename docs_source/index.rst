.. Sources documentation master file, created by
   sphinx-quickstart on Thu Apr 12 12:43:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CNNSurv's documentation!
===================================

This repository contains two scripts required to pre-process and to train a model that uses a deep
neural network to fit the survival data.

To pre-process the data use the script ``Sources/preprocess.py``. A ``.env`` file must
have already been defined on the root file of the repository. An example can be found on ``Sources/env_default.txt``.

If you want to train a deep model you can use the training script:
``Sources/train.py``. All the training options can be seen by passing the ``--help`` flag.
To use the training script the data must have been pre-processed before.

More information in how to configure and run the repository can be round in the ``README.md`` file.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
