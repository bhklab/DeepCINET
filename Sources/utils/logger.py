import logging
import os
import sys

import settings

if not os.path.exists(os.getenv("LOG_DIR", None)):
    os.makedirs(os.getenv("LOG_DIR"))

loggers = {}


def get_logger(name: str) -> logging.Logger:

    if name in loggers:
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(os.getenv("LOG_DIR"), name + ".log"))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    loggers[name] = logger
    return logger
