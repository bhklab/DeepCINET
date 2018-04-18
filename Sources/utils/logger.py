import logging
import os
import sys

import settings

if not os.path.exists(settings.LOG_DIR):
    os.makedirs(settings.LOG_DIR)

loggers = {}


def get_logger(name: str) -> logging.Logger:

    if name in loggers:
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(settings.LOG_DIR, name + ".log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    if len(name) > 0:
        logger.addHandler(ch)
    logger.debug("---- Logger Initiated ----")

    loggers[name] = logger
    return logger
