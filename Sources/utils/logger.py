import logging
import logging.handlers
import os
import sys

import settings

if not os.path.exists(settings.LOG_DIR):
    os.makedirs(settings.LOG_DIR)

loggers = {}
file_formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
console_formatter = logging.Formatter("(%(name)s): %(message)s")


def get_logger(name: str) -> logging.Logger:
    """
    Obtain a logger by its name. If the logger does not exists, a new one will be created instead.
    By default all loggers are initialized and set with ``logging.DEBUG`` level.

    :param name: Logger name
    :return: Created logger
    """
    if name in loggers:
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    loggers[name] = logger
    logger.debug("---- Logger Initiated ----")
    return logger


def init_logger(app_name: str) -> logging.Logger:
    """
    Initialize script logging functionality. This function should be called only once in the script's execution.
    All the logs are printed to the console and saved into disk with the name ``<app_name>.log``. The save location
    is defined by the environment variable ``LOG_DIR``.

    :param app_name: Name of the script that it's being executed. The saved logfile will have as name
                     ``<app_name>.log``.
    :return: Initialized logger
    """
    logger = get_logger('')

    fh = logging.FileHandler(os.path.join(settings.LOG_DIR, app_name + ".log"), encoding="utf-8")
    fh.setLevel(settings.LOG_LEVEL_FILE)
    fh.setFormatter(file_formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(settings.LOG_LEVEL_CONSOLE)
    ch.setFormatter(console_formatter)

    logger.addHandler(logging.handlers.SocketHandler('127.0.0.1', 19996))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

