import logging
import logging.handlers
import os
import sys

import tensorflow_src.config as settings


file_formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
console_formatter = logging.Formatter("(%(name)s): %(message)s")


def init_logger(app_name: str, directory: str = settings.SUMMARIES_DIR) -> logging.Logger:
    """
    Initialize script logging functionality. This function should be called only once in the script's execution.
    All the logs are printed to the console and saved into disk with the name ``<app_name>.log``. The save location
    is defined by the environment variable ``LOG_DIR``.

    :param app_name: Name of the script that it's being executed. The saved logfile will have as name
                     ``<app_name>.log``.
    :param directory: Path for the directory where the log files should be saved
    :return: Initialized logger
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    fh = logging.FileHandler(os.path.join(directory, app_name + ".log"), encoding="utf-8")
    fh.setLevel(settings.LOG_LEVEL_FILE)
    fh.setFormatter(file_formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(settings.LOG_LEVEL_CONSOLE)
    ch.setFormatter(console_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])
    logger = logging.getLogger('')
    logger.addHandler(logging.handlers.SocketHandler('127.0.0.1', 19996))

    return logger

