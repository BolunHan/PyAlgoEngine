__version__ = "0.4.0"

import logging
import os
import sys
import time
import traceback

from PyQuantKit import ColoredFormatter

LOGGER: logging.Logger | None = None
LOG_LEVEL = logging.INFO

if 'ALGO_DIR' in os.environ:
    WORKING_DIRECTORY = os.path.realpath(os.environ['ALGO_DIR'])
else:
    WORKING_DIRECTORY = str(os.getcwd())


def get_logger(**kwargs) -> logging.Logger:
    level = kwargs.get('level', LOG_LEVEL)
    stream_io = kwargs.get('stream_io', sys.stdout)
    formatter = kwargs.get('formatter', ColoredFormatter())
    global LOGGER

    if LOGGER is not None:
        return LOGGER

    LOGGER = logging.getLogger('PyAlgoEngine')
    LOGGER.setLevel(level)
    logging.Formatter.converter = time.gmtime

    if stream_io:
        have_handler = False
        for handler in LOGGER.handlers:
            # noinspection PyUnresolvedReferences
            if type(handler) == logging.StreamHandler and handler.stream == stream_io:
                have_handler = True
                break

        if not have_handler:
            logger_ch = logging.StreamHandler(stream=stream_io)
            logger_ch.setLevel(level=level)
            logger_ch.setFormatter(fmt=formatter)
            LOGGER.addHandler(logger_ch)

    return LOGGER


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    engine.LOGGER = LOGGER.getChild('Engine')
    back_test.LOGGER = LOGGER.getChild('BackTest')
    strategie.LOGGER = LOGGER.getChild('Strategies')


_ = get_logger()

from . import engine
from . import back_test
from . import profile
from . import strategie

engine.LOGGER.info(f'AlgoEngine version {__version__}')

# import addon module
try:
    from . import EngineAddon

    engine.LOGGER.info(f'AlgoEngine_Addons import successful, version {EngineAddon.__version__}')
except ImportError:
    engine.LOGGER.debug(f'Install AlgoEngine_Addons to use Statistics module\n{traceback.format_exc()}')
