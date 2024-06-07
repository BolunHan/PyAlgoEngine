__version__ = "0.4.2"

import logging
import os
import traceback

from . import profile
from .base.telemetrics import LOGGER

if 'ALGO_DIR' in os.environ:
    WORKING_DIRECTORY = os.path.realpath(os.environ['ALGO_DIR'])
else:
    WORKING_DIRECTORY = str(os.getcwd())


def set_logger(logger: logging.Logger):
    base.set_logger(logger=logger)
    engine.set_logger(logger=logger.getChild('Engine'))
    back_test.set_logger(logger=logger.getChild('BackTest'))
    strategy.set_logger(logger=logger.getChild('Strategy'))


from . import base
from . import engine
from . import back_test
from . import strategy

engine.LOGGER.info(f'AlgoEngine version {__version__}')

# import addon module
try:
    from . import algo_addon

    engine.LOGGER.info(f'PyAlgoEngineAddons import successful, version {algo_addon.__version__}')
except ImportError:
    algo_addon = None
    engine.LOGGER.debug(f'Install PyAlgoEngineAddons to use additional trading algos module\n{traceback.format_exc()}')

__all__ = ['LOGGER', 'base', 'engine', 'back_test', 'strategy', 'algo_addon']
