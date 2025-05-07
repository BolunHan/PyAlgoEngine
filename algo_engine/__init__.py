__version__ = "0.8.0.post5"

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
    from . import base
    from . import engine
    from . import backtest
    from . import strategy
    from . import apps

    base.set_logger(logger=logger)
    engine.set_logger(logger=logger.getChild('Engine'))
    backtest.set_logger(logger=logger.getChild('BackTest'))
    strategy.set_logger(logger=logger.getChild('Strategy'))
    apps.set_logger(logger=logger.getChild('Apps'))


LOGGER.info(f'AlgoEngine version {__version__}')

# import addon module
try:
    from . import algo_addon

    LOGGER.info(f'PyAlgoEngineAddons import successful, version {algo_addon.__version__}')
except ImportError:
    algo_addon = None
    LOGGER.debug(f'Install PyAlgoEngineAddons to use additional trading algos module\n{traceback.format_exc()}')

__all__ = [
    'apps', 'backtest', 'base', 'engine', 'monitor', 'profile', 'strategy', 'utils',
    'algo_addon',
    'LOGGER'
]
