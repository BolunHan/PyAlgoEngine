__version__ = "0.10.3.post2"

import functools
import logging
import os
import pathlib
import traceback

from .base.telemetrics import LOGGER

if 'ALGO_DIR' in os.environ:
    WORKING_DIRECTORY = os.path.realpath(os.environ['ALGO_DIR'])
else:
    WORKING_DIRECTORY = str(os.getcwd())

from . import base
from . import exchange_profile
from . import engine
from . import backtest
from . import strategy
from . import apps
from . import monitor
from . import utils


def set_logger(logger: logging.Logger):
    from . import base
    from . import exchange_profile
    from . import engine
    from . import backtest
    from . import strategy
    from . import apps

    base.set_logger(logger=logger)
    exchange_profile.set_logger(logger=logger.getChild('ExchangeProfile'))
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


@functools.cache
def get_include() -> list[str]:
    import os
    from .base import CONFIG

    res_dir = pathlib.Path(__file__).parent
    LOGGER.info(f'Building with <PyAlgoEngine> version: "{__version__}", resource directory: "{res_dir}", config: "{CONFIG}".')

    scr_dir = [
        os.path.realpath(res_dir),
        os.path.realpath(res_dir / 'base'),
        os.path.realpath(res_dir / 'base' / 'c_market_data'),
        os.path.realpath(res_dir / 'exchange_profile'),
        os.path.realpath(res_dir / 'engine'),
    ]

    import event_engine
    dep_dir = event_engine.get_include()

    return scr_dir + dep_dir


__all__ = [
    'apps', 'backtest', 'base', 'engine', 'exchange_profile', 'monitor', 'profile', 'strategy', 'utils',
    'algo_addon',
    'get_include',
    'LOGGER'
]
