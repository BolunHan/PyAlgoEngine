import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('BackTest')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    replay.LOGGER = LOGGER.getChild('Replay')
    c_simmatch.LOGGER = LOGGER.getChild('SimMatch')


from .replay import PyDataScope, MarketDateCallable, MarketDataLoader, MarketDataBulkLoader, Replay, SimpleReplay, ProgressReplay
from .c_simmatch import OrderMapView, SimMatch

__all__ = ['PyDataScope', 'MarketDateCallable', 'MarketDataLoader', 'MarketDataBulkLoader', 'Replay', 'SimpleReplay', 'ProgressReplay', 'OrderMapView', 'SimMatch']
