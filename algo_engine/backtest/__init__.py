import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('BackTest')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    replay.LOGGER = LOGGER.getChild('Replay')
    sim_match.LOGGER = LOGGER.getChild('SimMatch')


from .replay import PyDataScope, MarketDateCallable, MarketDataLoader, MarketDataBulkLoader, Replay, SimpleReplay, ProgressReplay
from .sim_match import SimMatch

__all__ = ['PyDataScope', 'MarketDateCallable', 'MarketDataLoader', 'MarketDataBulkLoader', 'Replay', 'SimpleReplay', 'ProgressReplay', 'SimMatch']
