import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('BackTest')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    replay.LOGGER = LOGGER.getChild('Replay')
    sim_match.LOGGER = LOGGER.getChild('SimMatch')


from .replay import MarketDateCallable, MarketDataLoader, MarketDataBulkLoader, Replay, SimpleReplay, ProgressReplay, ProgressiveReplay
from .sim_match import SimMatch

__all__ = ['MarketDateCallable', 'MarketDataLoader', 'MarketDataBulkLoader', 'Replay', 'SimpleReplay', 'ProgressReplay', 'ProgressiveReplay', 'SimMatch']
