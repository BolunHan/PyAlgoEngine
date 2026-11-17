import logging
from threading import Lock

from .. import LOGGER

from ..engine import EVENT_ENGINE, INVENTORY, PositionManagementService
from ..engine.c_dma import DMA, RISK_PROFILE, EventDMA

LOGGER = LOGGER.getChild('Strategy')

from .strategy_engine import StrategyEngine


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    strategy_engine.LOGGER = logger.getChild('Strategy')


REPLAY_LOCK = Lock()
POSITION_TRACKER = PositionManagementService()
STRATEGY_ENGINE = StrategyEngine(event_engine=EVENT_ENGINE, position_tracker=POSITION_TRACKER)  # need to be registered, also register MDS

__all__ = ['INVENTORY', 'RISK_PROFILE', 'DMA', 'POSITION_TRACKER', 'STRATEGY_ENGINE']
