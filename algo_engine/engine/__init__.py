from __future__ import annotations

import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Engine')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    algo_engine.LOGGER = logger.getChild('AlgoEngine')
    event_engine.EVENT_ENGINE.logger = logger.getChild('EventEngine')
    c_market_engine.LOGGER = logger.getChild('MarketEngine')
    trade_engine.LOGGER = logger.getChild('TradeEngine')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


from .event_engine import EVENT_ENGINE, TOPIC
# from .market_engine import MDS, MarketDataService, MarketDataMonitor, MonitorManager
from .c_market_engine import MDS, MarketDataService, MarketDataMonitor, MonitorManager
from .algo_engine import AlgoTemplate, ALGO_ENGINE, ALGO_REGISTRY
from .trade_engine import DirectMarketAccess, Balance, PositionManagementService, Inventory, RiskProfile

__all__ = ['EVENT_ENGINE', 'TOPIC',
           'AlgoTemplate', 'ALGO_ENGINE', 'ALGO_REGISTRY',
           'MDS', 'MarketDataService', 'MarketDataMonitor', 'MonitorManager', 'Singleton',
           'DirectMarketAccess', 'Balance', 'PositionManagementService', 'Inventory', 'RiskProfile']
