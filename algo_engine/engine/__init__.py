from __future__ import annotations

import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Engine')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    algo_engine.LOGGER = logger.getChild('AlgoEngine')
    c_event_engine.EVENT_ENGINE.logger = logger.getChild('EventEngine')
    c_market_engine.LOGGER = logger.getChild('MarketEngine')
    trade_engine.LOGGER = logger.getChild('TradeEngine')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


from .c_event_engine import EVENT_ENGINE, TOPIC_SET, TopicSet, TopicSetEx
from .c_market_engine import MDS, MarketDataMonitor, MarketDataMonitorMap, MarketDataMonitorProtocol, MarketDataService, MonitorManager
from .algo_engine import ALGO_ENGINE, ALGO_REGISTRY, AlgoTemplate
from .trade_engine import Balance, DirectMarketAccess, Inventory, PositionManagementService, RiskProfile

__all__ = [
    'LOGGER',
    'EVENT_ENGINE', 'TOPIC_SET', 'TopicSet', 'TopicSetEx',
    'AlgoTemplate', 'ALGO_ENGINE', 'ALGO_REGISTRY',
    'MDS', 'MarketDataService', 'MarketDataMonitor', 'MarketDataMonitorProtocol', 'MonitorManager', 'Singleton',
    'DirectMarketAccess', 'Balance', 'PositionManagementService', 'Inventory', 'RiskProfile'
]
