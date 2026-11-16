from __future__ import annotations

import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Engine')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    c_event_engine.EVENT_ENGINE.logger = logger.getChild('EventEngine')
    c_market_engine.LOGGER = logger.getChild('MarketEngine')
    c_dma.LOGGER = logger.getChild('DMA')
    c_inventory.LOGGER = logger.getChild('Inventory')
    c_position.LOGGER = logger.getChild('Position')
    c_risk_profile.LOGGER = logger.getChild('RiskProfile')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


from .c_event_engine import EVENT_ENGINE, TOPIC_SET, TopicSet, TopicSetEx
from .c_market_engine import MDS, MarketDataMonitor, MarketDataMonitorMap, MarketDataMonitorProtocol, MarketDataService, MonitorManager
from .algo import Aggressive, AggressiveTimeout, AlgoTemplate, AlgoStatus, Passive, PassiveTimeout
from .risk_profile import RefusalWindowAfterRule, RefusalWindowBeforeRule, RefusalWindowBetweenRule, RefusalWindowRule, RiskLimits, RiskLimitsMapView, RiskProfile, RiskRule, RiskRuleTemplate, RiskType, RiskViolation
from .c_inventory import INVENTORY, Inventory
from .c_dma_algo import ALGO_ENGINE, ALGO_REGISTRY, AlgoEngine, AlgoLimitMode, AlgoRegistry
from .c_balance import Balance, BalanceFlag
from .c_dma import DirectMarketAccess, DirectMarketAccessProtocol
from .c_position import PositionCallbackEvent, PositionManagementService, PositionOrderCallbackProtocol, PositionTradeCallbackProtocol

__all__ = [
    'LOGGER',
    'EVENT_ENGINE', 'TOPIC_SET', 'TopicSet', 'TopicSetEx',
    'AlgoTemplate', 'ALGO_ENGINE', 'ALGO_REGISTRY',
    'MDS', 'MarketDataService', 'MarketDataMonitor', 'MarketDataMonitorProtocol', 'MonitorManager', 'Singleton',
    'Balance', 'BalanceFlag', 'DirectMarketAccess', 'DirectMarketAccessProtocol', 'PositionManagementService', 'PositionCallbackEvent', 'PositionOrderCallbackProtocol', 'PositionTradeCallbackProtocol', 'INVENTORY', 'Inventory', 'RefusalWindowAfterRule', 'RefusalWindowBeforeRule', 'RefusalWindowBetweenRule', 'RefusalWindowRule', 'RiskLimits', 'RiskLimitsMapView', 'RiskProfile', 'RiskRule', 'RiskRuleTemplate', 'RiskType', 'RiskViolation',
    'AlgoEngine', 'AlgoLimitMode', 'AlgoRegistry', 'AlgoStatus',
    'Passive', 'PassiveTimeout', 'Aggressive', 'AggressiveTimeout'
]
