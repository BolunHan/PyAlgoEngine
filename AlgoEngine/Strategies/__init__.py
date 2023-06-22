from __future__ import annotations

import threading

from PyQuantKit import TradeInstruction

from ._StrategyEngine import StrategyEngine
from ..Engine import LOGGER
from ..Engine.EventEngine import EVENT_ENGINE, TOPIC
from ..Engine.MarketEngine import MDS, MarketDataService
from ..Engine.TradeEngine import Balance, Inventory, DirectMarketAccess, RiskProfile, PositionManagementService

LOGGER = LOGGER.getChild('Strategies')


class EventDMA(DirectMarketAccess):
    def __init__(self, mds: MarketDataService, risk_profile: RiskProfile, event_engine=None, cool_down: float = None):
        self.event_engine = EVENT_ENGINE if event_engine is None else event_engine
        super().__init__(mds=mds, risk_profile=risk_profile, cool_down=cool_down)

    def _launch_order_handler(self, order: TradeInstruction, **kwargs):
        self.event_engine.put(topic=TOPIC.launch_order(ticker=order.ticker), order=order, **kwargs)

    def _cancel_order_handler(self, order: TradeInstruction, **kwargs):
        self.event_engine.put(topic=TOPIC.cancel_order(ticker=order.ticker), order_id=order.order_id, **kwargs)

    def _reject_order_handler(self, order: TradeInstruction, **kwargs):
        raise NotImplementedError()


REPLAY_LOCK = threading.Lock()
INVENTORY = Inventory()
BALANCE = Balance(inventory=INVENTORY)
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = EventDMA(mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionManagementService(dma=DMA)
STRATEGY_ENGINE = StrategyEngine(event_engine=EVENT_ENGINE, position_tracker=POSITION_TRACKER)
BALANCE.add(strategy=STRATEGY_ENGINE, position_tracker=POSITION_TRACKER)

__all__ = ['INVENTORY', 'BALANCE', 'RISK_PROFILE', 'DMA', 'POSITION_TRACKER', 'STRATEGY_ENGINE']
