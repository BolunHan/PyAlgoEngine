import logging
from threading import Lock

from .. import LOGGER
from ..base import TradeInstruction
from ..engine import EVENT_ENGINE, TOPIC, MDS, MarketDataService, Balance, Inventory, DirectMarketAccess, RiskProfile, PositionManagementService

LOGGER = LOGGER.getChild('Strategy')

from .strategy_engine import StrategyEngine


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


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    strategy_engine.LOGGER = logger.getChild('Strategy')


REPLAY_LOCK = Lock()
INVENTORY = Inventory()
BALANCE = Balance(inventory=INVENTORY)  # need to be registered
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = EventDMA(mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionManagementService(dma=DMA)
STRATEGY_ENGINE = StrategyEngine(event_engine=EVENT_ENGINE, position_tracker=POSITION_TRACKER)  # need to be registered, also register MDS
BALANCE.add(strategy=STRATEGY_ENGINE, position_tracker=POSITION_TRACKER)

__all__ = ['INVENTORY', 'BALANCE', 'RISK_PROFILE', 'DMA', 'POSITION_TRACKER', 'STRATEGY_ENGINE']
