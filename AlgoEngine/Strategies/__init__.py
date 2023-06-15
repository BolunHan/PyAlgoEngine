from __future__ import annotations

import threading

from PyQuantKit import TradeInstruction

from ..Engine import LOGGER
from ..Engine.EventEngine import EVENT_ENGINE, TOPIC
from ..Engine.MarketEngine import MDS
from ..Engine.TradeEngine import Balance, DirectMarketAccess, RiskProfile, PositionManagementService

LOGGER = LOGGER.getChild('Strategies')


class SimDMA(DirectMarketAccess):

    def _launch_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.launch_order(ticker=order.ticker), order=order, **kwargs)

    def _cancel_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.cancel_order(ticker=order.ticker), order_id=order.order_id, **kwargs)

    def _reject_order_handler(self, order: TradeInstruction, **kwargs):
        raise NotImplementedError()


REPLAY_LOCK = threading.Lock()
BALANCE = Balance()
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = SimDMA(mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionManagementService(dma=DMA)
BALANCE.add(position_tracker=POSITION_TRACKER)
