from __future__ import annotations

import datetime

from PyQuantKit import TradeInstruction

from ._StrategyEngine import StrategyEngine
from ..Engine import LOGGER, EVENT_ENGINE, TOPIC, MDS, Balance, DirectMarketAccess, RiskProfile, PositionManagementService

LOGGER = LOGGER.getChild('BackTest')


class SimDMA(DirectMarketAccess):

    def _launch_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.launch_order(ticker=order.ticker), order=order, **kwargs)

    def _cancel_order_handler(self, order: TradeInstruction, **kwargs):
        EVENT_ENGINE.put(topic=TOPIC.cancel_order(ticker=order.ticker), order_id=order.order_id, **kwargs)

    def _reject_order_handler(self, order: TradeInstruction, **kwargs):
        raise NotImplementedError()


def test_stop(code=0):
    EVENT_ENGINE.stop()
    # noinspection PyUnresolvedReferences, PyProtectedMember
    # import os
    # os._exit(code)


def test_start(start_date: datetime.date, end_date: datetime.date, data_loader: callable, **kwargs):
    EVENT_ENGINE.start()
    STRATEGY_ENGINE.back_test(
        start_date=start_date,
        end_date=end_date,
        data_loader=data_loader,
        **kwargs
    )


BALANCE = Balance()
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = SimDMA(mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionManagementService(dma=DMA)
BALANCE.add(position_tracker=POSITION_TRACKER)
STRATEGY_ENGINE = StrategyEngine(position_tracker=POSITION_TRACKER)

EVENT_ENGINE.register_handler(topic=TOPIC.realtime, handler=MDS.on_market_data)
EVENT_ENGINE.register_handler(topic=TOPIC.on_report, handler=BALANCE.on_report)
EVENT_ENGINE.register_handler(topic=TOPIC.on_order, handler=BALANCE.on_order)
STRATEGY_ENGINE.register()

MDS.synthetic_orderbook = True
