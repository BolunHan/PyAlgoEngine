from __future__ import annotations

import datetime

import EventEngine

from . import EventDMA
from ._StrategyEngine import StrategyEngine
from ..Engine import LOGGER, TOPIC, MarketDataService, Balance, RiskProfile, PositionManagementService
from ..Engine.AlgoEngine import AlgoRegistry, AlgoEngine

LOGGER = LOGGER.getChild('BackTest')


def test_stop(code=0):
    EVENT_ENGINE.stop()
    # noinspection PyUnresolvedReferences, PyProtectedMember
    # `import os`
    # `os._exit(code)`


def test_start(start_date: datetime.date, end_date: datetime.date, data_loader: callable, **kwargs):
    EVENT_ENGINE.start()
    STRATEGY_ENGINE.back_test(
        start_date=start_date,
        end_date=end_date,
        data_loader=data_loader,
        **kwargs
    )


# in backtest, the global objects is newly inited to separate from production
EVENT_ENGINE = EventEngine.EventEngine()
MDS = MarketDataService()
ALGO_REGISTRY = AlgoRegistry()
ALGO_ENGINE = AlgoEngine(mds=MDS, registry=ALGO_REGISTRY)

BALANCE = Balance()
RISK_PROFILE = RiskProfile(mds=MDS, balance=BALANCE)
DMA = EventDMA(event_engine=EVENT_ENGINE, mds=MDS, risk_profile=RISK_PROFILE)
POSITION_TRACKER = PositionManagementService(dma=DMA, algo_engine=ALGO_ENGINE)
STRATEGY_ENGINE = StrategyEngine(event_engine=EVENT_ENGINE, position_tracker=POSITION_TRACKER)
BALANCE.add(strategy=STRATEGY_ENGINE, position_tracker=POSITION_TRACKER)

EVENT_ENGINE.register_handler(topic=TOPIC.realtime, handler=MDS.on_market_data)
EVENT_ENGINE.register_handler(topic=TOPIC.on_report, handler=BALANCE.on_report)
EVENT_ENGINE.register_handler(topic=TOPIC.on_order, handler=BALANCE.on_order)
STRATEGY_ENGINE.register()

MDS.synthetic_orderbook = True

__all__ = ['BALANCE', 'RISK_PROFILE', 'DMA', 'POSITION_TRACKER', 'STRATEGY_ENGINE', 'BALANCE', 'EVENT_ENGINE', 'MDS']
