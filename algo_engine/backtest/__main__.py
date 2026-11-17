__package__ = 'algo_engine.backtest'

import datetime
from collections.abc import Callable

import event_engine

from ..engine import TOPIC_SET, MarketDataService, RiskProfile, PositionManagementService
from ..engine._deprecated_algo_engine import AlgoRegistry, AlgoEngine
from ..strategy import EventDMA
from ..strategy.strategy_engine import StrategyEngine


def test_stop(code=0):
    EVENT_ENGINE.stop()
    # noinspection PyUnresolvedReferences, PyProtectedMember
    # `import os`
    # `os._exit(code)`


def test_start(start_date: datetime.date, end_date: datetime.date, data_loader: Callable, **kwargs):
    EVENT_ENGINE.start()
    STRATEGY_ENGINE.back_test(
        start_date=start_date,
        end_date=end_date,
        data_loader=data_loader,
        **kwargs
    )


# in backtest, the global objects is newly inited to separate from production
EVENT_ENGINE = event_engine.EventEngine()
MDS = MarketDataService()
ALGO_REGISTRY = AlgoRegistry()
ALGO_ENGINE = AlgoEngine(mds=MDS, registry=ALGO_REGISTRY)

RISK_PROFILE = RiskProfile()
DMA = EventDMA(event_engine=EVENT_ENGINE)
POSITION_TRACKER = PositionManagementService()
STRATEGY_ENGINE = StrategyEngine(event_engine=EVENT_ENGINE, position_tracker=POSITION_TRACKER)

EVENT_ENGINE.register_handler(topic=TOPIC_SET.realtime, handler=MDS.on_market_data)
EVENT_ENGINE.register_handler(topic=TOPIC_SET.on_report, handler=POSITION_TRACKER.algo_engine.on_trade)
EVENT_ENGINE.register_handler(topic=TOPIC_SET.on_order, handler=POSITION_TRACKER.algo_engine.on_order)
STRATEGY_ENGINE.register()

MDS.synthetic_orderbook = True

__all__ = ['RISK_PROFILE', 'DMA', 'POSITION_TRACKER', 'STRATEGY_ENGINE', 'EVENT_ENGINE', 'MDS']
