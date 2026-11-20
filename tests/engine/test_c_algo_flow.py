import datetime
import time
import unittest

from event_engine.capi import EventEngineEx

from algo_engine.backtest import SimMatch
from algo_engine.base import TransactionData, TransactionDirection as Direction, TransactionOffset as Offset, TransactionSide
from algo_engine.engine import MDS, TOPIC_SET, Aggressive, AggressiveTimeout, Passive, PassiveTimeout, PositionManagementService, RiskProfile
from algo_engine.engine.algo import AlgoStatus
from algo_engine.exchange_profile import PROFILE_CN
from algo_engine.strategy import DMA, RISK_PROFILE


TICKER = '600000.SH'
SIDE_LONG = Direction.DIRECTION_LONG | Offset.OFFSET_OPEN
TS = datetime.datetime(2024, 11, 11, 10, 0).timestamp()


def setup_profile_rules(risk_profile: RiskProfile) -> None:
    risk_profile.set_rule('max_percentile', float('inf'), ticker=TICKER)

    for key in ('max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short'):
        risk_profile.set_rule(key, 1000., ticker=TICKER)

    for key in ('max_notional_long', 'max_notional_short'):
        risk_profile.set_rule(key, 1e9, ticker=TICKER)

    for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
        risk_profile.set_rule(key, 1e9)


def make_transaction(price: float, timestamp: float) -> TransactionData:
    # A bid-side print is an ask execution: it fills the buy algos under test.
    return TransactionData(
        ticker=TICKER,
        price=price,
        volume=100,
        timestamp=timestamp,
        side=TransactionSide.SIDE_LONG_OPEN,
    )


def wait_until(predicate, timeout: float = 2.) -> bool:
    deadline = time.time() + timeout

    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.001)

    return predicate()


class AlgoFlowHarness:
    def __init__(self, engine, position):
        self.engine = engine
        self.position = position

        engine.register_handler(topic=TOPIC_SET.on_report, handler=self.on_report)
        engine.register_handler(topic=TOPIC_SET.on_order, handler=self.on_order)

    def on_report(self, report, **kwargs):
        self.position.algo_engine.on_trade(report)

    def on_order(self, order, **kwargs):
        self.position.algo_engine.on_order(order)

    def unregister(self):
        self.engine.unregister_handler(topic=TOPIC_SET.on_report, handler=self.on_report)
        self.engine.unregister_handler(topic=TOPIC_SET.on_order, handler=self.on_order)


class TestAlgoFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()
        MDS.on_market_data(make_transaction(price=10.5, timestamp=TS))

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        RISK_PROFILE.clear()
        setup_profile_rules(RISK_PROFILE)
        MDS.on_market_data(make_transaction(price=10.5, timestamp=TS))

        self.engine = EventEngineEx()
        self.process_event_engine = DMA.event_engine
        DMA.event_engine = self.engine
        self.position = PositionManagementService()
        self.sim = SimMatch(ticker=TICKER, event_engine=self.engine, topic_set=TOPIC_SET, hit_prob=1.)
        self.sim.register()
        self.harness = AlgoFlowHarness(self.engine, self.position)

        self.engine.start()

    def tearDown(self) -> None:
        DMA.event_engine = self.process_event_engine
        self.engine.stop()
        self.harness.unregister()
        self.sim.unregister()

    def open(self, name, algo_cls, **kwargs):
        algo = self.position.open(TICKER, 100., SIDE_LONG, algo=name, **kwargs)
        self.assertIsInstance(algo, algo_cls)
        return algo

    def fill(self, price: float = 10.4, timestamp: float = TS + 1.):
        market_data = make_transaction(price=price, timestamp=timestamp)
        self.sim(market_data=market_data)

    def test_00_passive_fills_through_the_flow(self) -> None:
        algo = self.open('passive', Passive)

        self.assertTrue(wait_until(lambda: len(algo.working_orders) == 1))
        self.assertEqual(algo.status, AlgoStatus.working)

        first = next(iter(algo.working_orders))
        self.assertTrue(wait_until(lambda: first in self.sim.working))

        self.fill()

        self.assertTrue(wait_until(lambda: algo.status == AlgoStatus.done))
        self.assertEqual(algo.filled_volume, 100.)
        self.assertEqual(self.position.exposure_volume, {TICKER: 100.})
        self.assertEqual(len(self.position.trades), 1)

    def test_01_aggressive_fills_through_the_flow(self) -> None:
        algo = self.open('aggressive', Aggressive)

        self.assertTrue(wait_until(lambda: len(algo.working_orders) == 1))
        first = next(iter(algo.working_orders))
        self.assertTrue(wait_until(lambda: first in self.sim.working))

        self.fill()

        self.assertTrue(wait_until(lambda: algo.status == AlgoStatus.done))
        self.assertEqual(algo.filled_volume, 100.)

    def test_02_passive_timeout_cancels(self) -> None:
        algo = self.open('passive_timeout', PassiveTimeout, timeout=10.)

        self.assertTrue(wait_until(lambda: len(algo.working_orders) == 1))

        self.position.algo_engine.on_market_data(make_transaction(price=10.5, timestamp=TS + 20.))

        self.assertTrue(wait_until(lambda: algo.status == AlgoStatus.closed))
        self.assertEqual(algo.filled_volume, 0.)

    def test_03_aggressive_timeout_closes(self) -> None:
        algo = self.open('aggressive_timeout', AggressiveTimeout, timeout=10.)

        self.assertTrue(wait_until(lambda: len(algo.working_orders) == 1))

        self.position.algo_engine.on_market_data(make_transaction(price=10.5, timestamp=TS + 20.))

        self.assertTrue(wait_until(lambda: algo.status == AlgoStatus.closed))
        self.assertEqual(algo.filled_volume, 0.)
        self.assertEqual(len(algo.working_orders), 0)


if __name__ == '__main__':
    unittest.main()
