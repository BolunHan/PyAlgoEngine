import datetime
import unittest

from algo_engine.base import OrderState, TradeInstruction, TradeReport, TransactionData, TransactionDirection as Direction, TransactionOffset as Offset, TransactionSide
from algo_engine.engine import (
    ALGO_ENGINE,
    ALGO_REGISTRY,
    MDS,
    Aggressive,
    AggressiveTimeout,
    AlgoLimitMode,
    DirectMarketAccess,
    Passive,
    PassiveTimeout,
    PositionManagementService,
    RiskProfile,
)
from algo_engine.engine.algo import AlgoStatus
from algo_engine.exchange_profile import PROFILE_CN
from algo_engine.strategy import RISK_PROFILE


TICKER = '600000.SH'
SIDE_LONG = Direction.DIRECTION_LONG | Offset.OFFSET_OPEN
TS = datetime.datetime(2024, 11, 11, 10, 0).timestamp()


def setup_profile_rules(risk_profile: RiskProfile, ticker: str = TICKER) -> None:
    risk_profile.set_rule('max_percentile', float('inf'), ticker=ticker)

    for key in ('max_trade_long', 'max_trade_short', 'max_exposure_long', 'max_exposure_short'):
        risk_profile.set_rule(key, 1000., ticker=ticker)

    for key in ('max_notional_long', 'max_notional_short'):
        risk_profile.set_rule(key, 1e9, ticker=ticker)

    for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
        risk_profile.set_rule(key, float('inf'))


def feed_price(price: float = 10.5, timestamp: float = TS) -> TransactionData:
    market_data = TransactionData(
        ticker=TICKER,
        price=price,
        volume=100,
        timestamp=timestamp,
        side=TransactionSide.SIDE_LONG_OPEN,
    )
    MDS.on_market_data(market_data)
    return market_data


def make_report(order, volume: float = 100., price: float = 10.5) -> TradeReport:
    return TradeReport(
        ticker=TICKER,
        volume=volume,
        price=price,
        side=TransactionSide.SIDE_LONG_OPEN,
        timestamp=TS,
        order_id=order.order_id,
    )


class TestAlgo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()
        feed_price()

    @classmethod
    def tearDownClass(cls) -> None:
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        RISK_PROFILE.clear()
        setup_profile_rules(RISK_PROFILE)
        feed_price()

    def make_algo(self, cls, **kwargs):
        return cls(TICKER, SIDE_LONG, 100., **kwargs)

    def test_00_passive_launch_and_fill(self) -> None:
        algo = self.make_algo(Passive)

        algo.launch()

        self.assertEqual(algo.status, AlgoStatus.Working)
        self.assertEqual(algo.working_volume, 100.)
        self.assertEqual(len(algo.working_orders), 1)
        self.assertEqual(len(algo.orders), 1)

        order = next(iter(algo.working_orders.values()))
        self.assertEqual(order.ticker, TICKER)
        self.assertEqual(order.volume, 100.)
        self.assertEqual(order.limit_price, 10.5)

        ALGO_ENGINE.on_trade(make_report(order))

        self.assertEqual(algo.status, AlgoStatus.Done)
        self.assertEqual(algo.filled_volume, 100.)
        self.assertEqual(algo.filled_notional, 1050.)
        self.assertEqual(algo.exposure_volume, 100.)
        self.assertEqual(algo.working_volume, 0.)
        self.assertEqual(len(list(algo.trades)), 1)

    def test_01_passive_cancel(self) -> None:
        algo = self.make_algo(Passive)

        algo.launch()
        order = next(iter(algo.working_orders.values()))

        algo.cancel()
        self.assertEqual(algo.status, AlgoStatus.Stopping)

        event = TradeInstruction(
            ticker=order.ticker,
            timestamp=TS,
            side=order.side,
            volume=order.volume,
            limit_price=order.limit_price,
            order_id=order.order_id,
        )
        event.set_order_state(OrderState.Canceled, TS)
        ALGO_ENGINE.on_order(event)
        del event

        self.assertEqual(algo.status, AlgoStatus.Closed)
        self.assertEqual(algo.filled_volume, 0.)
        self.assertEqual(next(iter(algo.orders.values())).ticker, TICKER)

    def test_02_aggressive_relaunch(self) -> None:
        algo = self.make_algo(Aggressive)

        algo.launch()
        first = next(iter(algo.working_orders.values()))

        event = TradeInstruction(
            ticker=first.ticker,
            timestamp=TS,
            side=first.side,
            volume=first.volume,
            limit_price=first.limit_price,
            order_id=first.order_id,
        )
        event.set_order_state(OrderState.Canceled, TS)
        ALGO_ENGINE.on_order(event)
        del event

        self.assertEqual(algo.status, AlgoStatus.Working)
        self.assertEqual(len(algo.working_orders), 1)

        second = next(iter(algo.working_orders.values()))
        self.assertNotEqual(second.order_id, first.order_id)

        ALGO_ENGINE.on_trade(make_report(second))

        self.assertEqual(algo.status, AlgoStatus.Done)
        self.assertEqual(algo.filled_volume, 100.)

    def test_03_passive_timeout(self) -> None:
        algo = self.make_algo(PassiveTimeout, timeout=10.)

        algo.launch()
        order = next(iter(algo.working_orders.values()))

        ALGO_ENGINE.on_market_data(feed_price(timestamp=TS + 20.))

        self.assertEqual(algo.status, AlgoStatus.Stopping)

        event = TradeInstruction(
            ticker=order.ticker,
            timestamp=TS + 20.,
            side=order.side,
            volume=order.volume,
            limit_price=order.limit_price,
            order_id=order.order_id,
        )
        event.set_order_state(OrderState.Canceled, TS + 20.)
        ALGO_ENGINE.on_order(event)
        del event

        self.assertEqual(algo.status, AlgoStatus.Closed)

    def test_04_aggressive_timeout(self) -> None:
        algo = self.make_algo(AggressiveTimeout, timeout=10.)

        algo.launch()
        order = next(iter(algo.working_orders.values()))

        ALGO_ENGINE.on_market_data(feed_price(timestamp=TS + 20.))

        self.assertEqual(algo.status, AlgoStatus.Stopping)
        self.assertEqual(len(algo.working_orders), 1)

        event = TradeInstruction(
            ticker=order.ticker,
            timestamp=TS + 20.,
            side=order.side,
            volume=order.volume,
            limit_price=order.limit_price,
            order_id=order.order_id,
        )
        event.set_order_state(OrderState.Canceled, TS + 20.)
        ALGO_ENGINE.on_order(event)
        del event

        self.assertEqual(algo.status, AlgoStatus.Closed)

    def test_05_position_open_and_fill(self) -> None:
        position = PositionManagementService()
        algo = position.open(TICKER, 100., SIDE_LONG)

        self.assertIsInstance(algo, Passive)
        self.assertEqual(len(position.working_algos), 1)
        self.assertEqual(position.working_volume, {'Long': {TICKER: 100.}, 'Short': {}})
        self.assertEqual(len(position.orders), 1)

        order = next(iter(position.working_order.values()))
        ALGO_ENGINE.on_trade(make_report(order))

        self.assertEqual(position.exposure_volume, {TICKER: 100.})
        self.assertEqual(position.notional, {TICKER: 1050.})
        self.assertEqual(position.pnl(), {TICKER: 0.})
        self.assertEqual(len(position.working_algos), 0)
        self.assertEqual(len(position.trades), 1)
        self.assertEqual(algo.ticker, TICKER)

    def test_06_registry_and_limit(self) -> None:
        self.assertIs(ALGO_REGISTRY.cast('passive'), Passive)
        self.assertIn('passive', ALGO_REGISTRY)
        self.assertEqual(set(ALGO_REGISTRY.names), {'passive', 'passive_timeout', 'aggressive', 'aggressive_timeout'})

        access = DirectMarketAccess()

        self.assertEqual(access.calculate_limit(TICKER, SIDE_LONG), 10.5)
        self.assertEqual(access.calculate_limit(TICKER, SIDE_LONG, mode=AlgoLimitMode.loose), 10.5)

        with self.assertRaises(TypeError):
            access.calculate_limit(TICKER, SIDE_LONG, mode='strict')

        with self.assertRaises(TypeError):
            Passive(TICKER, SIDE_LONG, 100., limit_mode='strict')

        algo = Passive(TICKER, SIDE_LONG, 100., limit_mode=AlgoLimitMode.loose)
        self.assertEqual(algo.limit_mode, AlgoLimitMode.loose)

        with self.assertRaises(TypeError):
            algo.limit_mode = 'strict'

        algo.limit_price = 10.25
        self.assertEqual(algo.limit_price, 10.25)

        algo.limit_adjust_factor = 0.5
        self.assertEqual(algo.limit_adjust_factor, 0.5)

        with self.assertRaises(TypeError):
            algo.limit_price = None

        with self.assertRaises(TypeError):
            algo.limit_adjust_factor = 'strict'

    def test_07_launch_without_price_is_refused(self) -> None:
        algo = Passive('NO_PRICE.SH', SIDE_LONG, 100.)

        with self.assertRaises(ValueError):
            algo.launch()

        self.assertEqual(algo.status, AlgoStatus.Idle)
        self.assertEqual(len(algo.orders), 0)
        self.assertEqual(len(algo.working_orders), 0)
        self.assertEqual(algo.working_volume, 0.)

    def test_08_registry_wrapper_dispatch(self) -> None:
        """A wrapped record resolves to the wrapper class of its registration."""
        position = PositionManagementService()

        passive = position.open(TICKER, 100., SIDE_LONG)
        aggressive = position.open(TICKER, 100., SIDE_LONG, algo='aggressive')

        wrapped = {algo.algo_id: algo for algo in position.algos.values()}

        self.assertIsInstance(wrapped[passive.algo_id], Passive)
        self.assertNotIsInstance(wrapped[passive.algo_id], Aggressive)
        self.assertFalse(wrapped[passive.algo_id].relaunch)
        self.assertIsInstance(wrapped[aggressive.algo_id], Aggressive)
        self.assertTrue(wrapped[aggressive.algo_id].relaunch)


if __name__ == '__main__':
    unittest.main()
