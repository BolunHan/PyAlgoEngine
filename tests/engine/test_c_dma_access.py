import datetime
import gc
import time
import unittest

from event_engine.capi import EventEngineEx

from algo_engine.base import TransactionData, TransactionDirection as Direction, TransactionOffset as Offset, TransactionSide
from algo_engine.engine import (
    ALGO_ENGINE,
    MDS,
    TOPIC_SET,
    AlgoLimitMode,
    AlgoStatus,
    DirectMarketAccess,
    DirectMarketAccessProtocol,
    PositionManagementService,
    RiskProfile,
)
from algo_engine.engine.c_dma import EventDMA
from algo_engine.exchange_profile import PROFILE_CN
from algo_engine.strategy import DMA, RISK_PROFILE


TICKER = '600001.SH'
SIDE_LONG = Direction.DIRECTION_LONG | Offset.OFFSET_OPEN
SIDE_SHORT = Direction.DIRECTION_SHORT | Offset.OFFSET_OPEN
TS = datetime.datetime(2024, 11, 11, 10, 0).timestamp()


def setup_profile_rules(risk_profile, ticker: str = TICKER, trade_long: float = 1000.) -> None:
    risk_profile.set_rule('max_percentile', float('inf'), ticker=ticker)

    risk_profile.set_rule('max_trade_long', trade_long, ticker=ticker)
    risk_profile.set_rule('max_trade_short', 1000., ticker=ticker)
    risk_profile.set_rule('max_exposure_long', 1000., ticker=ticker)
    risk_profile.set_rule('max_exposure_short', 1000., ticker=ticker)

    for key in ('max_notional_long', 'max_notional_short'):
        risk_profile.set_rule(key, 1e9, ticker=ticker)

    for key in ('max_ttl_notional_long', 'max_ttl_notional_short', 'max_net_notional_long', 'max_net_notional_short'):
        risk_profile.set_rule(key, 1e9)


def feed_price(ticker: str = TICKER, price: float = 10.5, timestamp: float = TS) -> None:
    MDS.on_market_data(TransactionData(
        ticker=ticker,
        price=price,
        volume=100,
        timestamp=timestamp,
        side=TransactionSide.SIDE_LONG_OPEN,
    ))


def wait_until(predicate, timeout: float = 2.) -> bool:
    deadline = time.time() + timeout

    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.001)

    return predicate()


class TestDirectMarketAccess(unittest.TestCase):
    """The C-backed access: the dispatch surface, its husk and the process wiring."""

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

    def test_00_is_a_virtual_protocol_subclass(self) -> None:
        self.assertTrue(issubclass(DirectMarketAccess, DirectMarketAccessProtocol))
        self.assertTrue(isinstance(DirectMarketAccess(), DirectMarketAccessProtocol))

        access = DirectMarketAccess()
        self.assertIn(DirectMarketAccess.__name__, repr(access))
        self.assertNotIn('OrderHandler', repr(access))

    def test_01_the_access_accepts_no_user_order(self) -> None:
        access = DirectMarketAccess()

        for name in ('launch_order', 'cancel_order', 'launch_order_unchecked', 'check_order'):
            self.assertFalse(hasattr(access, name))

    def test_02_wrapper_reads_its_own_blocks(self) -> None:
        access = DirectMarketAccess()

        self.assertIs(access.mds, MDS)
        self.assertIsInstance(access.risk_profile, RiskProfile)
        self.assertIsNot(access.risk_profile, RISK_PROFILE)
        self.assertIsNot(access.algo_engine, ALGO_ENGINE)
        self.assertIsInstance(access.timestamp, float)
        self.assertEqual(access.market_price[TICKER], 10.5)
        self.assertIsNotNone(access.market_time)
        self.assertEqual(access.process_queue(), 0)

        # Every access creates and owns a risk profile of its own.
        bare = DirectMarketAccess()
        self.assertIsInstance(bare.risk_profile, RiskProfile)
        self.assertIsNot(bare.risk_profile, access.risk_profile)
        self.assertEqual(bare.process_queue(), 0)

    def test_03_unbound_wrapper_is_a_husk(self) -> None:
        access = DirectMarketAccess.__new__(DirectMarketAccess)

        self.assertEqual(repr(access), '<DirectMarketAccess: unbound>')

        self.assertIsNone(access.mds)
        self.assertIsNone(access.risk_profile)

        for name in ('protocol_bound', 'timestamp', 'market_price', 'market_time'):
            with self.assertRaises(RuntimeError):
                getattr(access, name)

        # The functional entries are unguarded; the C block decides.
        with self.assertRaises(RuntimeError):
            access.process_queue()
        self.assertIsNone(access.calculate_limit(TICKER, SIDE_LONG))

    def test_04_calculate_limit_combines_the_three_prices(self) -> None:
        access = DirectMarketAccess()

        # The market price alone.
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_LONG), 10.5)

        # Strict tightens the long limit to the lowest of the three; loose
        # takes the highest.
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_LONG, limit_price=10.), 10.)
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_LONG, limit_price=10., mode=AlgoLimitMode.loose), 10.5)
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_LONG, limit_adjust_factor=0.1), 10.5)
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_LONG, limit_adjust_factor=0.1, mode=AlgoLimitMode.loose), 11.55)

        # The short side mirrors: strict lifts the limit, loose lowers it.
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_SHORT, limit_price=11.), 11.)
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_SHORT, limit_price=11., mode=AlgoLimitMode.loose), 10.5)
        self.assertAlmostEqual(access.calculate_limit(TICKER, SIDE_SHORT, limit_adjust_factor=0.1, mode=AlgoLimitMode.loose), 9.45)

        # A ticker without a quote yields no limit.
        self.assertIsNone(access.calculate_limit('NO_PRICE.SH', SIDE_LONG))

    def test_05_repeated_alloc_and_release(self) -> None:
        for index in range(64):
            access = DirectMarketAccess()
            self.assertIs(access.mds, MDS)
            self.assertEqual(access.process_queue(), 0)

        del access
        gc.collect()

    def test_06_process_singletons_are_wired(self) -> None:
        self.assertIsInstance(DMA, EventDMA)
        self.assertIs(DMA.mds, MDS)
        self.assertIs(DMA.risk_profile, RISK_PROFILE)
        self.assertIs(DMA.algo_engine, ALGO_ENGINE)

        position = PositionManagementService()
        self.assertIs(position.dma, DMA)
        self.assertIs(position.algo_engine, ALGO_ENGINE)

    def test_07_process_access_rejects_a_breaching_order(self) -> None:
        RISK_PROFILE.set_rule('max_trade_long', 5., ticker=TICKER)
        position = PositionManagementService()

        # The reject handler is a no-op for the process access: the C engine
        # has already applied the rejection to the algo.
        with self.assertNoLogs('PyAlgoEngine', level='ERROR'):
            algo = position.open(TICKER, 100., SIDE_LONG)

        self.assertEqual(algo.status, AlgoStatus.Closed)
        self.assertEqual(algo.filled_volume, 0.)
        self.assertEqual(algo.working_volume, 0.)
        self.assertEqual(len(algo.working_orders), 0)
        self.assertEqual(len(algo.orders), 1)

        order = next(iter(algo.orders.values()))
        self.assertEqual(order.order_state.name, 'STATE_REJECTED')
        self.assertEqual(order.volume, 100.)

        self.assertEqual(position.exposure_volume, {})
        self.assertEqual(position.working_volume, {'Long': {}, 'Short': {}})
        self.assertEqual(len(position.working_algos), 0)

    def test_08_process_access_publishes_launch_and_cancel(self) -> None:
        engine = EventEngineEx()
        previous_engine = DMA.event_engine
        DMA.event_engine = engine

        launched: list = []
        canceled: list = []

        engine.register_handler(topic=TOPIC_SET.launch_order(ticker=TICKER), handler=lambda order, **kwargs: launched.append(order))
        engine.register_handler(topic=TOPIC_SET.cancel_order(ticker=TICKER), handler=lambda order_id, **kwargs: canceled.append(order_id))
        engine.start()

        try:
            position = PositionManagementService()
            algo = position.open(TICKER, 100., SIDE_LONG)

            self.assertTrue(wait_until(lambda: len(launched) == 1))
            self.assertEqual(launched[0].ticker, TICKER)
            self.assertEqual(launched[0].volume, 100.)
            self.assertEqual(launched[0].limit_price, 10.5)

            order_id = launched[0].order_id

            algo.cancel()
            self.assertTrue(wait_until(lambda: canceled == [order_id]))
        finally:
            engine.stop()
            DMA.event_engine = previous_engine

    def test_09_protocol_binding_requires_both_handlers(self) -> None:
        class Bound(DirectMarketAccess):
            def launch_order_handler(self, order):
                pass

            def cancel_order_handler(self, order):
                pass

        class Partial(DirectMarketAccess):
            def launch_order_handler(self, order):
                pass

        class WithCallbacks(Bound):
            def rejected_order_callback(self, order):
                pass

            def filled_order_callback(self, order, report):
                pass

        self.assertFalse(DirectMarketAccess().protocol_bound)
        self.assertTrue(Bound().protocol_bound)
        self.assertFalse(Partial().protocol_bound)
        self.assertTrue(WithCallbacks().protocol_bound)

    def test_10_singletons_are_the_module_instances(self) -> None:
        import algo_engine.engine.c_dma as c_dma_module
        import algo_engine.strategy as strategy_package

        self.assertIs(c_dma_module.DMA, DMA)
        self.assertIs(c_dma_module.RISK_PROFILE, RISK_PROFILE)
        self.assertIs(strategy_package.DMA, DMA)
        self.assertIs(strategy_package.RISK_PROFILE, RISK_PROFILE)

    def test_11_cython_subclass_overrides_the_c_handlers(self) -> None:
        from tests.c_state_access_toolkit import CythonDMA

        access = CythonDMA()
        self.assertTrue(access.protocol_bound)

        access.fire_launch()
        access.fire_cancel()
        access.fire_reject()
        access.fire_filled()

        self.assertEqual(access.launch_count, 1)
        self.assertEqual(access.cancel_count, 1)
        self.assertEqual(access.rejected_count, 1)
        self.assertEqual(access.filled_count, 1)

        ret_code, available_cash, flags = access.fire_balance()
        self.assertEqual(ret_code, 0)
        self.assertEqual(access.balance_count, 1)
        self.assertEqual(available_cash, 123.)
        self.assertTrue(flags & 1 << 8)

    def test_12_the_algo_module_adopts_the_process_engine(self) -> None:
        import subprocess
        import sys

        import algo_engine.engine.c_dma as c_dma_module
        import algo_engine.engine.c_dma_algo as c_dma_algo_module

        # The algo module seeds its process engine from the access, and the
        # access module exports no engine of its own.
        self.assertIs(c_dma_algo_module.ALGO_ENGINE, DMA.algo_engine)
        self.assertIs(c_dma_algo_module.ALGO_REGISTRY, DMA.algo_engine.registry)
        self.assertFalse(hasattr(c_dma_module, 'ALGO_ENGINE'))
        self.assertFalse(hasattr(c_dma_module, 'ALGO_REGISTRY'))

        # A fresh interpreter whose first import is the algo module still
        # resolves the access's engine through the circular cimport.
        probe = (
            'from algo_engine.engine.c_dma_algo import ALGO_ENGINE;'
            'from algo_engine.engine.c_dma import DMA;'
            'print(DMA.algo_engine is ALGO_ENGINE)'
        )
        result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], 'True')


if __name__ == '__main__':
    unittest.main()
