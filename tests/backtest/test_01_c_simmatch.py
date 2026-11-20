import logging
import time
import unittest
from datetime import datetime

from event_engine.capi import EventEngineEx

from algo_engine.backtest import SimMatch
from algo_engine.base import (
    BarData,
    OrderData,
    OrderState,
    OrderType,
    TickData,
    TickDataLite,
    TradeInstruction,
    TradeReport,
    TransactionData,
    TransactionSide,
)
from algo_engine.engine import TOPIC_SET
from algo_engine.exchange_profile import PROFILE

LOGGER = logging.getLogger(__name__)


class RecordingEngine(EventEngineEx):
    """EventEngineEx test double: records publications and bindings."""

    def __init__(self):
        self.published = []
        self.bound = []
        self.unbound = []

    def put(self, topic, *args, **kwargs):
        self.published.append((topic, kwargs))

    def register_handler(self, topic, handler, deduplicate=False):
        self.bound.append((topic, handler))

    def unregister_handler(self, topic, handler):
        self.unbound.append((topic, handler))
        self.bound = [(bound_topic, bound_handler) for bound_topic, bound_handler in self.bound
                      if not (bound_topic == topic and bound_handler == handler)]


class FailingEngine(RecordingEngine):
    """RecordingEngine whose second handler registration fails."""

    def __init__(self):
        super().__init__()
        self.register_attempts = 0

    def register_handler(self, topic, handler, deduplicate=False):
        self.register_attempts += 1
        if self.register_attempts == 2:
            raise RuntimeError('bind failed')
        super().register_handler(topic, handler, deduplicate)


class Xorshift64Star:
    """Independent oracle of the C-layer PRNG (xorshift64*).

    Mirrors c_smm_rng_seed / c_smm_rng_next_raw / c_smm_rng_next in
    c_simmatch.h: state is initialized to the seed verbatim (nonzero) and
    each draw advances x ^= x<<13; x ^= x>>7; x ^= x<<17 before multiplying
    by 0x2545F4914F6CDD1D; the double is (raw >> 11) / 2**53.
    """
    MASK64 = (1 << 64) - 1
    MULT = 0x2545F4914F6CDD1D

    def __init__(self, seed: int):
        self.state = seed & self.MASK64

    def next_raw(self) -> int:
        x = self.state
        x ^= (x << 13) & self.MASK64
        x ^= x >> 7
        x ^= (x << 17) & self.MASK64
        self.state = x
        return (x * self.MULT) & self.MASK64

    def next(self) -> float:
        return (self.next_raw() >> 11) * (1.0 / 9007199254740992.0)


class TestSimMatch(unittest.TestCase):
    """Contract: SimMatch is the C-layer simulation matcher.

    Expected behavior (oracle: the legacy pure-Python SimMatch semantics,
    with the documented differences that the matcher owns a copy of every
    launched order, filled orders leave ``working`` immediately, and each
    placement / cancel / fill publishes one event):

        - launch_order copies the order into the registry; the launched object
          is never modified. ``working`` and ``history`` are read-only views
          over the native registries, not Python-side mirrors.
        - bar data matches buy orders at high/limit/vwap and sell orders at
          low/limit/vwap.
        - tick data accumulates order-book volume up to the working volume.
        - tick-lite / transaction data fill at their single price level.
        - order data is matched only in incremental_order_volume mode (SH
          style); SZ-style depth reports are ignored by default.
        - lag (time / transaction count), hit probability (seeded xorshift64*)
          and slippage gate every fill; match price is clamped to the limit.
        - cancel / eod / clear behave like the Python version; listeners
          survive clear() and multiple native listeners can be bound.
    """

    def setUp(self) -> None:
        self.event_engine = RecordingEngine()
        self.topic_set = TOPIC_SET
        self.sim = SimMatch(
            ticker='TEST',
            event_engine=self.event_engine,
            topic_set=self.topic_set,
            fee_rate=0.001,
            hit_prob=1.0,
            instant_fill=False,
        )
        self.base_order = {
            'ticker': 'TEST',
            'multiplier': 1.0,
            'order_type': OrderType.ORDER_LIMIT,
        }

    def matcher_order(self, order, sim=None):
        sim = self.sim if sim is None else sim
        working = sim.working.get(order.order_id)
        if working is not None:
            return working
        return sim.history[order.order_id]

    def create_order(self, side=TransactionSide.SIDE_BID, limit_price=50.0, timestamp=None, volume=100):
        return TradeInstruction(
            side=side,
            limit_price=limit_price,
            timestamp=time.time() if timestamp is None else timestamp,
            volume=volume,
            **self.base_order,
        )

    def create_tick(self, timestamp=None, **kwargs):
        defaults = dict(
            ticker='TEST',
            timestamp=time.time() if timestamp is None else timestamp,
            last_price=49.5,
        )
        defaults.update(kwargs)
        return TickData(**defaults)

    def create_bar(self, timestamp=None, high=49.5, low=49.0, close=49.3, volume=1000, notional=49200):
        return BarData(
            ticker='TEST',
            timestamp=time.time() if timestamp is None else timestamp,
            high_price=high, low_price=low, open_price=49.2, close_price=close,
            volume=volume, notional=notional, bar_span=60,
        )

    # === basic lifecycle ===

    def test_00_initial_state(self) -> None:
        """Fresh matcher has empty registries and default market state."""
        self.assertEqual(self.sim.working, {})
        self.assertEqual(self.sim.history, {})
        self.assertEqual(self.sim.timestamp, 0.)
        self.assertIsNone(self.sim.last_price)
        self.assertIsNone(self.sim.last_bid_price)
        self.assertIsNone(self.sim.last_ask_price)
        self.assertEqual(self.sim.last_transaction_count, 0)
        self.assertIsInstance(self.sim.market_time, datetime)
        self.assertIsInstance(self.sim.seed, int)
        config = self.sim.matching_config
        self.assertEqual(config['fee_rate'], 0.001)
        self.assertFalse(config['instant_fill'])
        self.assertEqual(config['lag'], {'ts': 0., 'n_transaction': 0})
        self.assertEqual(config['hit'], {'prob': 1.0, 'slippery': 0.0001})

    def test_01_launch_order_placed(self) -> None:
        """Launching an order copies it, marks the copy PLACED and works it."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.assertIn(order.order_id, self.sim.working)

        working = self.matcher_order(order)
        self.assertIsNot(working, order)
        self.assertIs(working, self.sim.working[order.order_id])
        self.assertEqual(working.order_state, OrderState.STATE_PLACED)
        self.assertEqual(working.placed_ts, self.sim.timestamp)
        self.assertEqual(working.working_volume, 100)
        self.assertEqual(len(self.event_engine.published), 1)
        self.assertEqual(order.order_state, OrderState.STATE_IDLE)

    def test_02_duplicate_launch_raises(self) -> None:
        """Launching the same order twice raises ValueError."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        with self.assertRaises(ValueError):
            self.sim.launch_order(order=order)
        self.assertEqual(len(self.sim.working), 1)

    def test_03_cancel_order(self) -> None:
        """Canceling a working order marks it CANCELED and moves it to history."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.sim.cancel_order(order=order)
        self.assertNotIn(order.order_id, self.sim.working)
        self.assertIn(order.order_id, self.sim.history)
        canceled = self.matcher_order(order)
        self.assertEqual(canceled.order_state, OrderState.STATE_CANCELED)
        self.assertEqual(canceled.canceled_ts, self.sim.timestamp)

    def test_04_cancel_by_order_id(self) -> None:
        """Cancel accepts an order_id alone."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.sim.cancel_order(order_id=order.order_id)
        self.assertIn(order.order_id, self.sim.history)
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_CANCELED)

    def test_05_cancel_unknown_order(self) -> None:
        """Canceling an unknown order is a silent no-op."""
        order = self.create_order()
        self.sim.cancel_order(order_id=order.order_id)
        self.assertEqual(self.sim.working, {})
        self.assertEqual(self.sim.history, {})

    def test_06_cancel_requires_argument(self) -> None:
        """cancel_order without order or order_id raises ValueError."""
        with self.assertRaises(ValueError):
            self.sim.cancel_order()

    # === bar matching ===

    def test_07_bar_fill_buy(self) -> None:
        """Buy order fills at the bar high when high < limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        bar = self.create_bar()
        self.sim(market_data=bar)

        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 100)
        expected_price = 49.5 * (1 + 0.0001)
        self.assertAlmostEqual(filled.average_price, expected_price)
        self.assertEqual(len(filled.trades), 1)
        report = next(iter(filled.trades.values()))
        self.assertEqual(report.volume, 100)
        self.assertAlmostEqual(report.price, expected_price)
        self.assertAlmostEqual(report.notional, 100 * expected_price)
        self.assertAlmostEqual(report.fee, 0.001 * 100 * expected_price)
        self.assertAlmostEqual(filled.fee, report.fee)
        self.assertNotIn(order.order_id, self.sim.working)
        self.assertIn(order.order_id, self.sim.history)
        self.assertEqual(len(self.event_engine.published), 3)

    def test_08_bar_fill_sell(self) -> None:
        """Sell order fills at the bar low when low > limit."""
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0)
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.8, low=50.5, close=50.7, notional=50700)
        self.sim(market_data=bar)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        expected_price = 50.5 * (1 - 0.0001)
        self.assertAlmostEqual(filled.average_price, expected_price)

    def test_09_bar_no_fill(self) -> None:
        """Buy order does not fill when the bar stays above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.5, low=50.2, close=50.4, notional=50400)
        self.sim(market_data=bar)
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_PLACED)
        self.assertIn(order.order_id, self.sim.working)

    def test_10_bar_market_order_vwap(self) -> None:
        """Market order fills at the bar VWAP."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.5, low=49.5, close=50.2, notional=50000)
        self.sim(market_data=bar)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        expected_price = 50.0 * (1 + 0.0001)
        self.assertAlmostEqual(filled.average_price, expected_price)

    # === tick matching ===

    def test_11_tick_full_fill(self) -> None:
        """Buy order fully fills at the best ask level."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 100)
        self.assertAlmostEqual(filled.average_price, 49.9 * (1 + 0.0001))

    def test_12_tick_partial_fill(self) -> None:
        """Order partially fills and keeps the remainder working."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=60)
        self.sim(market_data=tick)
        partial = self.matcher_order(order)
        self.assertEqual(partial.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(partial.filled_volume, 60)
        self.assertEqual(partial.working_volume, 40)
        self.assertIn(order.order_id, self.sim.working)

        tick2 = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick2)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 100)
        self.assertNotIn(order.order_id, self.sim.working)

    def test_13_tick_multi_level_accumulate(self) -> None:
        """Order book volume accumulates across ask levels up to the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0, volume=150)
        self.sim.launch_order(order=order)
        tick = self.create_tick(
            ask_price_1=49.9, ask_volume_1=100,
            ask_price_2=50.0, ask_volume_2=100,
        )
        self.sim(market_data=tick)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 150)
        avg = (100 * 49.9 + 50 * 50.0) / 150
        self.assertAlmostEqual(filled.average_price, avg * (1 + 0.0001))

    def test_14_tick_market_order(self) -> None:
        """Market buy order fills from the ask book."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.9 * (1 + 0.0001))

    def test_15_tick_no_fill_price(self) -> None:
        """No fill when the best ask is above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=50.5, ask_volume_1=100)
        self.sim(market_data=tick)
        working = self.matcher_order(order)
        self.assertEqual(working.order_state, OrderState.STATE_PLACED)
        self.assertEqual(working.filled_volume, 0)

    # === tick-lite matching ===

    def test_16_tick_lite_fill(self) -> None:
        """Buy order fills at the lite ask price when ask <= limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        lite = TickDataLite(
            ticker='TEST', timestamp=time.time(), last_price=49.5,
            bid_price=49.0, bid_volume=100, ask_price=49.9, ask_volume=100,
        )
        self.sim(market_data=lite)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.9 * (1 + 0.0001))

    def test_17_tick_lite_sell_fill(self) -> None:
        """Sell order fills at the lite bid price when bid >= limit."""
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0)
        self.sim.launch_order(order=order)
        lite = TickDataLite(
            ticker='TEST', timestamp=time.time(), last_price=50.5,
            bid_price=50.5, bid_volume=100, ask_price=50.6, ask_volume=100,
        )
        self.sim(market_data=lite)
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 50.5 * (1 - 0.0001))

    def test_18_tick_lite_no_fill(self) -> None:
        """No fill when the lite ask is above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        lite = TickDataLite(
            ticker='TEST', timestamp=time.time(), last_price=50.5,
            bid_price=50.4, bid_volume=100, ask_price=50.6, ask_volume=100,
        )
        self.sim(market_data=lite)
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_PLACED)

    # === transaction matching ===

    def test_19_transaction_fill(self) -> None:
        """Buy order fills at the transaction price when price < limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        txn = TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=30, side=TransactionSide.SIDE_LONG,
        )
        self.sim(market_data=txn)
        partial = self.matcher_order(order)
        self.assertEqual(partial.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(partial.filled_volume, 30)
        report = next(iter(partial.trades.values()))
        self.assertAlmostEqual(report.price, 49.5 * (1 + 0.0001))
        self.assertEqual(self.sim.last_transaction_count, 1)

    def test_20_transaction_market_order_side_copy(self) -> None:
        """Market order copies the next transaction with the same sign."""
        buy_order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=buy_order)
        txn_long = TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=30, side=TransactionSide.SIDE_LONG,
        )
        self.sim(market_data=txn_long)
        self.assertEqual(self.matcher_order(buy_order).order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(self.matcher_order(buy_order).filled_volume, 30)

        sell_order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=float('nan'))
        self.sim.launch_order(order=sell_order)
        txn_short = TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=50.5, volume=40, side=TransactionSide.SIDE_SHORT,
        )
        self.sim(market_data=txn_short)
        self.assertEqual(self.matcher_order(sell_order).order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(self.matcher_order(sell_order).filled_volume, 40)

    # === order data matching ===

    def test_21_order_data_ignored_by_default(self) -> None:
        """Order data is not matched unless incremental_order_volume is set.

        SZ-style depth reports carry resting volume, not incremental fills,
        so the default (False) must not treat them as matchable volume.
        """
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        od = OrderData(
            ticker='TEST', timestamp=time.time(),
            price=49.9, volume=50, side=TransactionSide.SIDE_ASK,
        )
        self.sim(market_data=od)
        working = self.matcher_order(order)
        self.assertEqual(working.order_state, OrderState.STATE_PLACED)
        self.assertEqual(working.filled_volume, 0)
        self.assertIn(order.order_id, self.sim.working)
        self.assertFalse(self.sim.matching_config['incremental_order_volume'])

    def test_22_order_data_fill_incremental(self) -> None:
        """With incremental_order_volume, a buy order fills against an
        OrderData print at price <= limit."""
        sim = SimMatch(
            ticker='TEST',
            event_engine=self.event_engine,
            topic_set=self.topic_set,
            incremental_order_volume=True,
            hit_prob=1.,
        )
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        od = OrderData(
            ticker='TEST', timestamp=time.time(),
            price=49.9, volume=50, side=TransactionSide.SIDE_ASK,
        )
        sim(market_data=od)
        partial = self.matcher_order(order, sim)
        self.assertEqual(partial.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(partial.filled_volume, 50)
        report = next(iter(partial.trades.values()))
        self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))

    # === instant fill ===

    def _instant_fill_sim(self):
        return SimMatch(
            ticker='TEST',
            event_engine=self.event_engine,
            topic_set=self.topic_set,
            instant_fill=True,
            hit_prob=1.,
        )

    def test_23_instant_fill_buy(self) -> None:
        """With instant_fill, a buy order fills at the worst of limit/last."""
        sim = self._instant_fill_sim()
        sim(market_data=self.create_tick(ask_price_1=49.0, ask_volume_1=1))
        self.assertEqual(sim.last_price, 49.5)

        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 50.0)
        self.assertNotIn(order.order_id, sim.working)

    def test_24_instant_fill_market_order(self) -> None:
        """With instant_fill, a market buy fills at the ask."""
        sim = self._instant_fill_sim()
        sim(market_data=self.create_tick(ask_price_1=49.0, ask_volume_1=1))
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        sim.launch_order(order=order)
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.0 * (1 + 0.0001))

    def test_25_instant_fill_requires_last_price(self) -> None:
        """Without a last price, an instant-fill market order stays working."""
        sim = self._instant_fill_sim()
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        sim.launch_order(order=order)
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)
        self.assertIn(order.order_id, sim.working)

    # === lag ===

    def test_26_lag_ts(self) -> None:
        """Time lag delays the fill until the elapsed time reaches lag_ts."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            lag_ts=5., hit_prob=1.,
        )
        order = self.create_order(timestamp=100.)
        sim.launch_order(order=order)
        sim(market_data=self.create_bar(timestamp=103.))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)
        self.assertEqual(self.matcher_order(order, sim).filled_volume, 0)

        sim(market_data=self.create_bar(timestamp=105.))
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 100)

    def test_27_lag_n_transaction(self) -> None:
        """Transaction lag delays the fill until enough transactions pass."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            lag_n_transaction=2, hit_prob=1.,
        )
        order = self.create_order()
        sim.launch_order(order=order)
        sim(market_data=TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=10, side=TransactionSide.SIDE_LONG,
        ))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)
        sim(market_data=TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=10, side=TransactionSide.SIDE_LONG,
        ))
        partial = self.matcher_order(order, sim)
        self.assertEqual(partial.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(partial.filled_volume, 10)

    # === hit probability ===

    def test_28_hit_probability_zero(self) -> None:
        """hit_prob=0 never fills."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0., seed=1,
        )
        order = self.create_order()
        sim.launch_order(order=order)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        working = self.matcher_order(order, sim)
        self.assertEqual(working.order_state, OrderState.STATE_PLACED)
        self.assertEqual(working.filled_volume, 0)

    def test_29_hit_probability_seeded_pattern(self) -> None:
        """With a fixed seed the fill pattern matches the xorshift64* oracle."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0.5, seed=42,
        )
        orders = [self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0) for _ in range(10)]
        for order in orders:
            sim.launch_order(order=order)

        oracle = Xorshift64Star(42)
        expected = []
        for _ in range(10):
            draw_hit = oracle.next() < 0.5
            if draw_hit:
                oracle.next_raw()
                oracle.next_raw()
            expected.append(draw_hit)

        sim(market_data=self.create_bar())

        for order, draw_hit in zip(reversed(orders), expected):
            if draw_hit:
                self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED,
                                 f'order {order.order_id} should have been filled')
            else:
                self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED,
                                 f'order {order.order_id} should not have been filled')

    # === slippage / fee / clamp ===

    def test_30_slippage_and_fee(self) -> None:
        """Slippage is applied per side and fee accumulates on the order."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            fee_rate=0.001, slippery_rate=0.0001, hit_prob=1.,
        )
        buy = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0, volume=50)
        sim.launch_order(order=buy)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=50))
        self.assertAlmostEqual(self.matcher_order(buy, sim).average_price, 49.9 * 1.0001)

        sell = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0, volume=50)
        sim.launch_order(order=sell)
        sim(market_data=self.create_tick(bid_price_1=50.5, bid_volume_1=50, last_price=50.4))
        self.assertAlmostEqual(self.matcher_order(sell, sim).average_price, 50.5 * 0.9999)

        filled_buy = self.matcher_order(buy, sim)
        report = next(iter(filled_buy.trades.values()))
        self.assertAlmostEqual(report.fee, 0.001 * report.notional)
        self.assertAlmostEqual(filled_buy.fee, report.fee)

    def test_31_limit_clamp(self) -> None:
        """Match price never exceeds the limit after slippage."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            slippery_rate=0.1, hit_prob=1.,
        )
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 50.0)

    # === lifecycle ===

    def test_32_eod_cancels_all(self) -> None:
        """eod cancels every working order."""
        orders = [self.create_order() for _ in range(3)]
        for order in orders:
            self.sim.launch_order(order=order)
        self.sim.eod()
        self.assertEqual(self.sim.working, {})
        for order in orders:
            self.assertIn(order.order_id, self.sim.history)
            self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_CANCELED)

    def test_33_clear_resets(self) -> None:
        """clear resets state but keeps the matching config."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(len(self.sim.history), 1)
        self.sim.clear()
        self.assertEqual(self.sim.working, {})
        self.assertEqual(self.sim.history, {})
        self.assertEqual(self.sim.timestamp, 0.)
        self.assertIsNone(self.sim.last_price)
        self.assertIsNone(self.sim.last_bid_price)
        self.assertIsNone(self.sim.last_ask_price)
        self.assertEqual(self.sim.last_transaction_count, 0)
        self.assertEqual(self.sim.matching_config['fee_rate'], 0.001)

    # === static price helpers ===

    def test_34_best_worst_price(self) -> None:
        """best/worst price follow the side sign and skip invalid values."""
        self.assertEqual(
            SimMatch.best_price(1., 2., 3., side=TransactionSide.SIDE_BID), 1.)
        self.assertEqual(
            SimMatch.worst_price(1., 2., 3., side=TransactionSide.SIDE_BID), 3.)
        self.assertEqual(
            SimMatch.best_price(1., 2., 3., side=TransactionSide.SIDE_ASK), 3.)
        self.assertEqual(
            SimMatch.worst_price(1., 2., 3., side=TransactionSide.SIDE_ASK), 1.)
        self.assertEqual(
            SimMatch.best_price(None, float('nan'), 2., side=TransactionSide.SIDE_BID), 2.)
        with self.assertRaises(ValueError):
            SimMatch.best_price(side=TransactionSide.SIDE_BID)
        with self.assertRaises(ValueError):
            SimMatch.best_price(1., 2., side=TransactionSide.SIDE_UNKNOWN)

    # === market_time ===

    def test_35_market_time(self) -> None:
        """market_time mirrors the profile conversion of the timestamp."""
        self.sim.timestamp = 1735689600.
        self.assertEqual(self.sim.market_time, PROFILE.timestamp_to_datetime(1735689600.))

    # === event engine C-API binding ===

    def test_36_register_unregister_capi(self) -> None:
        """register binds order handlers plus a DMAProtocol realtime callback."""
        engine = EventEngineEx()
        sim = SimMatch(ticker='TEST', event_engine=engine, topic_set=TOPIC_SET)
        sim.register()
        self.addCleanup(engine.stop)
        self.assertIsNotNone(sim.protocol)

        order = self.create_order()
        engine.start()
        try:
            engine.put(topic=TOPIC_SET.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working),
                            'launch_order handler did not fire')
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)

            engine.put(topic=TOPIC_SET.cancel_order(ticker='TEST'), order_id=order.order_id)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.history),
                            'cancel_order handler did not fire')
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_CANCELED)
        finally:
            engine.stop()

        sim.unregister()
        self.assertIsNone(sim.protocol)

        from algo_engine.engine import DMAProtocol

        order2 = self.create_order()
        engine.start()
        try:
            engine.put(topic=TOPIC_SET.launch_order(ticker='TEST'), order=order2)
            self.assertFalse(self._wait_until(lambda: order2.order_id in sim.working, timeout=1.),
                             'launch handler still fired after unregister')

            sim.launch_order(order2)
            sender = DMAProtocol(engine, TOPIC_SET)
            sender.send(self.create_tick(ask_price_1=49.9, ask_volume_1=100))
            time.sleep(0.2)
            self.assertEqual(self.matcher_order(order2, sim).order_state, OrderState.STATE_PLACED)
        finally:
            engine.stop()

    def test_37_realtime_dispatch_capi(self) -> None:
        """Market data routed through the realtime topic fills working orders."""
        engine = EventEngineEx()
        sim = SimMatch(ticker='TEST', event_engine=engine, topic_set=TOPIC_SET)
        sim.register()
        self.addCleanup(sim.unregister)
        self.addCleanup(engine.stop)

        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        engine.start()
        try:
            engine.put(topic=TOPIC_SET.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working))
            tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
            sim.protocol.send(tick)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.history),
                            'realtime callback did not fill the order')
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED)
        finally:
            engine.stop()

    def test_38_multi_listener_capi(self) -> None:
        """A second native listener receives match and cancel events and can
        be deregistered by id."""
        from tests.backtest import c_simmatch_toolkit

        events = []
        sim = SimMatch(ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set, hit_prob=1.)
        listener_id = c_simmatch_toolkit.register_listener(
            sim, lambda event, order_id, report: events.append((event, order_id, report)))

        order = self.create_order()
        sim.launch_order(order=order)
        self.assertEqual(len(events), 1)
        event, order_id, report = events[0]
        self.assertEqual(event, 2)  # SMM_EVENT_PLACED
        self.assertEqual(order_id, order.order_id)
        self.assertIsNone(report)

        sim.cancel_order(order=order)
        self.assertEqual(len(events), 2)
        event, order_id, report = events[1]
        self.assertEqual(event, 1)  # SMM_EVENT_CANCEL
        self.assertEqual(order_id, order.order_id)
        self.assertIsNone(report)

        order2 = self.create_order()
        sim.launch_order(order=order2)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(len(events), 4)
        event, order_id, report = events[3]
        self.assertEqual(event, 0)  # SMM_EVENT_MATCH
        self.assertEqual(order_id, order2.order_id)
        self.assertIsInstance(report, TradeReport)
        self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))

        c_simmatch_toolkit.deregister_listener(sim, listener_id)
        order3 = self.create_order()
        sim.launch_order(order=order3)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(len(events), 4)

    def test_39_clear_keeps_listener(self) -> None:
        """clear() keeps the bound listener; fills still sync after reset."""
        self.sim.clear()
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_FILLED)
        self.assertNotIn(order.order_id, self.sim.working)
        self.assertIn(order.order_id, self.sim.history)
        self.assertEqual(len(self.event_engine.published), 3)

    def test_40_native_publish_capi(self) -> None:
        """Registered on a real engine, fills publish to on_report / on_order
        from the matcher's listener."""
        engine = EventEngineEx()
        reports = []
        orders = []
        engine.register_handler(topic=TOPIC_SET.on_report, handler=lambda **kw: reports.append(kw['report']))
        engine.register_handler(topic=TOPIC_SET.on_order, handler=lambda **kw: orders.append(kw['order']))

        sim = SimMatch(ticker='TEST', event_engine=engine, topic_set=TOPIC_SET, hit_prob=1.)
        sim.register()
        self.addCleanup(engine.stop)

        protocol = sim.protocol
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        engine.start()
        try:
            engine.put(topic=TOPIC_SET.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working))
            self.assertTrue(self._wait_until(lambda: len(orders) >= 1),
                            'placed on_order not published')

            tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
            protocol.send(tick)
            self.assertTrue(self._wait_until(lambda: len(reports) >= 1),
                            'on_report not published')
            self.assertEqual(len(orders), 2)

            report = reports[0]
            self.assertIsInstance(report, TradeReport)
            self.assertEqual(report.volume, 100)
            self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))
            # Both events carry a view over the registry record; the fill moves
            # the order to history, which replaces the cached view object.
            self.assertEqual(orders[0].order_id, order.order_id)
            self.assertEqual(orders[1].order_id, order.order_id)
            self.assertIs(orders[1], sim.history[order.order_id])
            self.assertIsNot(orders[1], order)
        finally:
            engine.stop()

    # === order ring registry ===

    def test_41_registry_grows_past_initial_capacity(self) -> None:
        """The contiguous registry grows when it fills up and keeps every order."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=1.,
        )
        orders = [self.create_order(volume=10) for _ in range(80)]
        for order in orders:
            sim.launch_order(order)
        self.assertEqual(len(sim.working), len(orders))

        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertEqual(sim.working, {})
        self.assertEqual(len(sim.history), len(orders))
        for order in orders:
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED)

    def test_42_cancel_middle_slot_keeps_other_orders(self) -> None:
        """Canceling non-adjacent slots leaves the other orders matchable."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=1.,
        )
        orders = [self.create_order(volume=10) for _ in range(48)]
        for order in orders:
            sim.launch_order(order)

        cancelled = [order for index, order in enumerate(orders) if index % 2 == 0]
        kept = [order for index, order in enumerate(orders) if index % 2 == 1]
        for order in cancelled:
            sim.cancel_order(order=order)

        self.assertEqual(len(sim.working), len(kept))
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertEqual(sim.working, {})
        self.assertEqual(len(sim.history), len(orders))
        for order in kept:
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED)
        for order in cancelled:
            self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_CANCELED)

    def test_43_clear_reuses_the_expanded_registry(self) -> None:
        """clear keeps the expanded ring capacity and the matcher keeps working."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=1.,
        )
        for _ in range(80):
            sim.launch_order(self.create_order(volume=10))
        sim.clear()
        self.assertEqual(sim.working, {})

        order = self.create_order(volume=10)
        sim.launch_order(order)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED)
        self.assertEqual(sim.history[order.order_id].order_id, order.order_id)

    def test_44_ccp_binding_survives_repeated_alloc_and_release(self) -> None:
        """The CCP binding husks cleanly over repeated construct/destroy cycles."""
        import gc

        sim = SimMatch(ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set)
        self.assertNotEqual(sim.address, 'NULL')
        del sim
        gc.collect()

        for _ in range(50):
            sim = SimMatch(ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set)
            sim.launch_order(self.create_order(volume=10))
            sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
            del sim
        gc.collect()

    # === refactor contracts ===

    def test_45_price_helpers_reject_empty_input(self) -> None:
        """best/worst_price raise ValueError before allocating an empty array."""
        with self.assertRaises(ValueError):
            SimMatch.best_price(side=TransactionSide.SIDE_BID)
        with self.assertRaises(ValueError):
            SimMatch.worst_price(side=TransactionSide.SIDE_ASK)
        self.assertEqual(
            SimMatch.best_price(None, float('inf'), 2., side=TransactionSide.SIDE_BID), 2.)

    def test_46_infinite_limit_is_market_like_on_bar(self) -> None:
        """An infinite limit price constrains no price: the order fills at VWAP."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('inf'))
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_bar(high=50.5, low=49.5, close=50.2, notional=50000))
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 50.0 * (1 + 0.0001))

    def test_47_infinite_limit_is_market_like_on_tick(self) -> None:
        """An infinite limit price fills from the book like a market order."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('inf'))
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.9 * (1 + 0.0001))

    def test_48_zero_volume_bar_does_not_fill_at_nan(self) -> None:
        """A market order stays working on a zero-volume bar (no NaN report)."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_bar(volume=0, notional=0))
        working = self.matcher_order(order)
        self.assertEqual(working.order_state, OrderState.STATE_PLACED)
        self.assertEqual(working.filled_volume, 0)
        self.assertEqual(working.trades, {})
        self.assertIn(order.order_id, self.sim.working)

    def test_49_repr_names_the_ticker(self) -> None:
        """repr identifies the instance and its ticker."""
        text = repr(self.sim)
        self.assertIn('SimMatch', text)
        self.assertIn('TEST', text)

    def test_50_registry_tracks_working_and_history(self) -> None:
        """The C registries back the Python views and scan newest-first."""
        from tests.backtest import c_simmatch_toolkit

        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (0, 0))

        orders = [self.create_order(volume=10) for _ in range(10)]
        for order in orders:
            self.sim.launch_order(order)
        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (10, 0))
        self.assertEqual(len(self.sim.working), 10)
        self.assertEqual(c_simmatch_toolkit.working_order_ids(self.sim),
                         [order.order_id for order in reversed(orders)])

        canceled = orders[::2]
        for order in canceled:
            self.sim.cancel_order(order=order)
        kept = orders[1::2]
        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (5, 5))
        self.assertEqual(c_simmatch_toolkit.working_order_ids(self.sim),
                         [order.order_id for order in reversed(kept)])

        self.sim.clear()
        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (0, 0))
        self.assertEqual(self.sim.working, {})
        self.assertEqual(self.sim.history, {})

    def test_51_registry_grows_past_any_fixed_capacity(self) -> None:
        """The registry holds more orders than any fixed initial capacity."""
        from tests.backtest import c_simmatch_toolkit

        orders = [self.create_order(volume=10) for _ in range(200)]
        for order in orders:
            self.sim.launch_order(order)

        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (200, 0))
        self.assertEqual(len(self.sim.working), 200)

        for order in orders[::2]:
            self.sim.cancel_order(order=order)
        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (100, 100))

        self.sim.clear()
        self.assertEqual(c_simmatch_toolkit.registry_state(self.sim), (0, 0))

        # The cleared registries keep working after growth.
        replacement = self.create_order(volume=10)
        self.sim.launch_order(replacement)
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertIn(replacement.order_id, self.sim.history)

    def test_52_adopted_wrapper_shares_the_context(self) -> None:
        """c_from_header rebuilds a non-owning wrapper over the same context."""
        from tests.backtest import c_simmatch_toolkit

        order = self.create_order(volume=10)
        self.sim.launch_order(order)
        adopted = c_simmatch_toolkit.adopt(self.sim)

        self.assertEqual(adopted.ticker, self.sim.ticker)
        self.assertEqual(adopted.seed, self.sim.seed)
        self.assertEqual(set(adopted.working), set(self.sim.working))

        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertIn(order.order_id, self.sim.history)
        self.assertIn(order.order_id, adopted.history)

        del adopted
        # The adopted wrapper never owned the context; the owner still works.
        self.sim.launch_order(self.create_order(volume=10))
        self.assertEqual(len(self.sim.working), 1)

    def test_53_seed_is_read_only_and_reproducible(self) -> None:
        """The seed is read-only and a same-seed construction repeats the pattern."""
        first = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0.5, seed=123,
        )
        second = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0.5, seed=123,
        )

        def run(sim):
            orders = [self.create_order(volume=10) for _ in range(8)]
            for order in orders:
                sim.launch_order(order)
            sim(market_data=self.create_bar())
            return [self.matcher_order(order, sim).order_state for order in orders]

        first_pattern = run(first)
        second_pattern = run(second)
        self.assertEqual(first_pattern, second_pattern)
        self.assertIn(OrderState.STATE_FILLED, first_pattern)
        self.assertEqual(first.seed, 123)

        with self.assertRaises(AttributeError):
            first.seed = 5

    def test_54_relaunch_of_a_filled_order_raises(self) -> None:
        """A filled order id stays in history and cannot be relaunched."""
        order = self.create_order(volume=10)
        self.sim.launch_order(order=order)
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_FILLED)
        with self.assertRaises(ValueError):
            self.sim.launch_order(order=order)

    def test_55_lag_count_underflow_does_not_fill(self) -> None:
        """A transaction count moving backwards must not satisfy the lag."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            lag_n_transaction=2, hit_prob=1.,
        )
        sim.last_transaction_count = 10
        order = self.create_order()
        sim.launch_order(order=order)

        sim.last_transaction_count = 0
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)

        sim.last_transaction_count = 12
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_FILLED)

    def test_56_register_rolls_back_partial_binding(self) -> None:
        """A failed registration undoes the handlers bound before it."""
        engine = FailingEngine()
        sim = SimMatch(ticker='TEST', event_engine=engine, topic_set=self.topic_set)

        with self.assertRaises(RuntimeError):
            sim.register()

        self.assertIsNone(sim.protocol)
        self.assertEqual(engine.register_attempts, 2)
        self.assertEqual(len(engine.unbound), 1)
        self.assertEqual(engine.unbound[0][1], sim.launch_order)

        sim.unregister()
        self.assertEqual(len(engine.unbound), 1)

    def test_57_launch_copies_the_order(self) -> None:
        """The matcher owns its order state; the launched object is untouched."""
        order = self.create_order(volume=10)
        self.sim.launch_order(order=order)
        self.assertIsNot(self.sim.working[order.order_id], order)

        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=10))
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertEqual(filled.filled_volume, 10)
        self.assertEqual(len(filled.trades), 1)

        self.assertEqual(order.order_state, OrderState.STATE_IDLE)
        self.assertEqual(order.filled_volume, 0)
        self.assertEqual(order.trades, {})

    def test_58_constructor_type_enforcement(self) -> None:
        """The engine and topic set are typed at the Cython boundary."""
        with self.assertRaises(TypeError):
            SimMatch(ticker='TEST', event_engine=object())
        with self.assertRaises(TypeError):
            SimMatch(ticker='TEST', topic_set=object())
        with self.assertRaises(OverflowError):
            SimMatch(ticker='TEST', seed=-1)

    # === last-price provenance ===

    def test_59_order_data_does_not_set_last_price(self) -> None:
        """Order data carries its own limit price, not a market print."""
        self.sim(market_data=self._order_data(120.0, TransactionSide.SIDE_ASK))
        self.assertIsNone(self.sim.last_price)

    def test_60_order_data_does_not_overwrite_last_price(self) -> None:
        """A resting order never replaces an established traded price."""
        self.sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=1))
        self.assertEqual(self.sim.last_price, 49.5)
        self.sim(market_data=self._order_data(120.0, TransactionSide.SIDE_ASK))
        self.assertEqual(self.sim.last_price, 49.5)

    def test_61_tick_nan_last_uses_mid_price(self) -> None:
        """A tick without a last price supplies the bid/ask mid-point."""
        tick = TickData(
            ticker='TEST', timestamp=time.time(), last_price=float('nan'),
            bid_price_1=100.0, bid_volume_1=1.0,
            ask_price_1=102.0, ask_volume_1=1.0,
        )
        self.sim(market_data=tick)
        self.assertEqual(self.sim.last_price, 101.0)

    def test_62_tick_nan_last_uses_present_side(self) -> None:
        """One-sided ticks use the present side (the price-limit case)."""
        bid_only = TickData(
            ticker='TEST', timestamp=time.time(), last_price=float('nan'),
            bid_price_1=99.0, bid_volume_1=1.0,
            ask_price_1=float('nan'), ask_volume_1=0.0,
        )
        self.sim(market_data=bid_only)
        self.assertEqual(self.sim.last_price, 99.0)

        ask_only = TickData(
            ticker='TEST', timestamp=time.time(), last_price=float('nan'),
            bid_price_1=float('nan'), bid_volume_1=0.0,
            ask_price_1=101.0, ask_volume_1=1.0,
        )
        self.sim(market_data=ask_only)
        self.assertEqual(self.sim.last_price, 101.0)

    def test_63_bar_close_sets_last_price(self) -> None:
        """A bar close is a traded price."""
        self.sim(market_data=self.create_bar(close=88.5))
        self.assertEqual(self.sim.last_price, 88.5)

    def test_64_cancel_transaction_does_not_set_last_price(self) -> None:
        """A cancel report is not a trade."""
        cancel = TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=120.0, volume=10, side=TransactionSide.SIDE_LONG_CANCEL,
        )
        self.sim(market_data=cancel)
        self.assertIsNone(self.sim.last_price)

    # === book-side price provenance ===

    def _transaction(self, price, side, volume=10):
        return TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=price, volume=volume, side=side,
        )

    def _order_data(self, price, side, volume=10):
        return OrderData(
            ticker='TEST', timestamp=time.time(),
            price=price, volume=volume, side=side,
        )

    def test_65_bid_transaction_sets_ask_price(self) -> None:
        """A bid-side trade prints at the ask."""
        self.sim(market_data=self._transaction(10.0, TransactionSide.SIDE_LONG))
        self.assertEqual(self.sim.last_price, 10.0)
        self.assertEqual(self.sim.last_ask_price, 10.0)
        self.assertIsNone(self.sim.last_bid_price)

    def test_66_ask_transaction_sets_bid_price(self) -> None:
        """An ask-side trade prints at the bid."""
        self.sim(market_data=self._transaction(10.0, TransactionSide.SIDE_LONG))
        self.sim(market_data=self._transaction(9.0, TransactionSide.SIDE_SHORT))
        self.assertEqual(self.sim.last_price, 9.0)
        self.assertEqual(self.sim.last_bid_price, 9.0)
        self.assertEqual(self.sim.last_ask_price, 10.0)

    def test_67_tick_sets_both_book_sides(self) -> None:
        """A tick refreshes bid and ask from its book."""
        self.sim(market_data=self.create_tick(
            bid_price_1=49.0, bid_volume_1=1.0,
            ask_price_1=49.9, ask_volume_1=1.0,
        ))
        self.assertEqual(self.sim.last_price, 49.5)
        self.assertEqual(self.sim.last_bid_price, 49.0)
        self.assertEqual(self.sim.last_ask_price, 49.9)

    def test_68_order_data_sets_its_own_book_side(self) -> None:
        """A resting order supplies its side, never the last price."""
        self.sim(market_data=self._order_data(48.0, TransactionSide.SIDE_BID))
        self.assertEqual(self.sim.last_bid_price, 48.0)
        self.assertIsNone(self.sim.last_ask_price)
        self.assertIsNone(self.sim.last_price)

        self.sim(market_data=self._order_data(51.0, TransactionSide.SIDE_ASK))
        self.assertEqual(self.sim.last_bid_price, 48.0)
        self.assertEqual(self.sim.last_ask_price, 51.0)
        self.assertIsNone(self.sim.last_price)

    def test_69_limit_buy_ignores_ask_side_transaction(self) -> None:
        """A sell-side print proves the bid, not the ask a buy needs."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        self.sim(market_data=self._transaction(49.0, TransactionSide.SIDE_SHORT))
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_PLACED)

    def test_70_limit_buy_ignores_resting_bid_order(self) -> None:
        """A resting bid does not prove ask liquidity for a buy."""
        sim = SimMatch(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            incremental_order_volume=True, hit_prob=1.,
        )
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        sim(market_data=self._order_data(49.0, TransactionSide.SIDE_BID))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PLACED)

        sim(market_data=self._order_data(49.0, TransactionSide.SIDE_ASK))
        self.assertEqual(self.matcher_order(order, sim).order_state, OrderState.STATE_PARTFILLED)

    def test_71_limit_buy_fills_at_exact_limit_price(self) -> None:
        """A bid-side print at the limit price fills the buy."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        self.sim(market_data=self._transaction(50.0, TransactionSide.SIDE_LONG))
        filled = self.matcher_order(order)
        self.assertEqual(filled.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(filled.filled_volume, 10)

    def test_72_instant_fill_market_sell_uses_bid(self) -> None:
        """With instant_fill, a market sell fills at the bid."""
        sim = self._instant_fill_sim()
        sim(market_data=self.create_tick(
            bid_price_1=49.0, bid_volume_1=1.0,
            ask_price_1=49.9, ask_volume_1=1.0,
        ))
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=float('nan'))
        sim.launch_order(order=order)
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.0 * (1 - 0.0001))

    def test_73_instant_fill_falls_back_to_last_price(self) -> None:
        """A bookless tick leaves the last price as the market reference."""
        sim = self._instant_fill_sim()
        sim(market_data=self.create_tick(last_price=49.5))
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        sim.launch_order(order=order)
        filled = self.matcher_order(order, sim)
        self.assertEqual(filled.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(filled.average_price, 49.5 * (1 + 0.0001))

    def test_74_cancel_transaction_does_not_fill(self) -> None:
        """A cancel report is not a trade and cannot fill a working order."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        self.sim(market_data=self._transaction(
            49.0, TransactionSide.SIDE_LONG_CANCEL, volume=100,
        ))
        self.assertEqual(self.matcher_order(order).order_state, OrderState.STATE_PLACED)
        self.assertEqual(self.matcher_order(order).filled_volume, 0)

    @staticmethod
    def _wait_until(condition, timeout=5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return False


if __name__ == '__main__':
    unittest.main()
