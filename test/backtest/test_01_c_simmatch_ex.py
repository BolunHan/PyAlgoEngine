import logging
import time
import unittest
from datetime import datetime
from unittest.mock import Mock

from event_engine.capi import EventEngineEx

from algo_engine.backtest import SimMatchEx
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
from algo_engine.engine import TOPIC
from algo_engine.exchange_profile import PROFILE

LOGGER = logging.getLogger(__name__)


class Xorshift64Star:
    """Independent oracle of the C-layer PRNG (xorshift64*).

    Mirrors c_smm_rng_seed / c_smm_rng_next_raw / c_smm_rng_next in
    c_simmatch_ex.h: state is initialized to the seed verbatim (nonzero) and
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


class TestSimMatchEx(unittest.TestCase):
    """Contract: SimMatchEx is the C-layer simulation matcher.

    Expected behavior (oracle: the legacy pure-Python SimMatch semantics,
    with the documented differences that filled orders leave ``working``
    immediately and each placement / cancel / fill publishes one event):

        - launch_order registers the order as PLACED and fills it immediately
          when instant_fill is configured and no lag is set.
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
        self.event_engine = Mock()
        self.topic_set = Mock()
        self.sim = SimMatchEx(
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
        self.assertEqual(self.sim.last_transaction_count, 0)
        self.assertIsInstance(self.sim.market_time, datetime)
        config = self.sim.matching_config
        self.assertEqual(config['fee_rate'], 0.001)
        self.assertFalse(config['instant_fill'])
        self.assertEqual(config['lag'], {'ts': 0., 'n_transaction': 0})
        self.assertEqual(config['hit'], {'prob': 1.0, 'slippery': 0.0001})

    def test_01_launch_order_placed(self) -> None:
        """Launching an order marks it PLACED and adds it to working."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.assertIn(order.order_id, self.sim.working)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.placed_ts, self.sim.timestamp)
        self.assertEqual(order.working_volume, 100)
        self.assertEqual(order.transaction_count_at_placement, 0)
        self.assertEqual(self.event_engine.put.call_count, 1)

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
        self.assertEqual(order.order_state, OrderState.STATE_CANCELED)
        self.assertEqual(order.canceled_ts, self.sim.timestamp)

    def test_04_cancel_by_order_id(self) -> None:
        """Cancel accepts an order_id alone."""
        order = self.create_order()
        self.sim.launch_order(order=order)
        self.sim.cancel_order(order_id=order.order_id)
        self.assertIn(order.order_id, self.sim.history)
        self.assertEqual(order.order_state, OrderState.STATE_CANCELED)

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
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        expected_price = 49.5 * (1 + 0.0001)
        self.assertAlmostEqual(order.average_price, expected_price)
        self.assertEqual(len(order.trades), 1)
        report = next(iter(order.trades.values()))
        self.assertEqual(report.volume, 100)
        self.assertAlmostEqual(report.price, expected_price)
        self.assertAlmostEqual(report.notional, 100 * expected_price)
        self.assertAlmostEqual(report.fee, 0.001 * 100 * expected_price)
        self.assertAlmostEqual(order.fee, report.fee)
        self.assertNotIn(order.order_id, self.sim.working)
        self.assertIn(order.order_id, self.sim.history)
        # on_report and on_order are published
        self.assertEqual(self.event_engine.put.call_count, 3)

    def test_08_bar_fill_sell(self) -> None:
        """Sell order fills at the bar low when low > limit."""
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0)
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.8, low=50.5, close=50.7, notional=50700)
        self.sim(market_data=bar)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        expected_price = 50.5 * (1 - 0.0001)
        self.assertAlmostEqual(order.average_price, expected_price)

    def test_09_bar_no_fill(self) -> None:
        """Buy order does not fill when the bar stays above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.5, low=50.2, close=50.4, notional=50400)
        self.sim(market_data=bar)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertIn(order.order_id, self.sim.working)

    def test_10_bar_market_order_vwap(self) -> None:
        """Market order fills at the bar VWAP."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=order)
        bar = self.create_bar(high=50.5, low=49.5, close=50.2, notional=50000)
        self.sim(market_data=bar)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        expected_price = 50.0 * (1 + 0.0001)
        self.assertAlmostEqual(order.average_price, expected_price)

    # === tick matching ===

    def test_11_tick_full_fill(self) -> None:
        """Buy order fully fills at the best ask level."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
        self.assertAlmostEqual(order.average_price, 49.9 * (1 + 0.0001))

    def test_12_tick_partial_fill(self) -> None:
        """Order partially fills and keeps the remainder working."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=60)
        self.sim(market_data=tick)
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 60)
        self.assertEqual(order.working_volume, 40)
        self.assertIn(order.order_id, self.sim.working)
        # remainder fills on the next tick
        tick2 = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick2)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)
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
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 150)
        avg = (100 * 49.9 + 50 * 50.0) / 150
        self.assertAlmostEqual(order.average_price, avg * (1 + 0.0001))

    def test_14_tick_market_order(self) -> None:
        """Market buy order fills from the ask book."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
        self.sim(market_data=tick)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(order.average_price, 49.9 * (1 + 0.0001))

    def test_15_tick_no_fill_price(self) -> None:
        """No fill when the best ask is above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        tick = self.create_tick(ask_price_1=50.5, ask_volume_1=100)
        self.sim(market_data=tick)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

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
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(order.average_price, 49.9 * (1 + 0.0001))

    def test_17_tick_lite_sell_fill(self) -> None:
        """Sell order fills at the lite bid price when bid >= limit."""
        order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0)
        self.sim.launch_order(order=order)
        lite = TickDataLite(
            ticker='TEST', timestamp=time.time(), last_price=50.5,
            bid_price=50.5, bid_volume=100, ask_price=50.6, ask_volume=100,
        )
        self.sim(market_data=lite)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(order.average_price, 50.5 * (1 - 0.0001))

    def test_18_tick_lite_no_fill(self) -> None:
        """No fill when the lite ask is above the limit."""
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        self.sim.launch_order(order=order)
        lite = TickDataLite(
            ticker='TEST', timestamp=time.time(), last_price=50.5,
            bid_price=50.4, bid_volume=100, ask_price=50.6, ask_volume=100,
        )
        self.sim(market_data=lite)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)

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
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 30)
        report = next(iter(order.trades.values()))
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
        self.assertEqual(buy_order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(buy_order.filled_volume, 30)

        sell_order = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=float('nan'))
        self.sim.launch_order(order=sell_order)
        txn_short = TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=50.5, volume=40, side=TransactionSide.SIDE_SHORT,
        )
        self.sim(market_data=txn_short)
        self.assertEqual(sell_order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(sell_order.filled_volume, 40)

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
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)
        self.assertIn(order.order_id, self.sim.working)
        self.assertFalse(self.sim.matching_config['incremental_order_volume'])

    def test_22_order_data_fill_incremental(self) -> None:
        """With incremental_order_volume, a buy order fills against an
        OrderData print at price <= limit."""
        sim = SimMatchEx(
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
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 50)
        report = next(iter(order.trades.values()))
        self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))

    # === instant fill ===

    def _instant_fill_sim(self):
        return SimMatchEx(
            ticker='TEST',
            event_engine=self.event_engine,
            topic_set=self.topic_set,
            instant_fill=True,
            hit_prob=1.,
        )

    def test_23_instant_fill_buy(self) -> None:
        """With instant_fill, a buy order fills at the worst of limit/last."""
        sim = self._instant_fill_sim()
        # prime the last price
        sim(market_data=self.create_tick(ask_price_1=49.0, ask_volume_1=1))
        self.assertEqual(sim.last_price, 49.5)  # tick last_price
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        # worst of (50, 49.5) for a buy is 50; slippage clamps back to 50
        self.assertAlmostEqual(order.average_price, 50.0)
        self.assertNotIn(order.order_id, sim.working)

    def test_24_instant_fill_market_order(self) -> None:
        """With instant_fill, a market order fills at the last price."""
        sim = self._instant_fill_sim()
        sim(market_data=self.create_tick(ask_price_1=49.0, ask_volume_1=1))
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        sim.launch_order(order=order)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(order.average_price, 49.5 * (1 + 0.0001))

    def test_25_instant_fill_requires_last_price(self) -> None:
        """Without a last price, an instant-fill market order stays working."""
        sim = self._instant_fill_sim()
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=float('nan'))
        sim.launch_order(order=order)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertIn(order.order_id, sim.working)

    # === lag ===

    def test_26_lag_ts(self) -> None:
        """Time lag delays the fill until the elapsed time reaches lag_ts."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            lag_ts=5., hit_prob=1.,
        )
        order = self.create_order(timestamp=100.)
        sim.launch_order(order=order)
        bar_early = self.create_bar(timestamp=103.)
        sim(market_data=bar_early)
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

        bar_late = self.create_bar(timestamp=105.)
        sim(market_data=bar_late)
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertEqual(order.filled_volume, 100)

    def test_27_lag_n_transaction(self) -> None:
        """Transaction lag delays the fill until enough transactions pass."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            lag_n_transaction=2, hit_prob=1.,
        )
        order = self.create_order()
        sim.launch_order(order=order)
        # fill is allowed once transactions_since_placement >= lag_n_transaction
        sim(market_data=TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=10, side=TransactionSide.SIDE_LONG,
        ))
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        sim(market_data=TransactionData(
            ticker='TEST', timestamp=time.time(),
            price=49.5, volume=10, side=TransactionSide.SIDE_LONG,
        ))
        self.assertEqual(order.order_state, OrderState.STATE_PARTFILLED)
        self.assertEqual(order.filled_volume, 10)

    # === hit probability ===

    def test_28_hit_probability_zero(self) -> None:
        """hit_prob=0 never fills."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0., seed=1,
        )
        order = self.create_order()
        sim.launch_order(order=order)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(order.order_state, OrderState.STATE_PLACED)
        self.assertEqual(order.filled_volume, 0)

    def test_29_hit_probability_seeded_pattern(self) -> None:
        """With a fixed seed the fill pattern matches the xorshift64* oracle."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            hit_prob=0.5, seed=42,
        )
        orders = [self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0) for _ in range(10)]
        for order in orders:
            sim.launch_order(order=order)

        # one bar matches every working order; each attempt draws once in
        # registry order (last launched first, C push_front), and every
        # successful fill consumes two further draws for its UUIDv4
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
                self.assertEqual(order.order_state, OrderState.STATE_FILLED,
                                 f'order {order.order_id} should have been filled')
            else:
                self.assertEqual(order.order_state, OrderState.STATE_PLACED,
                                 f'order {order.order_id} should not have been filled')

    # === slippage / fee / clamp ===

    def test_30_slippage_and_fee(self) -> None:
        """Slippage is applied per side and fee accumulates on the order."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            fee_rate=0.001, slippery_rate=0.0001, hit_prob=1.,
        )
        buy = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0, volume=50)
        sim.launch_order(order=buy)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=50))
        self.assertAlmostEqual(buy.average_price, 49.9 * 1.0001)

        sell = self.create_order(side=TransactionSide.SIDE_ASK, limit_price=50.0, volume=50)
        sim.launch_order(order=sell)
        sim(market_data=self.create_tick(bid_price_1=50.5, bid_volume_1=50, last_price=50.4))
        self.assertAlmostEqual(sell.average_price, 50.5 * 0.9999)

        report = next(iter(buy.trades.values()))
        self.assertAlmostEqual(report.fee, 0.001 * report.notional)
        self.assertAlmostEqual(buy.fee, report.fee)

    def test_31_limit_clamp(self) -> None:
        """Match price never exceeds the limit after slippage."""
        sim = SimMatchEx(
            ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set,
            slippery_rate=0.1, hit_prob=1.,
        )
        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        sim.launch_order(order=order)
        # 49.9 * 1.1 = 54.89 > 50 -> clamped to 50
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertAlmostEqual(order.average_price, 50.0)

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
            self.assertEqual(order.order_state, OrderState.STATE_CANCELED)

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
        self.assertEqual(self.sim.last_transaction_count, 0)
        self.assertEqual(self.sim.matching_config['fee_rate'], 0.001)

    # === static price helpers ===

    def test_34_best_worst_price(self) -> None:
        """best/worst price follow the side sign and skip invalid values."""
        self.assertEqual(
            SimMatchEx.best_price(1., 2., 3., side=TransactionSide.SIDE_BID), 1.)
        self.assertEqual(
            SimMatchEx.worst_price(1., 2., 3., side=TransactionSide.SIDE_BID), 3.)
        self.assertEqual(
            SimMatchEx.best_price(1., 2., 3., side=TransactionSide.SIDE_ASK), 3.)
        self.assertEqual(
            SimMatchEx.worst_price(1., 2., 3., side=TransactionSide.SIDE_ASK), 1.)
        self.assertEqual(
            SimMatchEx.best_price(None, float('nan'), 2., side=TransactionSide.SIDE_BID), 2.)
        with self.assertRaises(ValueError):
            SimMatchEx.best_price(side=TransactionSide.SIDE_BID)
        with self.assertRaises(ValueError):
            SimMatchEx.best_price(1., 2., side=TransactionSide.SIDE_UNKNOWN)

    # === market_time ===

    def test_35_market_time(self) -> None:
        """market_time mirrors the profile conversion of the timestamp."""
        self.sim.timestamp = 1735689600.  # 2025-01-01 00:00:00 UTC
        self.assertEqual(self.sim.market_time, PROFILE.timestamp_to_datetime(1735689600.))

    # === event engine C-API binding ===

    def test_36_register_unregister_capi(self) -> None:
        """register binds handlers through the event engine C API."""
        engine = EventEngineEx()
        sim = SimMatchEx(ticker='TEST', event_engine=engine, topic_set=TOPIC)
        sim.register()
        self.addCleanup(engine.stop)

        order = self.create_order()
        engine.start()
        try:
            engine.put(topic=TOPIC.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working),
                            'launch_order handler did not fire')
            self.assertEqual(order.order_state, OrderState.STATE_PLACED)

            engine.put(topic=TOPIC.cancel_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.history),
                            'cancel_order handler did not fire')
            self.assertEqual(order.order_state, OrderState.STATE_CANCELED)
        finally:
            engine.stop()

        sim.unregister()
        # after unregister, a launch is no longer processed
        order2 = self.create_order()
        engine.start()
        try:
            engine.put(topic=TOPIC.launch_order(ticker='TEST'), order=order2)
            self.assertFalse(self._wait_until(lambda: order2.order_id in sim.working, timeout=1.),
                             'handler still fired after unregister')
        finally:
            engine.stop()

    def test_37_realtime_dispatch_capi(self) -> None:
        """Market data routed through the realtime topic fills working orders."""
        engine = EventEngineEx()
        sim = SimMatchEx(ticker='TEST', event_engine=engine, topic_set=TOPIC)
        sim.register()
        self.addCleanup(sim.unregister)
        self.addCleanup(engine.stop)

        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        engine.start()
        try:
            engine.put(topic=TOPIC.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working))
            tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
            engine.put(topic=TOPIC.realtime(ticker='TEST', dtype='TickData'), market_data=tick)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.history),
                            'realtime handler did not fill the order')
            self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        finally:
            engine.stop()

    def test_38_multi_listener_capi(self) -> None:
        """A second native listener receives match and cancel events and can
        be deregistered by id."""
        from test.backtest import c_simmatch_ex_toolkit

        events = []
        sim = SimMatchEx(ticker='TEST', event_engine=self.event_engine, topic_set=self.topic_set, hit_prob=1.)
        listener_id = c_simmatch_ex_toolkit.register_listener(
            sim, lambda event, order_id, report: events.append((event, order_id, report)))

        order = self.create_order()
        sim.launch_order(order=order)
        # launch fires SMM_EVENT_PLACED
        self.assertEqual(len(events), 1)
        event, order_id, report = events[0]
        self.assertEqual(event, 2)  # SMM_EVENT_PLACED
        self.assertEqual(order_id, order.order_id)
        self.assertIsNone(report)

        # cancel path fires SMM_EVENT_CANCEL
        sim.cancel_order(order=order)
        self.assertEqual(len(events), 2)
        event, order_id, report = events[1]
        self.assertEqual(event, 1)  # SMM_EVENT_CANCEL
        self.assertEqual(order_id, order.order_id)
        self.assertIsNone(report)

        # match path fires SMM_EVENT_MATCH with a copied report
        order2 = self.create_order()
        sim.launch_order(order=order2)
        sim(market_data=self.create_tick(ask_price_1=49.9, ask_volume_1=100))
        self.assertEqual(len(events), 4)
        event, order_id, report = events[3]
        self.assertEqual(event, 0)  # SMM_EVENT_MATCH
        self.assertEqual(order_id, order2.order_id)
        self.assertIsInstance(report, TradeReport)
        self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))

        # deregister by id stops delivery
        c_simmatch_ex_toolkit.deregister_listener(sim, listener_id)
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
        self.assertEqual(order.order_state, OrderState.STATE_FILLED)
        self.assertNotIn(order.order_id, self.sim.working)
        self.assertIn(order.order_id, self.sim.history)
        # on_order (launch) + on_report + on_order (fill)
        self.assertEqual(self.event_engine.put.call_count, 3)

    def test_40_native_publish_capi(self) -> None:
        """Registered on a real engine, fills publish natively to on_report /
        on_order without Python wrapper involvement."""
        engine = EventEngineEx()
        reports = []
        orders = []
        engine.register_handler(topic=TOPIC.on_report, handler=lambda **kw: reports.append(kw['report']))
        engine.register_handler(topic=TOPIC.on_order, handler=lambda **kw: orders.append(kw['order']))

        sim = SimMatchEx(ticker='TEST', event_engine=engine, topic_set=TOPIC, hit_prob=1.)
        sim.register()
        self.addCleanup(engine.stop)

        order = self.create_order(side=TransactionSide.SIDE_BID, limit_price=50.0)
        engine.start()
        try:
            engine.put(topic=TOPIC.launch_order(ticker='TEST'), order=order)
            self.assertTrue(self._wait_until(lambda: order.order_id in sim.working))
            # placed on_order published natively
            self.assertTrue(self._wait_until(lambda: len(orders) >= 1),
                            'placed on_order not published natively')

            tick = self.create_tick(ask_price_1=49.9, ask_volume_1=100)
            engine.put(topic=TOPIC.realtime(ticker='TEST', dtype='TickData'), market_data=tick)
            self.assertTrue(self._wait_until(lambda: len(reports) >= 1),
                            'on_report not published natively')
            self.assertEqual(len(orders), 2)  # placed + filled

            report = reports[0]
            self.assertIsInstance(report, TradeReport)
            self.assertEqual(report.volume, 100)
            self.assertAlmostEqual(report.price, 49.9 * (1 + 0.0001))
            # the published order object is the same Python object
            self.assertIs(orders[1], order)
        finally:
            engine.stop()

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
