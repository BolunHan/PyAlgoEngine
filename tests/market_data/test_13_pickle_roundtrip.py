import pickle
import unittest
from datetime import date, datetime

from algo_engine.base.c_market_data.c_candlestick import BarData, DailyBar
from algo_engine.base.c_market_data.c_internal import InternalData
from algo_engine.base.c_market_data.c_tick import TickData, TickDataLite
from algo_engine.base.c_market_data.c_trade_utils import TradeInstruction, TradeReport
from algo_engine.base.c_market_data.c_transaction import (
    OrderData,
    OrderType,
    TradeData,
    TransactionData,
    TransactionSide,
)

# Custom serializable attribute values shared by every variant below:
# a plain str key with a double value ("mid_price" / "open_interest") plus
# py_datetime / py_date values — the most frequent real-world extras.
PUB_TS = datetime(2026, 9, 8, 15, 30, 45, 123456)
PUB_DATE = date(2026, 9, 8)

PROTOCOLS = (2, pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL)


class TestMarketDataPickleRoundtrip(unittest.TestCase):
    """Contract: pickle round-trip of any MarketData variant restores the full
    C-level payload (dtype and to_bytes() identical) AND every custom
    __dict__ attribute — str keys with double values, and py_datetime /
    py_date values — regardless of pickle protocol.

    Expected behavior:
        - Custom attrs passed as constructor kwargs (and custom attrs
          assigned directly on the instance) live in instance __dict__ and
          survive pickle round-trip.
        - Reconstruction is dtype-driven (MarketData.__reduce__ →
          MarketData.from_bytes → __setstate__), so the unpickled class is
          the dtype-dispatch class: subclasses sharing a dtype with their
          parent (TradeData, BarData) come back as the dtype's canonical
          class (TransactionData, DailyBar).

    Oracle: original instance state captured before pickling; regen compared
    against it field by field (no implementation logic reused).
    """

    @staticmethod
    def _make_internal_data():
        return InternalData(
            ticker='600010.SH', timestamp=123.234, code=123,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_transaction_data():
        return TransactionData(
            ticker='600010.SH', timestamp=123.456, price=10.0, volume=2.0,
            side=TransactionSide.SIDE_LONG_OPEN,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_order_data():
        return OrderData(
            ticker='ORDER', timestamp=42.0, price=5.0, volume=3.0,
            side=TransactionSide.SIDE_ASK, order_id=123456789,
            order_type=OrderType.ORDER_LIMIT,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_trade_data():
        return TradeData(
            ticker='TRADE', timestamp=100.0, trade_price=20.0,
            trade_volume=4.0, trade_side=TransactionSide.SIDE_SHORT_CLOSE,
            multiplier=2.0,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_tick_data_lite():
        return TickDataLite(
            ticker='TEST', timestamp=100.0, last_price=101.0,
            bid_price=100.0, bid_volume=5.0, ask_price=102.0, ask_volume=4.0,
            # 'mid_price' is a real read-only property on tick variants, so
            # use a non-conflicting double key there.
            open_interest=12345.0, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_tick_data():
        return TickData(
            ticker='FULL', timestamp=200.0, last_price=101.5,
            open_price=95.0, prev_close=96.0,
            total_traded_volume=200.0, total_traded_notional=20000.0,
            total_trade_count=80,
            total_bid_volume=500.0, total_ask_volume=400.0,
            weighted_bid_price=99.9, weighted_ask_price=100.1,
            bid_price_1=100.0, bid_volume_1=10.0,
            ask_price_1=101.0, ask_volume_1=8.0,
            open_interest=12345.0, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_bar_data():
        return BarData(
            ticker='BAR', timestamp=120.0, high_price=11.0, low_price=10.0,
            open_price=10.5, close_price=10.8,
            volume=100.0, notional=1050.0, trade_count=7, bar_span=60.0,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_trade_report():
        return TradeReport(
            ticker='600010.SH', timestamp=200.0, price=10.0, volume=1.0,
            side=TransactionSide.SIDE_LONG_OPEN, order_id=987654321,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    @staticmethod
    def _make_trade_instruction():
        return TradeInstruction(
            ticker='600010.SH', timestamp=200.0,
            side=TransactionSide.SIDE_LONG_OPEN, volume=1.0,
            order_id=111222333,
            mid_price=100.25, pub_ts=PUB_TS, pub_date=PUB_DATE,
        )

    def test_constructor_kwargs_become_custom_attrs(self):
        """Unknown constructor kwargs land in __dict__ with values intact:
        double under a plain str key plus py_datetime / py_date values."""
        for name, factory in (
            ('InternalData', self._make_internal_data),
            ('TransactionData', self._make_transaction_data),
            ('OrderData', self._make_order_data),
            ('TradeData', self._make_trade_data),
            ('BarData', self._make_bar_data),
            ('TradeReport', self._make_trade_report),
            ('TradeInstruction', self._make_trade_instruction),
        ):
            with self.subTest(variant=name):
                md = factory()
                self.assertEqual(md.__dict__['mid_price'], 100.25)
                self.assertIsInstance(md.__dict__['mid_price'], float)
                self.assertEqual(md.mid_price, 100.25)
                self.assertEqual(md.__dict__['pub_ts'], PUB_TS)
                self.assertIsInstance(md.__dict__['pub_ts'], datetime)
                self.assertEqual(md.pub_ts, PUB_TS)
                self.assertEqual(md.__dict__['pub_date'], PUB_DATE)
                self.assertIsInstance(md.__dict__['pub_date'], date)
                self.assertEqual(md.pub_date, PUB_DATE)

        # Tick variants expose a read-only mid_price property, so the custom
        # double key must not collide with it.
        for name, factory in (
            ('TickDataLite', self._make_tick_data_lite),
            ('TickData', self._make_tick_data),
        ):
            with self.subTest(variant=name):
                md = factory()
                self.assertEqual(md.__dict__['open_interest'], 12345.0)
                self.assertIsInstance(md.__dict__['open_interest'], float)
                self.assertEqual(md.open_interest, 12345.0)
                self.assertEqual(md.__dict__['pub_ts'], PUB_TS)
                self.assertIsInstance(md.__dict__['pub_ts'], datetime)
                self.assertEqual(md.pub_ts, PUB_TS)
                self.assertEqual(md.__dict__['pub_date'], PUB_DATE)
                self.assertIsInstance(md.__dict__['pub_date'], date)
                self.assertEqual(md.pub_date, PUB_DATE)

    def test_pickle_roundtrip_preserves_custom_attrs(self):
        """Every variant round-trips with identical bytes, dtype, canonical
        class and __dict__ across pickle protocols.

        Reconstruction is dtype-driven (MarketData.__reduce__ →
        MarketData.from_bytes → __setstate__), so the canonical class of a
        variant is the dtype-dispatch class: TradeData shares DTYPE_TRANSACTION
        with TransactionData and BarData shares DTYPE_BAR with DailyBar."""
        for name, factory, canonical_type in (
            ('InternalData', self._make_internal_data, InternalData),
            ('TransactionData', self._make_transaction_data, TransactionData),
            ('OrderData', self._make_order_data, OrderData),
            ('TradeData', self._make_trade_data, TransactionData),
            ('TickDataLite', self._make_tick_data_lite, TickDataLite),
            ('TickData', self._make_tick_data, TickData),
            ('BarData', self._make_bar_data, DailyBar),
            ('TradeReport', self._make_trade_report, TradeReport),
            ('TradeInstruction', self._make_trade_instruction, TradeInstruction),
        ):
            md = factory()
            for protocol in PROTOCOLS:
                with self.subTest(variant=name, protocol=protocol):
                    regen = pickle.loads(pickle.dumps(md, protocol=protocol))

                    self.assertIs(type(regen), canonical_type)
                    self.assertEqual(regen.dtype, md.dtype)
                    self.assertEqual(regen.to_bytes(), md.to_bytes())
                    self.assertTrue(regen.owner)
                    self.assertEqual(regen.__dict__, md.__dict__)
                    if name in ('TickDataLite', 'TickData'):
                        self.assertEqual(regen.open_interest, 12345.0)
                        self.assertEqual(regen.pub_ts, PUB_TS)
                        self.assertEqual(regen.pub_date, PUB_DATE)
                    else:
                        self.assertEqual(regen.mid_price, 100.25)
                        self.assertEqual(regen.pub_ts, PUB_TS)
                        self.assertEqual(regen.pub_date, PUB_DATE)

    def test_direct_attribute_assignment_survives_pickle(self):
        """Custom attrs assigned after construction (plain assignment path,
        not constructor kwargs) also survive the round-trip."""
        for name, factory in (
            ('InternalData', self._make_internal_data),
            ('TransactionData', self._make_transaction_data),
            ('TickDataLite', self._make_tick_data_lite),
            ('BarData', self._make_bar_data),
            ('TradeReport', self._make_trade_report),
        ):
            md = factory()
            md.ref_price = 99.99
            md.ref_ts = datetime(2026, 9, 7, 8, 0, 0)
            for protocol in PROTOCOLS:
                with self.subTest(variant=name, protocol=protocol):
                    regen = pickle.loads(pickle.dumps(md, protocol=protocol))

                    self.assertEqual(regen.__dict__, md.__dict__)
                    self.assertEqual(regen.ref_price, 99.99)
                    self.assertEqual(regen.ref_ts, datetime(2026, 9, 7, 8, 0, 0))
                    self.assertEqual(regen.to_bytes(), md.to_bytes())

    def test_regen_mutation_does_not_leak_back(self):
        """Unpickled instance owns an independent __dict__: mutating the
        regen's custom attrs must not affect the original."""
        md = self._make_internal_data()
        regen = pickle.loads(pickle.dumps(md))

        regen.mid_price = 200.5
        self.assertEqual(md.mid_price, 100.25)
        self.assertEqual(regen.mid_price, 200.5)


if __name__ == '__main__':
    unittest.main()
