import math
import unittest

from algo_engine.base.c_market_data.c_tick import (
    OrderBook,
    TickData,
    TickDataLite,
)

from algo_engine.base.c_market_data.c_transaction import TransactionDirection, TransactionSide
from algo_engine.base.c_market_data.c_market_data import MarketData


def build_order_book(direction: int) -> OrderBook:
    prices = [100.0, 99.5, 99.0]
    volumes = [10.0, 8.0, 6.0]
    n_orders = [2, 1, 1]
    return OrderBook(direction=direction, price=prices, volume=volumes, n_orders=n_orders)


class TestTickDataLite(unittest.TestCase):

    def test_basic_properties_and_mid_spread(self):
        tick = TickDataLite(
            ticker='TEST',
            timestamp=100.0,
            last_price=101.0,
            bid_price=100.0,
            bid_volume=5.0,
            ask_price=102.0,
            ask_volume=4.0,
            open_price=95.0,
            prev_close=96.0,
            total_traded_volume=1000.0,
            total_traded_notional=100000.0,
            total_trade_count=50,
        )

        self.assertEqual(tick.last_price, 101.0)
        self.assertEqual(tick.bid_price, 100.0)
        self.assertEqual(tick.ask_price, 102.0)
        self.assertEqual(tick.bid_volume, 5.0)
        self.assertEqual(tick.ask_volume, 4.0)
        self.assertEqual(tick.open_price, 95.0)
        self.assertEqual(tick.prev_close, 96.0)
        self.assertEqual(tick.total_traded_volume, 1000.0)
        self.assertEqual(tick.total_traded_notional, 100000.0)
        self.assertEqual(tick.total_trade_count, 50)
        self.assertEqual(tick.mid_price, 101.0)
        self.assertEqual(tick.spread, 2.0)

    def test_serialize(self):
        tick_lite = TickDataLite(
            ticker='TEST',
            timestamp=100.0,
            last_price=101.0,
            bid_price=100.0,
            bid_volume=5.0,
            ask_price=102.0,
            ask_volume=4.0,
            open_price=95.0,
            prev_close=96.0,
            total_traded_volume=1000.0,
            total_traded_notional=100000.0,
            total_trade_count=50,
        )

        blob = tick_lite.to_bytes()
        regen_1 = TickDataLite.from_bytes(blob)
        regen_2 = MarketData.from_bytes(blob)

        assert regen_1.to_bytes() == regen_2.to_bytes() == blob


class TestOrderBook(unittest.TestCase):

    def test_iteration_and_indexing(self):
        book = build_order_book(direction=TransactionDirection.DIRECTION_SHORT)

        levels = list(book)
        self.assertEqual(len(levels), 3)
        self.assertEqual(levels[-1][0], 100.0)
        self.assertEqual(levels[-1][1], 10.0)
        self.assertEqual(book[1][0], 99.5)
        self.assertEqual(book[0][0], 99.0)

        self.assertEqual(book.direction, TransactionDirection.DIRECTION_SHORT)
        self.assertEqual(book.side, TransactionSide.Ask)
        self.assertEqual(book.capacity, 10)
        self.assertEqual(book.size, 3)

    def test_at_price_and_at_level(self):
        book = build_order_book(direction=TransactionDirection.DIRECTION_SHORT)
        self.assertEqual(book.at_price(99.5), (99.5, 8.0, 1))
        self.assertEqual(book.at_level(0), (99.0, 6.0, 1))
        book = build_order_book(direction=TransactionDirection.DIRECTION_LONG)
        self.assertEqual(book.at_level(2), (99.0, 6.0, 1))

        with self.assertRaises(IndexError):
            book.at_price(101.0)

    def test_loc_volume_and_sort(self):
        book = build_order_book(direction=TransactionDirection.DIRECTION_LONG)
        self.assertEqual(book.loc_volume(99.0, 100.1), 24)
        self.assertEqual(book.loc_volume(99.0, 99.6), 14)
        self.assertTrue(book.sorted)

    def test_serialization_round_trip(self):
        book = build_order_book(direction=TransactionDirection.DIRECTION_LONG)
        blob = book.to_bytes()
        restored = OrderBook.from_bytes(blob)

        self.assertEqual(list(book), list(restored))


class TestTickData(unittest.TestCase):

    def test_full_tick_initialization(self):
        tick = TickData(
            ticker='FULL',
            timestamp=200.0,
            last_price=101.5,
            open_price=95.0,
            prev_close=96.0,
            total_traded_volume=200.0,
            total_traded_notional=20000.0,
            total_trade_count=80,
            total_bid_volume=500.0,
            total_ask_volume=400.0,
            weighted_bid_price=99.9,
            weighted_ask_price=100.1,
            bid_price_1=100.0,
            bid_volume_1=10.0,
            ask_price_1=101.0,
            ask_volume_1=8.0,
            bid_price_2=99.5,
            bid_volume_2=9.0,
            ask_price_2=101.5,
            ask_volume_2=7.0,
        )

        self.assertEqual(tick.last_price, 101.5)
        self.assertEqual(tick.open_price, 95.0)
        self.assertEqual(tick.prev_close, 96.0)
        self.assertEqual(tick.total_traded_volume, 200.0)
        self.assertEqual(tick.total_traded_notional, 20000.0)
        self.assertEqual(tick.total_trade_count, 80)
        self.assertEqual(tick.total_bid_volume, 500.0)
        self.assertEqual(tick.total_ask_volume, 400.0)
        self.assertEqual(tick.weighted_bid_price, 99.9)
        self.assertEqual(tick.weighted_ask_price, 100.1)

        self.assertEqual(tick.best_bid_price, 100.0)
        self.assertEqual(tick.best_bid_volume, 10.0)
        self.assertEqual(tick.best_ask_price, 101.0)
        self.assertEqual(tick.best_ask_volume, 8.0)
        self.assertEqual(tick.bid_price, 100.0)
        self.assertEqual(tick.ask_price, 101.0)
        self.assertEqual(tick.mid_price, 100.5)
        self.assertEqual(tick.spread, 1.0)

        lite_view = tick.lite()
        self.assertEqual(lite_view.last_price, 101.5)
        self.assertEqual(lite_view.bid_price, 100.0)
        self.assertEqual(lite_view.ask_price, 101.0)

    def test_serialize(self):
        tick = TickData(
            ticker='FULL',
            timestamp=200.0,
            last_price=101.5,
            open_price=95.0,
            prev_close=96.0,
            total_traded_volume=200.0,
            total_traded_notional=20000.0,
            total_trade_count=80,
            total_bid_volume=500.0,
            total_ask_volume=400.0,
            weighted_bid_price=99.9,
            weighted_ask_price=100.1,
            bid_price_1=100.0,
            bid_volume_1=10.0,
            ask_price_1=101.0,
            ask_volume_1=8.0,
            bid_price_2=99.5,
            bid_volume_2=9.0,
            ask_price_2=101.5,
            ask_volume_2=7.0,
        )

        blob = tick.to_bytes()
        regen_1 = TickData.from_bytes(blob)
        regen_2 = MarketData.from_bytes(blob)

        assert regen_1.to_bytes() == regen_2.to_bytes() == blob

    def test_02_lite_view_reference(self):
        """Verify lite(copy=False) returns non-owning view to embedded lite data."""
        tick = TickData(
            ticker='LITE_VIEW',
            timestamp=300.0,
            last_price=105.0,
            bid_price_1=104.0,
            bid_volume_1=12.0,
            ask_price_1=106.0,
            ask_volume_1=11.0,
            open_price=98.0,
            prev_close=99.0,
            total_traded_volume=500.0,
            total_traded_notional=50000.0,
            total_trade_count=100,
        )

        # Get view with copy=False (default)
        lite_view = tick.lite(copy=False)

        # Verify non-owning view properties
        self.assertFalse(lite_view.owner)
        self.assertEqual(lite_view.ticker, 'LITE_VIEW')
        self.assertEqual(lite_view.timestamp, 300.0)
        self.assertEqual(lite_view.last_price, 105.0)
        self.assertEqual(lite_view.bid_price, 104.0)
        self.assertEqual(lite_view.bid_volume, 12.0)
        self.assertEqual(lite_view.ask_price, 106.0)
        self.assertEqual(lite_view.ask_volume, 11.0)
        self.assertEqual(lite_view.open_price, 98.0)
        self.assertEqual(lite_view.prev_close, 99.0)
        self.assertEqual(lite_view.total_traded_volume, 500.0)
        self.assertEqual(lite_view.total_traded_notional, 50000.0)
        self.assertEqual(lite_view.total_trade_count, 100)

    def test_03_lite_owned_copy(self):
        """Verify lite(copy=True) returns independently owned copy."""
        tick = TickData(
            ticker='LITE_COPY',
            timestamp=400.0,
            last_price=110.0,
            bid_price_1=109.0,
            bid_volume_1=15.0,
            ask_price_1=111.0,
            ask_volume_1=14.0,
            open_price=100.0,
            prev_close=101.0,
            total_traded_volume=1000.0,
            total_traded_notional=100000.0,
            total_trade_count=200,
        )

        # Get owned copy with copy=True
        lite_copy = tick.lite(copy=True)

        # Verify owned copy properties
        self.assertTrue(lite_copy.owner)
        self.assertEqual(lite_copy.ticker, 'LITE_COPY')
        self.assertEqual(lite_copy.timestamp, 400.0)
        self.assertEqual(lite_copy.last_price, 110.0)
        self.assertEqual(lite_copy.bid_price, 109.0)
        self.assertEqual(lite_copy.bid_volume, 15.0)
        self.assertEqual(lite_copy.ask_price, 111.0)
        self.assertEqual(lite_copy.ask_volume, 14.0)
        self.assertEqual(lite_copy.open_price, 100.0)
        self.assertEqual(lite_copy.prev_close, 101.0)
        self.assertEqual(lite_copy.total_traded_volume, 1000.0)
        self.assertEqual(lite_copy.total_traded_notional, 100000.0)
        self.assertEqual(lite_copy.total_trade_count, 200)

    def test_04_lite_copy_independence(self):
        """Verify copy/view data consistency and independence."""
        tick = TickData(
            ticker='LITE_INDEPENDENCE',
            timestamp=500.0,
            last_price=115.0,
            bid_price_1=114.0,
            bid_volume_1=20.0,
            ask_price_1=116.0,
            ask_volume_1=19.0,
            open_price=102.0,
            prev_close=103.0,
            total_traded_volume=1500.0,
            total_traded_notional=150000.0,
            total_trade_count=300,
        )

        # Get both view and copy
        lite_view = tick.lite(copy=False)
        lite_copy = tick.lite(copy=True)

        # Verify both have identical data
        self.assertEqual(lite_view.last_price, lite_copy.last_price)
        self.assertEqual(lite_view.bid_price, lite_copy.bid_price)
        self.assertEqual(lite_view.ask_price, lite_copy.ask_price)
        self.assertEqual(lite_view.bid_volume, lite_copy.bid_volume)
        self.assertEqual(lite_view.ask_volume, lite_copy.ask_volume)
        self.assertEqual(lite_view.open_price, lite_copy.open_price)
        self.assertEqual(lite_view.prev_close, lite_copy.prev_close)
        self.assertEqual(lite_view.total_traded_volume, lite_copy.total_traded_volume)
        self.assertEqual(lite_view.total_traded_notional, lite_copy.total_traded_notional)
        self.assertEqual(lite_view.total_trade_count, lite_copy.total_trade_count)

        # Verify ownership difference
        self.assertFalse(lite_view.owner)
        self.assertTrue(lite_copy.owner)

        # Verify copy can be serialized and deserialized independently
        copy_bytes = lite_copy.to_bytes()
        restored_copy = TickDataLite.from_bytes(copy_bytes)

        # Verify restored copy has same data values
        self.assertEqual(restored_copy.last_price, 115.0)
        self.assertEqual(restored_copy.bid_price, 114.0)
        self.assertEqual(restored_copy.ask_price, 116.0)
        self.assertEqual(restored_copy.bid_volume, 20.0)
        self.assertEqual(restored_copy.ask_volume, 19.0)


if __name__ == '__main__':
    unittest.main()
