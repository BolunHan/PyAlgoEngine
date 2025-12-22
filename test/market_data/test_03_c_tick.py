import math
import unittest

from algo_engine.base.c_market_data_ng.c_tick import (
    OrderBook,
    TickData,
    TickDataLite,
)

from algo_engine.base.c_market_data_ng.c_transaction import TransactionDirection, TransactionSide
from algo_engine.base.c_market_data_ng.c_market_data import MarketData


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


if __name__ == '__main__':
    unittest.main()
