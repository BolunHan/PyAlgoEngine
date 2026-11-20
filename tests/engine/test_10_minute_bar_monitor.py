import unittest

from algo_engine.base import OrderData, TickDataLite, TransactionSide
from algo_engine.monitor.advanced_data_interface import MinuteBarMonitor

TICKER = '600012.SH'


class TestMinuteBarMonitor(unittest.TestCase):
    """Contract: the minute-bar monitor aggregates traded prices only.

    Expected behavior:
        - A tick creates / updates the ticker's minute bar.
        - Order data carries a limit price, not a market print; it neither
          creates a bar nor changes an existing bar's OHLC.
        - The bar's close is the latest traded / price-bearing observation.

    Oracle: the literal tick price fed in; the order price must never appear
    in the bar.
    """

    def setUp(self) -> None:
        self.monitor = MinuteBarMonitor(interval=60.)

    def tearDown(self) -> None:
        self.monitor._minute_bar_data.clear()
        self.monitor._last_bar_data.clear()

    @staticmethod
    def tick(price: float, timestamp: float) -> TickDataLite:
        return TickDataLite(
            ticker=TICKER,
            timestamp=timestamp,
            last_price=price,
            bid_price=price - 0.1,
            bid_volume=1.,
            ask_price=price + 0.1,
            ask_volume=1.,
        )

    @staticmethod
    def order(price: float, timestamp: float) -> OrderData:
        return OrderData(
            ticker=TICKER,
            timestamp=timestamp,
            price=price,
            volume=1.,
            side=TransactionSide.SIDE_BID,
        )

    def test_00_order_data_does_not_create_a_bar(self) -> None:
        """Order data alone leaves the monitor empty."""
        self.monitor(market_data=self.order(99., 1_000.))
        self.assertNotIn(TICKER, self.monitor._minute_bar_data)

    def test_01_tick_creates_a_bar(self) -> None:
        """A tick seeds the bar with its traded price."""
        self.monitor(market_data=self.tick(10., 1_000.))
        bar = self.monitor._minute_bar_data[TICKER]
        self.assertEqual(bar.open_price, 10.)
        self.assertEqual(bar.high_price, 10.)
        self.assertEqual(bar.low_price, 10.)
        self.assertEqual(bar.close_price, 10.)

    def test_02_order_does_not_change_an_existing_bar(self) -> None:
        """An order after a tick leaves the bar's OHLC untouched."""
        self.monitor(market_data=self.tick(10., 1_000.))
        bar = self.monitor._minute_bar_data[TICKER]

        self.monitor(market_data=self.order(99., 1_001.))

        self.assertEqual(bar.open_price, 10.)
        self.assertEqual(bar.high_price, 10.)
        self.assertEqual(bar.low_price, 10.)
        self.assertEqual(bar.close_price, 10.)


if __name__ == '__main__':
    unittest.main()
