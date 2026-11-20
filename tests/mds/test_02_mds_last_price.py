import datetime
import math
import unittest

from algo_engine.base import BarData, OrderData, TickDataLite, TransactionData, TransactionSide
from algo_engine.engine import MDS
from algo_engine.exchange_profile import PROFILE_CN

TICKER = '600011.SH'


class TestMDSLastPrice(unittest.TestCase):
    """Contract: MDS.last_price is a traded price, with a stale-tick fallback.

    Expected behavior:
        - A transaction fill is the primary source of the ticker's last price.
        - A cancel transaction (side offset OFFSET_CANCEL) is not a trade and
          must not update the last price.
        - Order data, internal data and bars never update the last price.
        - A tick updates the last price only when at least
          MDS_TICK_PRICE_STALE_SECONDS (3 s) have elapsed since the last
          transaction-driven price update. Within that window the traded
          price stands.
        - A stale tick uses ``last_price`` when it is not NaN, otherwise the
          bid/ask mid-point; when one side is NaN it uses the present side
          (the price-limit case); when both sides are NaN it updates nothing.
        - MDS.timestamp still advances for every accepted observation; the
          subscription's last price and last-update time advance together.

    Oracle: literal values fed in; elapsed seconds are the literal timestamp
    differences inside the continuous session, and the mid-point is computed
    from the literal bid/ask arguments.
    """

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

    @classmethod
    def tearDownClass(cls) -> None:
        MDS.clear()
        PROFILE_CN.deactivate()

    def setUp(self) -> None:
        MDS.clear()

    def tearDown(self) -> None:
        MDS.clear()

    @staticmethod
    def ts(hour: int, minute: int, second: float = 0., day: int = 11) -> float:
        seconds = int(second)
        microsecond = round((second - seconds) * 1_000_000)
        return datetime.datetime(2024, 11, day, hour, minute, seconds, microsecond).timestamp()

    @classmethod
    def trade(cls, price: float, timestamp: float) -> TransactionData:
        return TransactionData(
            ticker=TICKER,
            timestamp=timestamp,
            price=price,
            volume=100.,
            side=TransactionSide.LongFilled,
        )

    @classmethod
    def cancel(cls, price: float, timestamp: float) -> TransactionData:
        return TransactionData(
            ticker=TICKER,
            timestamp=timestamp,
            price=price,
            volume=100.,
            side=TransactionSide.SIDE_LONG_CANCEL,
        )

    @classmethod
    def order(cls, price: float, timestamp: float) -> OrderData:
        return OrderData(
            ticker=TICKER,
            timestamp=timestamp,
            price=price,
            volume=100.,
            side=TransactionSide.SIDE_BID,
        )

    @classmethod
    def tick(cls, last_price: float, bid_price: float, ask_price: float, timestamp: float) -> TickDataLite:
        return TickDataLite(
            ticker=TICKER,
            timestamp=timestamp,
            last_price=last_price,
            bid_price=bid_price,
            bid_volume=0.,
            ask_price=ask_price,
            ask_volume=0.,
        )

    @classmethod
    def bar(cls, close_price: float, timestamp: float) -> BarData:
        return BarData(
            ticker=TICKER,
            timestamp=timestamp,
            high_price=close_price,
            low_price=close_price,
            open_price=close_price,
            close_price=close_price,
            bar_span=60.,
        )

    def price(self) -> float:
        return MDS.get_market_price(TICKER)

    # === non-trade observations ===

    def test_00_order_does_not_set_last_price(self) -> None:
        """Order data never seeds the last price."""
        MDS.on_market_data(self.order(120., self.ts(9, 45)))
        self.assertTrue(math.isnan(self.price()))

    def test_01_cancel_transaction_does_not_set_last_price(self) -> None:
        """A cancel transaction is not a trade and never seeds the price."""
        MDS.on_market_data(self.cancel(120., self.ts(9, 45)))
        self.assertTrue(math.isnan(self.price()))

    def test_02_bar_does_not_set_last_price(self) -> None:
        """Bar close is not a transaction and never seeds the price."""
        MDS.on_market_data(self.bar(120., self.ts(9, 45)))
        self.assertTrue(math.isnan(self.price()))

    # === transaction primary source ===

    def test_03_transaction_sets_last_price(self) -> None:
        """A fill transaction sets the last price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        self.assertEqual(self.price(), 10.5)

    def test_04_cancel_does_not_overwrite_trade_price(self) -> None:
        """A cancel report leaves the traded price intact."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.cancel(99., self.ts(9, 45, 1.)))
        self.assertEqual(self.price(), 10.5)

    def test_05_order_does_not_overwrite_trade_price(self) -> None:
        """A resting order's limit price never replaces the traded price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.order(120., self.ts(9, 45, 1.)))
        self.assertEqual(self.price(), 10.5)

    # === stale-tick window ===

    def test_06_tick_inside_stale_window_keeps_trade_price(self) -> None:
        """A tick 2.9 s after the trade does not replace the traded price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(10.9, 10.8, 11.0, self.ts(9, 45, 2.9)))
        self.assertEqual(self.price(), 10.5)

    def test_07_tick_after_stale_window_uses_tick_last_price(self) -> None:
        """A tick beyond 3 s supplies its last price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(10.9, 10.8, 11.0, self.ts(9, 45, 3.1)))
        self.assertEqual(self.price(), 10.9)

    def test_08_tick_with_nan_last_uses_mid_price(self) -> None:
        """A stale tick with no last price uses the bid/ask mid-point."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(float('nan'), 10.8, 11.0, self.ts(9, 45, 4.)))
        self.assertEqual(self.price(), 10.9)

    def test_09_tick_with_nan_last_uses_bid_when_ask_missing(self) -> None:
        """A stale one-sided tick (ask NaN) uses the bid price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(float('nan'), 10.8, float('nan'), self.ts(9, 45, 4.)))
        self.assertEqual(self.price(), 10.8)

    def test_10_tick_with_nan_last_uses_ask_when_bid_missing(self) -> None:
        """A stale one-sided tick (bid NaN) uses the ask price."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(float('nan'), float('nan'), 11.0, self.ts(9, 45, 4.)))
        self.assertEqual(self.price(), 11.0)

    def test_11_tick_without_any_price_keeps_previous(self) -> None:
        """A stale tick with every price field NaN updates nothing."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        MDS.on_market_data(self.tick(float('nan'), float('nan'), float('nan'), self.ts(9, 45, 4.)))
        self.assertEqual(self.price(), 10.5)

    # === rollover ===

    def test_12_tick_on_next_day_uses_fallback(self) -> None:
        """A new session date re-arms the tick fallback immediately."""
        MDS.on_market_data(self.trade(10.5, self.ts(14, 59, 59., day=11)))
        MDS.on_market_data(self.tick(10.7, 10.6, 10.8, self.ts(9, 45, day=12)))
        self.assertEqual(self.price(), 10.7)

    # === bookkeeping ===

    def test_13_timestamp_advances_for_non_price_observations(self) -> None:
        """Every accepted observation advances MDS.timestamp, price or not."""
        MDS.on_market_data(self.trade(10.5, self.ts(9, 45)))
        order_ts = self.ts(9, 45, 1.)
        MDS.on_market_data(self.order(120., order_ts))
        self.assertEqual(MDS.timestamp, order_ts)
        self.assertEqual(self.price(), 10.5)

    def test_14_order_only_ticker_is_nan_in_market_price_map(self) -> None:
        """An order-only ticker is subscribed but carries a NaN price."""
        MDS.on_market_data(self.order(120., self.ts(9, 45)))
        self.assertIn(TICKER, MDS.market_price)
        self.assertTrue(math.isnan(MDS.market_price[TICKER]))


if __name__ == '__main__':
    unittest.main()
