import math
import unittest
from datetime import date, timedelta

from algo_engine.base.c_market_data_ng.c_candlestick import BarData, DailyBar
from algo_engine.base.c_market_data_ng.c_market_data import MarketData


class TestBarData(unittest.TestCase):

    def _make_bar(self, **overrides):
        params = dict(
            ticker='BAR',
            timestamp=120.0,
            high_price=11.0,
            low_price=9.0,
            open_price=10.0,
            close_price=10.5,
            volume=100.0,
            notional=1050.0,
            trade_count=7,
            bar_span=60.0,
        )
        params.update(overrides)
        return BarData(**params)

    def test_requires_span_or_start(self):
        with self.assertRaises(ValueError):
            BarData(
                ticker='ERR',
                timestamp=10.0,
                high_price=1.0,
                low_price=1.0,
                open_price=1.0,
                close_price=1.0,
            )

    def test_span_and_start_timestamp_derivation(self):
        bar = self._make_bar(bar_span=None, start_timestamp=60.0, timestamp=120.0)
        self.assertEqual(bar.start_timestamp, 60.0)
        self.assertEqual(bar.bar_span_seconds, 60.0)
        self.assertEqual(bar.bar_span, timedelta(seconds=60.0))
        self.assertEqual(bar.trade_count, 7)

    def test_item_access_and_mutation(self):
        bar = self._make_bar()
        self.assertEqual(bar['high_price'], 11.0)
        self.assertEqual(bar['volume'], 100.0)
        bar['close_price'] = 11.2
        bar['volume'] = 120.0
        self.assertEqual(bar.close_price, 11.2)
        self.assertEqual(bar.volume, 120.0)
        with self.assertRaises(KeyError):
            _ = bar['unknown']
        with self.assertRaises(KeyError):
            bar['unknown'] = 1

    def test_vwap_behavior(self):
        bar = self._make_bar(volume=50.0, notional=2500.0)
        self.assertEqual(bar.vwap, 50.0)
        bar = self._make_bar(volume=0.0, notional=2500.0)
        self.assertTrue(math.isnan(bar.vwap))

    def test_bar_type_classification(self):
        cases = (
            ('Sub-Minute', 30.0),
            ('Minute', 60.0),
            ('Minute-Plus', 90.0),
            ('Hourly', 3600.0),
            ('Hourly-Plus', 7200.0),
        )
        for expected, span in cases:
            bar = self._make_bar(bar_span=span, timestamp=span)
            self.assertEqual(bar.bar_type, expected)

    def test_repr_contains_key_fields(self):
        bar = self._make_bar()
        rep = repr(bar)
        self.assertIn('BAR', rep)
        self.assertIn('open=10.0', rep)
        self.assertIn('volume=100.0', rep)

    def test_serialization(self):
        bar_data = BarData(
            ticker='BAR',
            timestamp=120.0,
            high_price=11.0,
            low_price=9.0,
            open_price=10.0,
            close_price=10.5,
            volume=100.0,
            notional=1050.0,
            trade_count=7,
            bar_span=60.0,
        )

        daily = DailyBar(
            ticker='DAY',
            market_date=date(2024, 3, 15),
            high_price=11.0,
            low_price=9.0,
            open_price=10.0,
            close_price=10.5,
            volume=100.0,
            notional=1050.0,
            trade_count=5,
            bar_span=1,
        )

        regen_bar_1 = BarData.from_bytes(bar_data.to_bytes())
        self.assertEqual(regen_bar_1.to_bytes(), bar_data.to_bytes())
        regen_bar_2 = MarketData.from_bytes(bar_data.to_bytes())
        self.assertEqual(regen_bar_2.to_bytes(), bar_data.to_bytes())

        regen_daily_1 = DailyBar.from_bytes(daily.to_bytes())
        self.assertEqual(regen_daily_1.to_bytes(), daily.to_bytes())
        regen_daily_2 = MarketData.from_bytes(daily.to_bytes())
        self.assertEqual(regen_daily_2.to_bytes(), daily.to_bytes())


class TestDailyBar(unittest.TestCase):

    def _make_daily_bar(self, **overrides):
        params = dict(
            ticker='DAY',
            market_date=date(2024, 3, 15),
            high_price=11.0,
            low_price=9.0,
            open_price=10.0,
            close_price=10.5,
            volume=100.0,
            notional=1050.0,
            trade_count=5,
            bar_span=1,
        )
        params.update(overrides)
        return DailyBar(**params)

    def test_market_date_encoding(self):
        market_date = date(2025, 1, 2)
        bar = self._make_daily_bar(market_date=market_date)
        self.assertEqual(bar.timestamp, 20250102)
        self.assertEqual(bar.market_date, market_date)
        self.assertEqual(bar.market_time, market_date)

    def test_repr_variants(self):
        single = self._make_daily_bar(bar_span=1)
        self.assertIn('[2024-03-15]', repr(single))
        self.assertNotIn('span=', repr(single))
        multi = self._make_daily_bar(bar_span=3)
        self.assertIn('span=3d', repr(multi))

    def test_bar_span_and_time_window(self):
        market_date = date(2023, 12, 31)
        bar = self._make_daily_bar(market_date=market_date, bar_span=3)
        self.assertEqual(bar.bar_span, timedelta(days=3))
        self.assertEqual(bar.bar_start_time, market_date - timedelta(days=3))
        self.assertEqual(bar.bar_end_time, market_date)

    def test_bar_type_and_validation(self):
        daily = self._make_daily_bar(bar_span=1)
        self.assertEqual(daily.bar_type, 'Daily')
        multi = self._make_daily_bar(bar_span=5)
        self.assertEqual(multi.bar_type, 'Daily-Plus')
        invalid = self._make_daily_bar(bar_span=0)
        with self.assertRaises(ValueError):
            _ = invalid.bar_type


if __name__ == '__main__':
    unittest.main()
