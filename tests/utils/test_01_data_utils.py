import datetime
import unittest

import numpy as np

from algo_engine.exchange_profile import PROFILE, PROFILE_CN
from algo_engine.utils.data_utils import fake_daily_data, fake_data, ts_indices


class TestTSIndices(unittest.TestCase):
    """Unit tests for :func:`algo_engine.utils.data_utils.ts_indices`.

    The tests activate the `PROFILE_CN` exchange profile in order to ensure
    deterministic behaviour when converting timestamps to datetimes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

    def test_basic_end_mode_timestamps(self) -> None:
        """Verify that `ts_mode='end'` returns end-of-candle timestamps with correct times.

        Session: 09:00 - 09:05, interval 60s -> expected end times: 09:01,09:02,09:03,09:04,09:05
        """
        market_date = datetime.date(2024, 2, 29)
        session_start = datetime.time(9, 40)
        session_end = datetime.time(9, 50)
        indices = ts_indices(
            market_date,
            60,
            session_start,
            session_end,
            None,
            'end',
            'timestamp'
        )

        # Expect five 1-minute candles ending at 09:41..09:50
        self.assertEqual(len(indices), 10)
        times = [PROFILE.timestamp_to_datetime(ts).time().replace(microsecond=0) for ts in indices]
        expected = [
            datetime.time(9, 41),
            datetime.time(9, 42),
            datetime.time(9, 43),
            datetime.time(9, 44),
            datetime.time(9, 45),
            datetime.time(9, 46),
            datetime.time(9, 47),
            datetime.time(9, 48),
            datetime.time(9, 49),
            datetime.time(9, 50),
        ]
        self.assertEqual(times, expected)

    def test_session_time_format_and_start_mode(self) -> None:
        """Verify `ts_format='session_time'` and `ts_mode='start'` produce session starts."""
        market_date = datetime.date(2024, 2, 29)
        session_start = datetime.time(9, 30)
        session_end = datetime.time(9, 35)
        indices = ts_indices(
            market_date,
            60,
            session_start,
            session_end,
            None,
            'start',
            'session_time'
        )

        # Expect five 1-minute candle starts at 09:00..09:04
        self.assertEqual(len(indices), 5)
        times = [st.to_pytime().replace(microsecond=0) for st in indices]
        expected = [datetime.time(9, 30), datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33), datetime.time(9, 34)]
        self.assertEqual(times, expected)

    def test_session_breaks_exclusion(self) -> None:
        """Ensure intervals that fall inside session breaks are excluded.

        Session: 09:30 - 09:36, interval 60s, break 09:33-09:35
        Expected end-of-candle times: 09:31,09:32,09:33,09:36 (09:34 and 09:35 are inside break)
        """
        market_date = datetime.date(2024, 2, 29)
        session_start = datetime.time(9, 30)
        session_end = datetime.time(9, 36)
        session_breaks = [(datetime.time(9, 33), datetime.time(9, 35))]

        indices = ts_indices(
            market_date,
            60,
            session_start,
            session_end,
            session_breaks,
            'end',
            'timestamp'
        )

        times = [PROFILE.timestamp_to_datetime(ts).time().replace(microsecond=0) for ts in indices]
        expected = [datetime.time(9, 31), datetime.time(9, 32), datetime.time(9, 33), datetime.time(9, 36)]
        self.assertEqual(times, expected)

    def test_ts_indices_of_date(self) -> None:
        """Call ts_indices with only market_date (CN profile active) and verify full-day indices.

        For the CN stock profile we expect two trading blocks: 09:30-11:30 and 13:00-15:00.
        With 60s interval and ts_mode='end' the first end-of-candle is 09:31 and the last is 15:00.
        There should be 120 minutes in each block (total 240).
        """
        md = datetime.date(2024, 11, 11)
        indices = ts_indices(market_date=md)

        # Basic sanity
        self.assertIsInstance(indices, list)
        self.assertTrue(indices)

        # Total expected candles: 120 (morning) + 120 (afternoon)
        self.assertEqual(len(indices), 240)

        first_dt = PROFILE.timestamp_to_datetime(indices[0])
        last_dt = PROFILE.timestamp_to_datetime(indices[-1])
        self.assertEqual(first_dt.time().replace(microsecond=0), datetime.time(9, 31))
        self.assertEqual(last_dt.time().replace(microsecond=0), datetime.time(15, 0))

        # Verify morning and afternoon counts by checking times
        times = [PROFILE.timestamp_to_datetime(ts).time().replace(microsecond=0) for ts in indices]
        morning_count = sum(1 for t in times if t <= datetime.time(11, 30))
        afternoon_count = sum(1 for t in times if t >= datetime.time(13, 1))
        self.assertEqual(morning_count, 120)
        self.assertEqual(afternoon_count, 120)

    def test_ts_indices_interval_5s_end_timestamp(self) -> None:
        """Verify ts_indices with 5s interval (timestamps, end mode) on CN profile."""
        md = datetime.date(2024, 11, 11)
        indices = ts_indices(market_date=md, interval=5)

        # total candles: 7200s per block / 5s = 1440 per block -> 2880 total
        self.assertEqual(len(indices), 2880)

        first_dt = PROFILE.timestamp_to_datetime(indices[0])
        last_dt = PROFILE.timestamp_to_datetime(indices[-1])
        # first end-of-candle should be 09:30:05, last should be 15:00:00
        self.assertEqual(first_dt.time().replace(microsecond=0), datetime.time(9, 30, 5))
        self.assertEqual(last_dt.time().replace(microsecond=0), datetime.time(15, 0, 0))

    def test_ts_indices_interval_5s_start_session_time(self) -> None:
        """Verify ts_indices with 5s interval, start mode and session_time format."""
        md = datetime.date(2024, 11, 11)
        indices = ts_indices(market_date=md, interval=5, ts_mode='start', ts_format='session_time')

        # same number of items, but items are SessionTime objects
        self.assertEqual(len(indices), 2880)

        first_st = indices[0]
        last_st = indices[-1]
        # first start-of-candle should be exactly 09:30:00
        self.assertEqual(first_st.to_pytime().replace(microsecond=0), datetime.time(9, 30, 0))
        # last start-of-candle should be 14:59:55 (15:00 - 5s)
        self.assertEqual(last_st.to_pytime().replace(microsecond=0), datetime.time(14, 59, 55))


class TestFakeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

    def test_fake_data_basic_matches_ts_indices(self) -> None:
        """Ensure fake_data returns a DataFrame whose index matches ts_indices for the same inputs."""
        md = datetime.date(2024, 11, 11)

        # reference indices using default behaviour
        ref_indices = ts_indices(market_date=md)

        df = fake_data(market_date=md)

        # DataFrame sanity
        self.assertTrue(hasattr(df, 'index'))
        self.assertEqual(len(df.index), len(ref_indices))

        # index values (timestamps) should match the reference indices
        # fake_data sets the index to the list returned from ts_indices
        self.assertEqual(list(df.index), list(ref_indices))

        # price columns exist
        for col in ('open_price', 'high_price', 'low_price', 'close_price'):
            self.assertIn(col, df.columns)

    def test_fake_data_with_custom_session_and_interval(self) -> None:
        """Test fake_data with explicit short session and interval and verify prices shape and first open price."""
        md = datetime.date(2024, 2, 29)
        session_start = datetime.time(9, 30)
        session_end = datetime.time(9, 35)
        interval = 60
        p0 = 123.45

        ref_indices = ts_indices(market_date=md, interval=interval, session_start=session_start, session_end=session_end)

        # call fake_data with matching parameters (note fake_data uses 'session_break' singular)
        df = fake_data(market_date=md, p0=p0, interval=interval, session_start=session_start, session_end=session_end, session_break=None)

        self.assertEqual(len(df.index), len(ref_indices))

        # first open price equals p0 by construction
        self.assertAlmostEqual(float(df['open_price'].iloc[0]), float(p0), places=6)


class TestFakeDailyData(unittest.TestCase):
    """Tests for :func:`algo_engine.utils.data_utils.fake_daily_data`.

    These tests provide an explicit `calendar` to avoid depending on
    third-party exchange calendar libraries in CI.
    """

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

    def test_fake_daily_data_with_explicit_calendar(self) -> None:
        cal = [
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4),
            datetime.date(2024, 1, 5),
            datetime.date(2024, 1, 8),
        ]

        df = fake_daily_data(start_date=cal[0], end_date=cal[-1], calendar=cal)

        # Basic DataFrame shape and index
        self.assertEqual(len(df), len(cal))
        self.assertEqual(list(df.index), cal)

        # Columns exist and positive prices
        for col in ('open_price', 'high_price', 'low_price', 'close_price'):
            self.assertIn(col, df.columns)
            self.assertTrue((df[col] > 0).all())

    def test_fake_daily_data_reproducible_with_seed(self) -> None:
        cal = [datetime.date(2024, 2, 1 + i) for i in range(10)]

        np.random.seed(12345)
        a = fake_daily_data(start_date=cal[0], end_date=cal[-1], calendar=cal)

        np.random.seed(12345)
        b = fake_daily_data(start_date=cal[0], end_date=cal[-1], calendar=cal)

        # DataFrames should be equal when seeded identically
        self.assertEqual(list(a.index), list(b.index))
        for col in ('open_price', 'high_price', 'low_price', 'close_price'):
            self.assertTrue((a[col].values == b[col].values).all())

    def test_fake_daily_data_generated_calendar_from_dates(self) -> None:
        """When no calendar is provided, fake_daily_data should generate the calendar from start/end dates via the profile."""
        start_date = datetime.date(2024, 11, 1)
        end_date = datetime.date(2024, 11, 11)

        # Use PROFILE to resolve the expected trading dates between start and end
        sd_range = PROFILE.trade_calendar(start_date, end_date)
        expected_dates = [d.to_pydate() for d in sd_range.dates]

        df = fake_daily_data(start_date=start_date, end_date=end_date)

        self.assertEqual(len(df), len(expected_dates))
        self.assertEqual(list(df.index), expected_dates)

        # nothing to assert here beyond calendar generation


if __name__ == '__main__':
    unittest.main()
