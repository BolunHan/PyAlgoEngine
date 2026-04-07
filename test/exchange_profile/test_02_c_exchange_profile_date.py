import pathlib
import sys
import unittest
from datetime import datetime, date, timedelta

sys.path.insert(0, pathlib.Path(__file__).parents[2] / 'algo_engine')

from algo_engine.exchange_profile.c_exchange_profile import SessionDate, SessionDateRange, local_utc_offset_seconds

TZ_OFFSET_SECONDS = local_utc_offset_seconds()


class TestSessionDate(unittest.TestCase):
    def test_from_ts_same_day_for_sample_timestamps(self):
        samples = [
            datetime(2024, 11, 11, 0, 0, 30).timestamp(),
            datetime(2024, 11, 11, 23, 59, 30).timestamp(),
        ]

        for ts in samples:
            with self.subTest(ts=ts):
                sd = SessionDate.from_unix(ts + TZ_OFFSET_SECONDS)  # monkey patch for no active profile
                expected = datetime.fromtimestamp(ts).date()
                self.assertEqual((sd.year, sd.month, sd.day), (expected.year, expected.month, expected.day))
                self.assertTrue(sd.is_valid())

    def test_from_ts_matches_python_datetime_many_samples(self):
        samples = [
            -2208988800.0,  # 1900-01-01 UTC
            -1234567890.0,
            -86400.0,
            -1.0,
            0.0,
            1.0,
            86399.0,
            86400.0,
            946684800.0,  # 2000-01-01 UTC
            1582934400.0,  # 2020-02-29 UTC
            1709164800.0,  # 2024-02-29 UTC
            1893456000.0,  # 2030-01-01 UTC
            2145916800.0,  # 2038-01-01 UTC
            4102444800.0,  # 2100-01-01 UTC
        ]

        for ts in samples:
            with self.subTest(ts=ts):
                sd = SessionDate.from_unix(ts + TZ_OFFSET_SECONDS)  # monkey patch for no active profile
                expected = datetime.fromtimestamp(ts).date()
                self.assertEqual((sd.year, sd.month, sd.day), (expected.year, expected.month, expected.day))

    def test_from_ts_fractional_seconds(self):
        ts = 1709164800.75
        sd = SessionDate.from_unix(ts)
        expected = datetime.fromtimestamp(ts).date()
        self.assertEqual((sd.year, sd.month, sd.day), (expected.year, expected.month, expected.day))

    def test_from_ts_invalid_inputs(self):
        with self.assertRaises(RuntimeError):
            SessionDate.from_unix(float("nan"))

        with self.assertRaises(RuntimeError):
            SessionDate.from_unix(float("inf"))

        with self.assertRaises(RuntimeError):
            SessionDate.from_unix(float("-inf"))

        with self.assertRaises(RuntimeError):
            SessionDate.from_unix(-62135683200.0)

    def test_static_apis_and_ordinal_roundtrip(self):
        self.assertTrue(SessionDate.is_leap_year(2000))
        self.assertFalse(SessionDate.is_leap_year(1900))
        self.assertTrue(SessionDate.is_leap_year(2024))
        self.assertFalse(SessionDate.is_leap_year(2023))

        self.assertEqual(SessionDate.days_in_month(2024, 2), 29)
        self.assertEqual(SessionDate.days_in_month(2023, 2), 28)
        self.assertEqual(SessionDate.days_in_month(2023, 4), 30)
        self.assertEqual(SessionDate.days_in_month(2023, 12), 31)

        ordinals = [1, 2, 31, 59, 60, 365, 366, 719163, 738945, 3652059]
        for ordinal in ordinals:
            with self.subTest(ordinal=ordinal):
                sd = SessionDate.from_ordinal(ordinal)
                self.assertEqual(sd.to_ordinal(), ordinal)

    def test_arithmetic_and_diff_results(self):
        d1 = SessionDate.from_pydate(date(2024, 2, 29))
        d2 = SessionDate.from_pydate(date(2024, 2, 28))

        # difference via ordinal
        self.assertEqual(d1.to_ordinal() - d2.to_ordinal(), 1)

        # add_days should return a new date
        d2b = d2.add_days(1)
        self.assertEqual((d2b.year, d2b.month, d2b.day), (2024, 2, 29))

    def test_comparison_results_and_from_to_pydate(self):
        a = SessionDate(2024, 1, 1)
        b = SessionDate(2024, 1, 2)
        c = SessionDate(2024, 1, 2)

        self.assertTrue(a < b)
        self.assertTrue(b > a)
        self.assertTrue(b == c)
        self.assertFalse(a == b)

        pydate = date.today()
        s = SessionDate.from_pydate(pydate)
        self.assertEqual(s.year, pydate.year)
        self.assertEqual(s.month, pydate.month)
        self.assertEqual(s.day, pydate.day)
        self.assertEqual(s.to_pydate(), pydate)

    def test_session_date_range_iteration_and_indexing(self):
        start = SessionDate.from_pydate(date(2024, 2, 27))
        end = SessionDate.from_pydate(date(2024, 3, 2))
        drange = SessionDateRange(start, end)
        self.assertEqual(drange.n_days, 5)
        # iteration yields SessionDate objects
        lst = list(iter(drange))
        self.assertEqual(len(lst), 5)
        self.assertEqual(lst[0].to_pydate(), date(2024, 2, 27))
        # indexing
        self.assertEqual(drange[2].to_pydate(), date(2024, 2, 29))


if __name__ == "__main__":
    unittest.main()
