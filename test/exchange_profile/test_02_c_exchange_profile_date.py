import pathlib
import sys
import unittest
from datetime import datetime, date, timedelta

sys.path.insert(0, pathlib.Path(__file__).parents[2] / 'algo_engine')

from algo_engine.exchange_profile.c_exchange_profile import (
    SessionDate,
    SessionDateRange,
    SessionType,
    local_utc_offset_seconds,
)

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


class TestSessionDatePythonInterfaces(unittest.TestCase):
    """Contract tests for SessionDate Python interfaces and datetime.date compatibility."""

    @staticmethod
    def _mk(y, m, d):
        return SessionDate(y, m, d), date(y, m, d)

    def test_00_explicit_session_date_interfaces(self):
        sd, pd = self._mk(2024, 3, 1)

        self.assertEqual((sd.year, sd.month, sd.day), (2024, 3, 1))
        self.assertIn("<SessionDate>(2024-03-01)", repr(sd))
        self.assertEqual(hash(sd), sd.to_ordinal())
        self.assertEqual(sd.to_pydate(), pd)
        self.assertTrue(sd.is_valid())
        self.assertIsInstance(sd.session_type, SessionType)

        forked = sd.fork()
        self.assertIsInstance(forked, SessionDate)
        self.assertEqual(forked.to_pydate(), pd)

        self.assertEqual(SessionDate.from_pydate(pd).to_pydate(), pd)
        self.assertEqual(SessionDate.fromisoformat("2024-03-01").to_pydate(), pd)
        self.assertEqual(SessionDate.fromisocalendar(2024, 9, 5).to_pydate(), date(2024, 3, 1))

        today = SessionDate.today()
        self.assertEqual(today.to_pydate(), date.today())

        self.assertTrue(SessionDate.is_leap_year(2024))
        self.assertFalse(SessionDate.is_leap_year(2100))
        self.assertEqual(SessionDate.days_in_month(2024, 2), 29)
        self.assertEqual(SessionDate.days_in_month(2024, 11), 30)

        ordinal = pd.toordinal()
        self.assertEqual(SessionDate.from_ordinal(ordinal).to_ordinal(), ordinal)

        unix_ts = datetime(2024, 3, 1, 18, 0, 0).timestamp()
        from_unix = SessionDate.from_unix(unix_ts + TZ_OFFSET_SECONDS)
        self.assertEqual(from_unix.to_pydate(), datetime.fromtimestamp(unix_ts).date())
        self.assertEqual(SessionDate.unix_to_ordinal(unix_ts + TZ_OFFSET_SECONDS), datetime.fromtimestamp(unix_ts).date().toordinal())

        self.assertEqual(sd.add_days(1).to_pydate(), date(2024, 3, 2))
        self.assertEqual(sd.add_days(30).to_pydate(), date(2024, 3, 31))

        # timestamp() contract: a unix roundtrip should recover the same session date.
        self.assertEqual(SessionDate.from_unix(sd.timestamp()).to_pydate(), pd)

    def test_01_subtraction_and_comparison_contracts(self):
        sd1, pd1 = self._mk(2024, 3, 1)
        sd2, pd2 = self._mk(2024, 3, 3)

        rng_ss = sd2 - sd1
        self.assertIsInstance(rng_ss, SessionDateRange)
        self.assertEqual(rng_ss.n_days, 3)

        rng_sp = sd2 - pd1
        self.assertIsInstance(rng_sp, SessionDateRange)
        self.assertEqual(rng_sp[0].to_pydate(), pd1)

        rng_ps = pd2 - sd1
        self.assertIsInstance(rng_ps, SessionDateRange)
        self.assertEqual(rng_ps[-1].to_pydate(), pd2)

        self.assertTrue(sd1 < sd2)
        self.assertTrue(sd1 <= sd2)
        self.assertTrue(sd2 > sd1)
        self.assertTrue(sd2 >= sd1)
        self.assertTrue(sd1 == pd1)
        self.assertTrue(sd1 != pd2)

        with self.assertRaises(TypeError):
            _ = sd1 - "2024-03-01"

        with self.assertRaises(TypeError):
            _ = sd1 < "2024-03-01"

    def test_02_weekend_and_invalid_input_paths(self):
        saturday = SessionDate.from_pydate(date(2024, 3, 2))
        monday = SessionDate.from_pydate(date(2024, 3, 4))
        self.assertTrue(saturday.is_weekend())
        self.assertFalse(monday.is_weekend())

        with self.assertRaises(RuntimeError):
            SessionDate.from_ordinal(0)

        with self.assertRaises(RuntimeError):
            SessionDate.from_unix(float("nan"))

    def test_03_datetime_date_inherited_interface_contract(self):
        # These checks intentionally mirror datetime.date behavior to catch protocol regressions.
        sd, pd = self._mk(2024, 3, 1)

        self.assertEqual(str(sd), str(pd))
        self.assertEqual(sd.isoformat(), pd.isoformat())
        self.assertEqual(sd.ctime(), pd.ctime())
        self.assertEqual(sd.strftime("%Y-%m-%d %A"), pd.strftime("%Y-%m-%d %A"))
        self.assertEqual(format(sd, "%Y%m%d"), format(pd, "%Y%m%d"))

        self.assertEqual(sd.weekday(), pd.weekday())
        self.assertEqual(sd.isoweekday(), pd.isoweekday())
        self.assertEqual(sd.isocalendar(), pd.isocalendar())
        self.assertEqual(sd.timetuple(), pd.timetuple())

        self.assertEqual(sd.replace(day=2), pd.replace(day=2))

        # Inherited arithmetic should match date semantics exactly.
        self.assertEqual(sd + timedelta(days=1), pd + timedelta(days=1))
        self.assertEqual(sd - timedelta(days=1), pd - timedelta(days=1))

        # Inherited constructor-style interfaces from datetime.date.
        self.assertEqual(SessionDate.fromordinal(pd.toordinal()), pd)
        self.assertEqual(SessionDate.min, date.min)
        self.assertEqual(SessionDate.max, date.max)
        self.assertEqual(SessionDate.resolution, date.resolution)


if __name__ == "__main__":
    unittest.main()
