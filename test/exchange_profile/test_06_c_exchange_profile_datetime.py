import pathlib
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, pathlib.Path(__file__).parents[2] / 'algo_engine')

from algo_engine.exchange_profile.c_exchange_profile import SessionDateTime, SessionDateEx, SessionTime
from algo_engine.exchange_profile.c_profile_cn import PROFILE_CN

class TestSessionDateTime(unittest.TestCase):
    """Contract tests for SessionDateTime.

    Contracts:
    - `from_unix` and `to_pydatetime` are consistent with Python datetime under active profile timezone.
    - Scalar properties (`timestamp`, `ts`, `ordinal`) stay coherent with date/time views.
    - `fork()` creates a new wrapper sharing the same underlying C header.
    - `update()` mutates in place and updates all observable representations consistently.
    """

    @classmethod
    def setUpClass(cls):
        PROFILE_CN.activate()

    @classmethod
    def tearDownClass(cls):
        PROFILE_CN.deactivate()

    @staticmethod
    def _seconds_of_day(dt_obj: datetime) -> float:
        return (
            dt_obj.hour * 3600
            + dt_obj.minute * 60
            + dt_obj.second
            + dt_obj.microsecond / 1_000_000
        )

    @staticmethod
    def _mk_aware(y: int, m: int, d: int, hh: int, mm: int, ss: int, us: int = 0) -> datetime:
        return datetime(y, m, d, hh, mm, ss, us, tzinfo=PROFILE_CN.time_zone)

    def test_00_from_unix_roundtrip_and_scalar_invariants(self):
        samples = [
            self._mk_aware(1970, 1, 1, 0, 0, 0, 0),
            self._mk_aware(2024, 2, 29, 9, 30, 0, 123456),
            self._mk_aware(2024, 12, 31, 23, 59, 59, 999999),
            # self._mk_aware(1969, 12, 31, 23, 59, 59, 250000),
        ]

        for src in samples:
            with self.subTest(src=src.isoformat()):
                unix_ts = src.timestamp()
                session_dt = SessionDateTime.from_unix(unix_ts)
                dst = session_dt.to_pydatetime()

                self.assertEqual(dst, src)
                self.assertAlmostEqual(session_dt.timestamp, unix_ts, places=6)
                self.assertEqual(session_dt.ordinal, src.date().toordinal())
                self.assertAlmostEqual(session_dt.ts, self._seconds_of_day(src), places=6)

    def test_01_from_pydatetime_wall_clock_roundtrip(self):
        src = self._mk_aware(2024, 2, 8, 10, 15, 30, 123456)
        session_dt = SessionDateTime.from_pydatetime(src)
        dst = session_dt.to_pydatetime()

        self.assertEqual(dst.year, src.year)
        self.assertEqual(dst.month, src.month)
        self.assertEqual(dst.day, src.day)
        self.assertEqual(dst.hour, src.hour)
        self.assertEqual(dst.minute, src.minute)
        self.assertEqual(dst.second, src.second)
        self.assertEqual(dst.microsecond, src.microsecond)
        self.assertEqual(dst.utcoffset(), src.utcoffset())

        self.assertIsInstance(session_dt.date, SessionDateEx)
        self.assertIsInstance(session_dt.time, SessionTime)
        self.assertEqual(session_dt.date.to_pydate(), src.date())
        self.assertEqual(session_dt.time.hour, src.hour)
        self.assertEqual(session_dt.time.minute, src.minute)
        self.assertEqual(session_dt.time.second, src.second)
        self.assertEqual(session_dt.time.microsecond, src.microsecond)

    def test_02_repr_contract_with_and_without_fraction(self):
        with_fraction = SessionDateTime.from_pydatetime(self._mk_aware(2024, 2, 8, 9, 30, 1, 123000))
        without_fraction = SessionDateTime.from_pydatetime(self._mk_aware(2024, 2, 8, 9, 30, 1, 0))

        repr_fraction = repr(with_fraction)
        repr_no_fraction = repr(without_fraction)

        self.assertEqual(repr_fraction, "<SessionDateTime>(2024-02-08 09:30:01.123000)")
        self.assertEqual(repr_no_fraction, "<SessionDateTime>(2024-02-08 09:30:01)")

    def test_03_fork_shared_state_is_bidirectional(self):
        src = self._mk_aware(2024, 2, 8, 14, 0, 0)
        original = SessionDateTime.from_unix(src.timestamp())
        forked = original.fork()

        self.assertIsNot(original, forked)
        self.assertIsNot(original.date, forked.date)
        self.assertIsNot(original.time, forked.time)

        forked.update(src.timestamp() + 90.0)
        self.assertAlmostEqual(original.timestamp, forked.timestamp, places=6)
        self.assertEqual(original.ordinal, forked.ordinal)
        self.assertEqual(original.to_pydatetime(), forked.to_pydatetime())

        original.update(src.timestamp() + 3600.0)
        self.assertAlmostEqual(original.timestamp, forked.timestamp, places=6)
        self.assertEqual(original.to_pydatetime(), forked.to_pydatetime())

    def test_04_update_mutates_in_place_and_keeps_views_coherent(self):
        src = self._mk_aware(2024, 2, 8, 23, 59, 30)
        session_dt = SessionDateTime.from_unix(src.timestamp())
        date_view = session_dt.date
        time_view = session_dt.time

        ret = session_dt.update((src + timedelta(seconds=120)).timestamp())
        self.assertIs(ret, session_dt)

        expected = self._mk_aware(2024, 2, 9, 0, 1, 30)
        self.assertEqual(session_dt.to_pydatetime(), expected)
        self.assertEqual(date_view.to_pydate(), expected.date())
        self.assertEqual(time_view.hour, expected.hour)
        self.assertEqual(time_view.minute, expected.minute)
        self.assertEqual(time_view.second, expected.second)
        self.assertEqual(session_dt.ordinal, expected.date().toordinal())
        self.assertAlmostEqual(session_dt.ts, self._seconds_of_day(expected), places=6)

    def test_05_update_matches_direct_from_unix(self):
        base = self._mk_aware(2024, 2, 8, 10, 0, 0, 111111)
        target = self._mk_aware(2024, 2, 8, 14, 37, 5, 222222)

        via_update = SessionDateTime.from_unix(base.timestamp())
        via_update.update(target.timestamp())
        direct = SessionDateTime.from_unix(target.timestamp())

        self.assertEqual(via_update.to_pydatetime(), direct.to_pydatetime())
        self.assertAlmostEqual(via_update.timestamp, direct.timestamp, places=6)
        self.assertAlmostEqual(via_update.ts, direct.ts, places=6)
        self.assertEqual(via_update.ordinal, direct.ordinal)

    def test_06_from_unix_invalid_inputs_raise_runtime_error(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    SessionDateTime.from_unix(bad)

    def test_07_update_invalid_inputs_raise_runtime_error(self):
        src = self._mk_aware(2024, 2, 8, 10, 0, 0)
        session_dt = SessionDateTime.from_unix(src.timestamp())
        baseline = session_dt.to_pydatetime()

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    session_dt.update(bad)

        self.assertEqual(session_dt.to_pydatetime(), baseline)


if __name__ == "__main__":
    unittest.main()
