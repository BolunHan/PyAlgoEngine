import unittest
import datetime
import time
from algo_engine.profile.c_exchange_profile import *


class TestSessionTime(unittest.TestCase):
    def test_init_and_properties(self):
        # Create a SessionTime with explicit nanoseconds
        st = SessionTime(9, 30, 15, 123456789)

        # Basic properties
        self.assertEqual(st.hour, 9)
        self.assertEqual(st.minute, 30)
        self.assertEqual(st.second, 15)
        self.assertEqual(st.nanosecond, 123456789)
        self.assertEqual(st.microsecond, 123456)  # floor division

        # elapsed_seconds and ts should include fractional seconds from nanoseconds
        expected_ts = 9 * 3600 + 30 * 60 + 15 + 123456789 / 1e9
        self.assertAlmostEqual(st.ts, expected_ts, places=9)
        # When no EX_PROFILE is active, elapsed_seconds equals ts (see C implementation)
        self.assertAlmostEqual(st.elapsed_seconds, expected_ts, places=9)

        self.assertEqual(st.session_phase, SessionPhase.CONTINUOUS)

    def test_repr_and_str_roundtrip(self):
        st = SessionTime(0, 0, 1, 500000000)
        # repr should include the isoformat of the underlying time
        r = repr(st)
        self.assertIn("00:00:01", r)

    def test_from_and_to_pytime(self):
        t = datetime.time(14, 5, 6, 789123)
        st = SessionTime.from_pytime(t)
        # to_pytime should round-trip to the same microsecond precision
        t2 = st.to_pytime()
        self.assertEqual(t.hour, t2.hour)
        self.assertEqual(t.minute, t2.minute)
        self.assertEqual(t.second, t2.second)
        self.assertEqual(t.microsecond, t2.microsecond)

    def test_isoformat_and_fromisoformat(self):
        st = SessionTime(23, 59, 59, 999000000)
        s = st.isoformat()
        # Should parse back to equivalent SessionTime
        st2 = SessionTime.fromisoformat(s)
        self.assertEqual(st.hour, st2.hour)
        self.assertEqual(st.minute, st2.minute)
        self.assertEqual(st.second, st2.second)
        self.assertEqual(st.microsecond, st2.microsecond)

    def test_comparisons_with_sessiontime_and_pytime(self):
        a = SessionTime(9, 0, 0, 0)
        b = SessionTime(10, 0, 0, 0)
        self.assertTrue(a < b)
        self.assertTrue(b > a)
        self.assertTrue(a <= b)
        self.assertTrue(b >= a)
        self.assertFalse(a == b)
        self.assertTrue(a != b)

        # Compare with python datetime.time
        t = datetime.time(9, 0, 0)
        self.assertTrue(a == t or a == SessionTime.from_pytime(t))
        self.assertTrue(b > t)

    def test_from_timestamp(self):
        # Use a unix_ts inside a day for deterministic mapping
        # unix_ts = 9 * 3600 + 30 * 60 + 15 + 0.123456789
        unix_ts = time.time()
        ts_offset = local_utc_offset_seconds()
        py_time = datetime.datetime.fromtimestamp(unix_ts).time()  # with default profile activated, the tz_offset_seconds are unset, we have to adjust manually.
        st = SessionTime.from_timestamp(unix_ts + ts_offset)

        # Check that fields match the components of unix_ts
        self.assertEqual(st.hour, py_time.hour)
        self.assertEqual(st.minute, py_time.minute)
        self.assertEqual(st.second, py_time.second)
        # Allow microsecond rounding tolerance
        self.assertAlmostEqual(st.microsecond, py_time.microsecond, delta=1)  # max 1 microsecond difference due to rounding vs flooring, SessionTime uses flooring to guarantee trading system safety

    def test_timedelta_subtraction(self):
        # Timedelta-like subtraction returns a SessionTimeRange with elapsed_seconds
        a = SessionTime(9, 0, 0, 0)
        b = SessionTime(10, 0, 1, 500000000)
        tr = b - a
        # elapsed should equal difference of their elapsed_seconds values
        self.assertAlmostEqual(tr.elapsed_seconds, b.elapsed_seconds - a.elapsed_seconds, places=9)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
