import unittest
from datetime import date, datetime, time as py_time

from algo_engine.profile.c_exchange_profile import (
    PROFILE_CN,
    PROFILE,
    SessionDate,
    SessionType,
    SessionPhase,
    AuctionPhase,
)


class TestCNProfile(unittest.TestCase):
    def setUp(self):
        self.cn = PROFILE_CN
        self.cn.activate()

    def tearDown(self):
        self.cn.deactivate()

    def test_profile_basic_values(self):
        self.assertEqual(self.cn.profile_id, "CN_STOCK")
        self.assertEqual(self.cn.session_start.hour, 9)
        self.assertEqual(self.cn.session_start.minute, 30)
        self.assertEqual(self.cn.session_end.hour, 15)
        self.assertEqual(self.cn.session_end.minute, 0)
        self.assertAlmostEqual(self.cn.tz_offset_seconds, 28800.0)

    def test_resolve_session_type_known_holiday_and_circuit_break(self):
        # Lunar New Year 2024 is in the precomputed holiday list: 2024-02-12 is holiday
        hol = SessionDate.from_pydate(date(2024, 2, 12))
        stype = self.cn.resolve_session_type(hol)
        self.assertEqual(stype, SessionType.NON_TRADING)

        # Circuit break date from list (2016-01-04)
        cb = SessionDate.from_pydate(date(2016, 1, 4))
        stype_cb = self.cn.resolve_session_type(cb)
        self.assertEqual(stype_cb, SessionType.CIRCUIT_BREAK)

        # A normal trading date
        normal = SessionDate.from_pydate(date(2024, 2, 8))
        stype_n = self.cn.resolve_session_type(normal)
        self.assertEqual(stype_n, SessionType.NORMINAL)

        # Weekend (Saturday) should be non-trading
        sat = SessionDate.from_pydate(date(2024, 2, 10))
        self.assertEqual(self.cn.resolve_session_type(sat), SessionType.NON_TRADING)

    def test_trade_calendar_includes_circuit_break_and_skips_holidays(self):
        start = SessionDate.from_pydate(date(2016, 1, 1))
        end = SessionDate.from_pydate(date(2016, 1, 10))
        drange = self.cn.trade_calendar(start, end)
        # Ensure the circuit break date appears and is marked appropriately
        found_cb = False
        for d in drange:
            if (d.year, d.month, d.day) == (2016, 1, 4):
                found_cb = True
                self.assertEqual(d.session_type, SessionType.CIRCUIT_BREAK)
        self.assertTrue(found_cb)

    def test_resolve_session_and_auction_phases(self):
        # Preopen (before 09:15)
        self.assertEqual(self.cn.resolve_session_phase(py_time(9, 0)), SessionPhase.PREOPEN)
        # Open auction window
        self.assertEqual(self.cn.resolve_session_phase(py_time(9, 16)), SessionPhase.OPEN_AUCTION)
        # During break
        self.assertEqual(self.cn.resolve_session_phase(py_time(12, 0)), SessionPhase.BREAK)

        # Auction phases
        self.assertEqual(self.cn.resolve_auction_phase(py_time(9, 16)), AuctionPhase.ACTIVE)
        self.assertEqual(self.cn.resolve_auction_phase(py_time(9, 21)), AuctionPhase.NO_CANCEL)
        self.assertEqual(self.cn.resolve_auction_phase(py_time(9, 25)), AuctionPhase.UNCROSSING)

    def test_call_auction_and_break_structs(self):
        # open call auction fields
        oca = self.cn.open_call_auction
        self.assertIsNotNone(oca)
        self.assertEqual(oca.auction_start.hour, 9)
        self.assertEqual(oca.auction_start.minute, 15)
        self.assertEqual(oca.active.start_time.hour, 9)
        self.assertEqual(oca.active.start_time.minute, 15)

        # session break
        breaks = self.cn.session_breaks
        self.assertTrue(len(breaks) >= 1)
        br = breaks[0]
        self.assertEqual(br.break_start.hour, 11)
        self.assertEqual(br.break_start.minute, 30)
        self.assertAlmostEqual(br.break_length_seconds, 5400.0)


class TestProfileCompatible(unittest.TestCase):
    def setUp(self):
        # Activate CN profile for PROFILE singleton
        PROFILE_CN.activate()

    def tearDown(self):
        PROFILE_CN.deactivate()

    def test_time_and_timestamp_to_seconds(self):
        # 10:00 local -> ts = 36000
        t = py_time(10, 0)
        self.assertAlmostEqual(PROFILE.time_to_seconds(t, break_adjusted=False), 36000.0)
        # break_adjusted: elapsed since session start (9:30 -> 34200): 1800
        self.assertAlmostEqual(PROFILE.time_to_seconds(t, break_adjusted=True), 1800.0)

        # unix timestamp that maps to local 10:00 on a known trading date
        unix = datetime(2024, 2, 8, 10, 0, tzinfo=PROFILE.time_zone).timestamp()
        self.assertAlmostEqual(PROFILE.timestamp_to_seconds(unix, break_adjusted=False), 36000.0)
        self.assertAlmostEqual(PROFILE.timestamp_to_seconds(unix, break_adjusted=True), 1800.0)

        # Also test a time in the latter half of the session (14:00)
        t2 = py_time(14, 0)
        # raw seconds-of-day
        self.assertAlmostEqual(PROFILE.time_to_seconds(t2, break_adjusted=False), 14 * 3600)
        # break_adjusted: subtract lunch break (5400s) and session start offset (9:30 -> 34200)
        # elapsed since session start adjusted = (50400 - 34200) - 5400 = 10800
        self.assertAlmostEqual(PROFILE.time_to_seconds(t2, break_adjusted=True), 10800.0)

        unix2 = datetime(2024, 2, 8, 14, 0, tzinfo=PROFILE.time_zone).timestamp()
        self.assertAlmostEqual(PROFILE.timestamp_to_seconds(unix2, break_adjusted=False), 50400.0)
        self.assertAlmostEqual(PROFILE.timestamp_to_seconds(unix2, break_adjusted=True), 10800.0)

    def test_break_adjusted_and_trading_time_between(self):
        # break_adjusted should behave same as time_to_seconds for ts
        self.assertAlmostEqual(PROFILE.break_adjusted(36000.0), 1800.0)

        from datetime import datetime
        st = datetime(2024, 2, 1, 10, 0)
        et = datetime(2024, 2, 1, 14, 0)
        # 10:00-11:30 = 5400, 13:00-14:00 = 3600 -> total 9000
        self.assertAlmostEqual(PROFILE.trading_time_between(st, et), 9000.0)

    def test_is_market_and_auction_session(self):
        self.assertTrue(PROFILE.is_market_session(py_time(10, 0)))
        self.assertFalse(PROFILE.is_market_session(py_time(9, 0)))
        self.assertTrue(PROFILE.is_auction_session(py_time(9, 16)))

    def test_trade_calendar_and_trading_days_helpers(self):
        # get trade calendar around lunar new year 2024
        tc_list = PROFILE.trade_calendar(date(2024, 2, 1), date(2024, 2, 29))
        # ensure that holiday 2024-02-12 is not in tc_list
        dates = [d.to_pydate() for d in tc_list]
        self.assertNotIn(date(2024, 2, 12), dates)

        # trading_days_before and after: find index for 2024-02-08
        target = date(2024, 2, 8)
        idx = dates.index(target)
        prev_expected = dates[idx - 1]
        next_expected = dates[idx + 1]

        self.assertEqual(PROFILE.trading_days_before(target, 1), prev_expected)
        self.assertEqual(PROFILE.trading_days_after(target, 1), next_expected)

        # trading_days_between from 2024-02-01 to 2024-02-08 should equal index difference
        td_between = PROFILE.trading_days_between(date(2024, 2, 1), date(2024, 2, 8))
        # compute expected via dates list
        start_idx = dates.index(date(2024, 2, 1))
        end_idx = dates.index(date(2024, 2, 8))
        self.assertEqual(td_between, end_idx - start_idx)

        # nearest_trading_date previous/next for holiday 2024-02-12
        nd_prev = PROFILE.nearest_trading_date(date(2024, 2, 12), method='previous')
        nd_next = PROFILE.nearest_trading_date(date(2024, 2, 12), method='next')
        self.assertEqual(nd_prev, target)
        self.assertEqual(nd_next, next_expected)

    def test_is_trading_day(self):
        self.assertFalse(PROFILE.is_trading_day(date(2024, 2, 12)))
        self.assertTrue(PROFILE.is_trading_day(date(2024, 2, 8)))


if __name__ == "__main__":
    unittest.main()
