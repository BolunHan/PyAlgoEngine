import unittest
from datetime import date, time as py_time

from algo_engine.profile.c_exchange_profile import (
    ExchangeProfile,
    PROFILE_DEFAULT,
    SessionDate,
    SessionDateRange,
    SessionTime,
    CallAuction,
    SessionBreak,
    SessionType,
    SessionPhase,
    AuctionPhase,
)


class TestExchangeProfile(unittest.TestCase):
    def test_default_profile_basic_properties(self):
        dp = PROFILE_DEFAULT
        self.assertIsInstance(dp, ExchangeProfile)
        self.assertIsInstance(dp.profile_id, str)
        # session start/end are SessionTime
        self.assertIsInstance(dp.session_start, SessionTime)
        self.assertIsInstance(dp.session_end, SessionTime)
        # numeric properties
        self.assertIsInstance(dp.session_start_ts, float)
        self.assertIsInstance(dp.session_end_ts, float)
        self.assertIsInstance(dp.session_length_seconds, float)
        self.assertIsInstance(dp.tz_offset_seconds, float)
        # session_breaks is a tuple
        self.assertIsInstance(dp.session_breaks, tuple)

    def test_resolve_helpers_return_enums(self):
        dp = PROFILE_DEFAULT
        # resolve using SessionTime instance
        sp = dp.resolve_session_phase(SessionTime(9, 0))
        self.assertIsInstance(sp, SessionPhase)

        ap = dp.resolve_auction_phase(py_time(9, 0))
        self.assertIsInstance(ap, AuctionPhase)

        # resolve session type using a SessionDate and a python date
        sdate = SessionDate.from_pydate(date.today())
        st = dp.resolve_session_type(sdate)
        self.assertIsInstance(st, SessionType)

    def test_trade_calendar_and_call_auction_structs(self):
        dp = PROFILE_DEFAULT
        # Use a small date range within same month
        start = SessionDate.from_pydate(date(2024, 2, 27))
        end = SessionDate.from_pydate(date(2024, 3, 2))
        drange = dp.trade_calendar(start, end)
        self.assertIsInstance(drange, SessionDateRange)
        self.assertEqual(drange.start_date.to_pydate(), start.to_pydate())

        # call auctions may be present or None; ensure attribute access works
        if dp.open_call_auction is not None:
            self.assertIsInstance(dp.open_call_auction, CallAuction)
            self.assertIsInstance(dp.open_call_auction.auction_start, SessionTime)

        if dp.close_call_auction is not None:
            self.assertIsInstance(dp.close_call_auction, CallAuction)

        # session_breaks elements, if any, should be SessionBreak
        for br in dp.session_breaks:
            self.assertIsInstance(br, SessionBreak)
            self.assertIsInstance(br.break_start, SessionTime)


if __name__ == "__main__":
    unittest.main()

