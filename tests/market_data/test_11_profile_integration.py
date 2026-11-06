import datetime
import unittest

from algo_engine.base.c_market_data import TransactionData, TransactionSide
from algo_engine.exchange_profile import PROFILE_CN, SessionPhase


class TestMarketDataProfileIntegration(unittest.TestCase):
    """Integration tests for MarketData TransactionData with CN exchange profile.

    These tests verify that TransactionData.session_time is assigned a SessionPhase
    according to the active CN profile for several representative timestamps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure CN profile is activated for these integration tests
        PROFILE_CN.activate()

    def test_transaction_session_phases(self):
        """Three sample transactions should map to expected session phases.

        - td_1 at 09:45 should be in the continuous session
        - td_2 at 11:35 should be in a break session
        - td_3 at 09:21 should be in the open auction session
        """
        td_1 = TransactionData(
            ticker='600010.SH',
            price=102.34,
            volume=125,
            timestamp=datetime.datetime(2024, 11, 11, 9, 45).timestamp(),
            side=TransactionSide.LongFilled,
        )

        td_2 = TransactionData(
            ticker='600010.SH',
            price=105.20,
            volume=210,
            timestamp=datetime.datetime(2024, 11, 11, 11, 35).timestamp(),
            side=TransactionSide.ShortFilled,
        )

        td_3 = TransactionData(
            ticker='600010.SH',
            price=105.20,
            volume=210,
            timestamp=datetime.datetime(2024, 11, 11, 9, 21).timestamp(),
            side=TransactionSide.SIDE_LONG_CANCEL,
        )

        # Verify session phases
        self.assertEqual(td_1.session_time.session_phase, SessionPhase.CONTINUOUS)
        self.assertEqual(td_2.session_time.session_phase, SessionPhase.BREAK)
        self.assertEqual(td_3.session_time.session_phase, SessionPhase.OPEN_AUCTION)

        # Verify session_date matches the original datetime date
        self.assertEqual((td_1.session_date.year, td_1.session_date.month, td_1.session_date.day), (2024, 11, 11))
        self.assertEqual((td_2.session_date.year, td_2.session_date.month, td_2.session_date.day), (2024, 11, 11))
        self.assertEqual((td_3.session_date.year, td_3.session_date.month, td_3.session_date.day), (2024, 11, 11))

        # Verify session_time preserves the clock time used when creating the timestamp
        self.assertEqual(td_1.session_time.to_pytime(), datetime.time(9, 45))
        self.assertEqual(td_2.session_time.to_pytime(), datetime.time(11, 35))
        self.assertEqual(td_3.session_time.to_pytime(), datetime.time(9, 21))


if __name__ == "__main__":
    unittest.main()
