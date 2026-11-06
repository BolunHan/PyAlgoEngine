import datetime
import unittest
from math import isnan

from algo_engine.base import TransactionData, TransactionSide
from algo_engine.engine import MDS
from algo_engine.exchange_profile import PROFILE_CN


class TestMDSBasic(unittest.TestCase):
    """Contract: the MDS singleton ingests market data and tracks market state.

    Expected behavior:
        - Virgin state: timestamp is NaN, market_date is None, n_subscribed == 0.
        - on_market_data(td) subscribes the ticker (once per ticker) and updates
          get_market_price, timestamp, market_time and market_date.
        - Timestamps are NOT filtered for monotonicity (no filter implemented):
          an older transaction still overwrites price/time state.

    Oracle: expected values are the literal fields of the TransactionData
    objects fed in; market_time/market_date derived from datetime literals.

    Notes:
        - Tests are sequential (test_NN order): the MDS singleton carries state
          from one test to the next by design.
        - PROFILE_CN is activated in setUpClass and deactivated in
          tearDownClass so profile state does not leak into other test modules
          (the legacy script activated it at import time, which polluted the
          exchange_profile suite during full-suite runs).
    """

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

        cls.td_1 = TransactionData(
            ticker='600010.SH',
            price=102.34,
            volume=125,
            timestamp=datetime.datetime(2024, 11, 11, 9, 45).timestamp(),
            side=TransactionSide.LongFilled,
        )
        cls.td_2 = TransactionData(
            ticker='600010.SH',
            price=105.20,
            volume=210,
            timestamp=datetime.datetime(2024, 11, 11, 11, 35).timestamp(),
            side=TransactionSide.ShortFilled,
        )
        cls.td_3 = TransactionData(
            ticker='600010.SH',
            price=102.13,
            volume=210,
            timestamp=datetime.datetime(2024, 11, 11, 9, 21).timestamp(),
            side=TransactionSide.SIDE_LONG_CANCEL,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        MDS.clear()
        PROFILE_CN.deactivate()

    def test_00_virgin_state(self) -> None:
        """MDS starts with NaN timestamp, no market date and no subscriptions."""
        self.assertTrue(isnan(MDS.timestamp))
        self.assertIsNone(MDS.market_date)
        self.assertEqual(MDS.n_subscribed, 0)

    def test_01_first_transaction_updates_state(self) -> None:
        """First transaction subscribes the ticker and sets price/time state."""
        MDS.on_market_data(self.td_1)

        self.assertEqual(MDS.get_market_price('600010.SH'), 102.34)
        self.assertEqual(MDS.timestamp, self.td_1.timestamp)
        self.assertEqual(MDS.market_time.replace(tzinfo=None), datetime.datetime(2024, 11, 11, 9, 45))
        self.assertEqual(MDS.market_date, datetime.date(2024, 11, 11))
        self.assertEqual(MDS.n_subscribed, 1)

    def test_02_second_transaction_advances_state(self) -> None:
        """A later transaction on the same ticker advances state, no new subscription."""
        MDS.on_market_data(self.td_2)

        self.assertEqual(MDS.get_market_price('600010.SH'), 105.20)
        self.assertEqual(MDS.timestamp, self.td_2.timestamp)
        self.assertEqual(MDS.market_time.replace(tzinfo=None), datetime.datetime(2024, 11, 11, 11, 35))
        self.assertEqual(MDS.market_date, datetime.date(2024, 11, 11))
        self.assertEqual(MDS.n_subscribed, 1)

    def test_03_non_monotonic_transaction_still_applies(self) -> None:
        """An out-of-order (older) transaction overwrites state — no monotonic filter."""
        MDS.on_market_data(self.td_3)

        self.assertEqual(MDS.get_market_price('600010.SH'), 102.13)
        self.assertEqual(MDS.timestamp, self.td_3.timestamp)
        self.assertEqual(MDS.market_time.replace(tzinfo=None), datetime.datetime(2024, 11, 11, 9, 21))
        self.assertEqual(MDS.market_date, datetime.date(2024, 11, 11))
        self.assertEqual(MDS.n_subscribed, 1)


if __name__ == '__main__':
    unittest.main()
