import datetime
import unittest
from math import isnan

from algo_engine.base import TransactionData, TransactionSide
from algo_engine.engine import MDS
from algo_engine.exchange_profile import PROFILE_CN

from tests.c_state_access_toolkit import market_price


class TestCStateAccess(unittest.TestCase):
    """The C reads the strategy state updater uses: the MDS ticker price."""

    @classmethod
    def setUpClass(cls) -> None:
        PROFILE_CN.activate()

        cls.td = TransactionData(
            ticker='C_STATE.TEST',
            price=42.5,
            volume=10,
            timestamp=datetime.datetime(2024, 11, 11, 10, 0).timestamp(),
            side=TransactionSide.LongFilled,
        )

        MDS.on_market_data(cls.td)

    @classmethod
    def tearDownClass(cls) -> None:
        MDS.clear()
        PROFILE_CN.deactivate()

    def test_00_the_c_price_read_matches_the_python_one(self) -> None:
        """The bytemap index lookup answers the same price as the Python accessor."""
        self.assertEqual(market_price(b'C_STATE.TEST'), 42.5)
        self.assertEqual(market_price(b'C_STATE.TEST'), MDS.get_market_price('C_STATE.TEST'))

        self.assertTrue(isnan(market_price(b'NO_SUCH_TICKER')))


if __name__ == '__main__':
    unittest.main()
