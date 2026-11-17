import unittest

from algo_engine.base import TransactionSide
from algo_engine.engine import AlgoStatus, Passive


SIDE_LONG = TransactionSide.SIDE_LONG_OPEN


class TestAlgo(unittest.TestCase):
    def make_algo(self, ticker='AAPL', **kwargs):
        return Passive(ticker, SIDE_LONG, 100., **kwargs)

    def test_00_lifecycle_and_fields(self) -> None:
        record = self.make_algo()

        self.assertEqual(record.algo_id.version, 4)
        self.assertEqual(record.ticker, 'AAPL')
        self.assertEqual(record.side, SIDE_LONG)
        self.assertEqual(record.status, AlgoStatus.Idle)
        self.assertEqual(record.target_volume, 100.)
        self.assertEqual(record.multiplier, 1.)
        self.assertEqual(record.working_volume, 0.)
        self.assertEqual(record.exposure_volume, 0.)
        self.assertEqual(record.filled_volume, 0.)
        self.assertEqual(record.filled_notional, 0.)
        self.assertEqual(record.fee, 0.)
        self.assertEqual(record.cash_flow, 0.)
        self.assertEqual(record.n_working_orders, 0)
        self.assertGreater(record.algo_reg_id, 0)

        self.assertIsNone(record.ts_started)
        self.assertIsNone(record.ts_finished)

    def test_01_generated_identity(self) -> None:
        first = Passive('AAPL', SIDE_LONG, 100.)
        second = Passive('AAPL', SIDE_LONG, 100.)

        self.assertNotEqual(first.algo_id, second.algo_id)


if __name__ == '__main__':
    unittest.main()
