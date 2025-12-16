import unittest

from algo_engine.base.c_market_data_ng.c_internal import InternalData
from algo_engine.base.c_market_data_ng.c_market_data import MarketData


class TestInternalData(unittest.TestCase):

    def test_init_sets_fields(self):
        data = InternalData(ticker='600010.SH', timestamp=123.234, code=123)
        self.assertEqual(data.ticker, '600010.SH')
        self.assertEqual(data.timestamp, 123.234)
        self.assertEqual(data.code, 123)
        self.assertTrue(data.owner)
        self.assertIsNotNone(data.data_addr)

    def test_round_trip_serialization(self):
        data = InternalData(ticker='600010.SH', timestamp=123.234, code=123)
        data_bytes = data.to_bytes()

        regen = InternalData.from_bytes(data_bytes)

        self.assertEqual(regen.ticker, '600010.SH')
        self.assertEqual(regen.timestamp, 123.234)
        self.assertEqual(regen.code, 123)
        self.assertTrue(regen.owner)
        self.assertNotEqual(regen.data_addr, data.data_addr)

    def test_pickle_serialization(self):
        import pickle

        data = InternalData(ticker='600010.SH', timestamp=123.234, code=123)
        data_pickled = pickle.dumps(data)

        regen = pickle.loads(data_pickled)

        self.assertEqual(regen.ticker, '600010.SH')
        self.assertEqual(regen.timestamp, 123.234)
        self.assertEqual(regen.code, 123)
        self.assertTrue(regen.owner)
        self.assertNotEqual(regen.data_addr, data.data_addr)

        data_bin = data.to_bytes()
        regen_2 = MarketData.from_bytes(data_bin)
        self.assertEqual(regen_2.ticker, '600010.SH')
        self.assertEqual(regen_2.timestamp, 123.234)
        self.assertEqual(regen_2.code, 123)
        self.assertTrue(regen_2.owner)
        self.assertNotEqual(regen_2.data_addr, data.data_addr)

        self.assertEqual(regen.to_bytes(), data_bin)
        self.assertEqual(regen_2.to_bytes(), data_bin)


if __name__ == '__main__':
    unittest.main()
