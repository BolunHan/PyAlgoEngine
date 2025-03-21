import datetime
import unittest
import pickle
import timeit
from market_data_wrapper import MarketData as PyMarketData, TransactionData as PyTransactionData, TransactionSide as PyTransactionSide, BarData as PyBarData, DailyBar as PyDailyBar
import market_data as mdc
import algo_engine  # pip install PyAlgoEngine


class TransactionDataTest(unittest.TestCase):
    def run_test(self):
        self.transaction_data_init()
        self.transaction_data_ids()
        self.transaction_data_flow()
        self.transaction_data_pickle()
        self.transaction_side_enum()

    def transaction_data_init(self):
        """Test basic initialization of TransactionData"""
        # Test Cython class
        ctd = mdc.TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1)  # Using 1 for LongOpen
        ntd = algo_engine.base.TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1)
        wtd = PyTransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1)

        print('\n---\n')
        print('cython td init time per 10k: ', timeit.timeit(lambda: mdc.TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1), number=10000))
        print('native td init time per 10k: ', timeit.timeit(lambda: algo_engine.base.TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1), number=10000))
        print('wrapped td init time per 10k: ', timeit.timeit(lambda: PyTransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1), number=10000))

        print('\n---\n')
        print('cython td pickle time per 10k: ', timeit.timeit(lambda: ctd.from_bytes(ctd.to_bytes()), number=10000))
        print('native td pickle time per 10k: ', timeit.timeit(lambda: pickle.loads(pickle.dumps(ntd)), number=10000))
        print('wrapped td pickle time per 10k: ', timeit.timeit(lambda: pickle.loads(pickle.dumps(wtd)), number=10000))

        print('\n---\n')
        print('cython td access ticker time per 10k', timeit.timeit(lambda: ctd.ticker, number=10000))
        print('native td access ticker time per 10k', timeit.timeit(lambda: ntd.ticker, number=10000))
        print('wrapped td access ticker time per 10k', timeit.timeit(lambda: wtd.ticker, number=10000))

        print('\n---\n')
        print('cython td access id time per 10k', timeit.timeit(lambda: ctd.transaction_id, number=10000))
        print('native td access id time per 10k', timeit.timeit(lambda: ntd.transaction_id, number=10000))
        print('wrapped td access id time per 10k', timeit.timeit(lambda: wtd.transaction_id, number=10000))

        print('\n---\n')
        print('cython td access data time per 10k', timeit.timeit(lambda: ctd.price, number=10000))
        print('native td access data time per 10k', timeit.timeit(lambda: ntd.price, number=10000))
        print('wrapped td access data time per 10k', timeit.timeit(lambda: wtd.price, number=10000))

        print('\n---\n')
        print('cython td access side time per 10k', timeit.timeit(lambda: ctd.side, number=10000))
        print('native td access side time per 10k', timeit.timeit(lambda: ntd.side, number=10000))
        print('wrapped td access side time per 10k', timeit.timeit(lambda: wtd.side, number=10000))

        self.assertEqual(ctd.ticker, "AAPL")
        self.assertEqual(ctd.timestamp, 1614556800.0)
        self.assertEqual(ctd.price, 150.0)
        self.assertEqual(ctd.volume, 100.0)
        self.assertEqual(ctd.side, 1)  # LongOpen
        self.assertEqual(ctd.notional, 150.0 * 100.0)  # price * volume

        # Test Python wrapper
        py_td = PyTransactionData("AAPL", 1614556800.0, 150.0, 100.0, PyTransactionSide.LongOpen)
        self.assertEqual(py_td.ticker, "AAPL")
        self.assertEqual(py_td.price, 150.0)
        self.assertEqual(py_td.volume, 100.0)
        self.assertEqual(py_td.side, PyTransactionSide.LongOpen)

    def transaction_data_ids(self):
        """Test ID handling in TransactionData"""
        # Test with integer IDs
        td = mdc.TransactionData("AAPL", 1614556800.0, 150.0, 100.0, 1,
                                 transaction_id=12345, buy_id=67890, sell_id=54321)

        self.assertEqual(td.transaction_id, 12345)
        self.assertEqual(td.buy_id, 67890)
        self.assertEqual(td.sell_id, 54321)

        # Test with string IDs
        td2 = mdc.TransactionData("AAPL", 1614556800.0, 150.0, 100.0, 1,
                                  transaction_id="TX12345", buy_id="BUY67890", sell_id="SELL54321")

        self.assertEqual(td2.transaction_id, "TX12345")
        self.assertEqual(td2.buy_id, "BUY67890")
        self.assertEqual(td2.sell_id, "SELL54321")

    def transaction_data_flow(self):
        """Test flow calculation in TransactionData"""
        # Test LongOpen (positive flow)
        td_long = mdc.TransactionData("AAPL", 1614556800.0, 150.0, 100.0, 1)  # LongOpen
        self.assertEqual(td_long.flow, 100.0)  # positive for long

        # Test ShortOpen (negative flow)
        td_short = mdc.TransactionData("AAPL", 1614556800.0, 150.0, 100.0, -2)  # ShortOpen
        self.assertEqual(td_short.flow, -100.0)  # negative for short

    def transaction_data_pickle(self):
        """Test pickle serialization of Python wrapper"""
        py_td = PyTransactionData("000016.SH", 1614556800.0, 150.0, 100.0, PyTransactionSide.LongOpen, transaction_id=12345)
        py_td.abc = 111

        # Pickle and unpickle
        pickled = pickle.dumps(py_td)
        unpickled = pickle.loads(pickled)

        self.assertEqual(unpickled.ticker, "000016.SH")
        self.assertEqual(unpickled.price, 150.0)
        self.assertEqual(unpickled.volume, 100.0)
        self.assertEqual(unpickled.side, PyTransactionSide.LongOpen)
        self.assertEqual(unpickled.transaction_id, 12345)
        self.assertEqual(unpickled.abc, 111)

    def transaction_side_enum(self):
        """Test TransactionSide enum in Python wrapper"""
        # Use the specific enum value directly instead of relying on the constructor
        side_enum = PyTransactionSide.LongOpen
        py_td = PyTransactionData("AAPL", 1614556800.0, 150.0, 100.0, side_enum)

        # Check that side is returned as an IntEnum
        self.assertIsInstance(py_td.side, PyTransactionSide)
        self.assertEqual(py_td.side, PyTransactionSide.LongOpen)
        self.assertEqual(py_td.side.value, 1)
        # Compare with the actual enum instance name rather than a string
        self.assertEqual(py_td.side.name, side_enum.name)


class BarDataTest(unittest.TestCase):
    def run_test(self):
        self.bar_data_init()
        self.bar_data_serialization()
        self.daily_bar_test()

    def bar_data_init(self):
        """Test basic initialization of BarData"""
        # Test Cython class
        bar = mdc.BarData("AAPL", 1614556800.0, 155.0, 145.0, 150.0, 152.0, 1000.0, 150000.0, 50)

        self.assertEqual(bar.ticker, "AAPL")
        self.assertEqual(bar.timestamp, 1614556800.0)
        self.assertEqual(bar.high_price, 155.0)
        self.assertEqual(bar.low_price, 145.0)
        self.assertEqual(bar.open_price, 150.0)
        self.assertEqual(bar.close_price, 152.0)
        self.assertEqual(bar.volume, 1000.0)
        self.assertEqual(bar.notional, 150000.0)
        self.assertEqual(bar.trade_count, 50)

        # Test Python wrapper
        py_bar = PyBarData(ticker="AAPL", timestamp=1614556800.0, bar_span=3600.0, high_price=155.0, low_price=145.0, open_price=150.0, close_price=152.0, volume=1000.0, notional=150000.0, trade_count=50)
        self.assertEqual(py_bar.ticker, "AAPL")
        self.assertEqual(py_bar.start_timestamp, 1614553200.0)
        self.assertEqual(py_bar.bar_span.total_seconds(), 3600.0)
        self.assertEqual(py_bar.vwap, 150.0)  # 150000.0 / 1000.0

    def bar_data_serialization(self):
        """Test to_bytes and from_bytes methods for BarData"""
        # Create a bar data object
        py_bar = PyBarData(ticker="AAPL", timestamp=1614556800.0, bar_span=3600.0, high_price=155.0, low_price=145.0, open_price=150.0, close_price=152.0, volume=1000.0, notional=150000.0, trade_count=50)

        # Convert to bytes
        bar_bytes = py_bar.to_bytes()

        # Create a new instance from bytes
        bar2 = mdc.BarData.from_bytes(bar_bytes)

        # Verify the data is the same
        self.assertEqual(bar2.ticker, "AAPL")
        self.assertEqual(bar2.timestamp, 1614556800.0)
        self.assertEqual(bar2.bar_span, 3600.0)
        self.assertEqual(bar2.high_price, 155.0)
        self.assertEqual(bar2.low_price, 145.0)
        self.assertEqual(bar2.open_price, 150.0)
        self.assertEqual(bar2.close_price, 152.0)
        self.assertEqual(bar2.volume, 1000.0)
        self.assertEqual(bar2.notional, 150000.0)
        self.assertEqual(bar2.trade_count, 50)
        self.assertEqual(bar2.start_timestamp, 1614553200.0)

    def daily_bar_test(self):
        """Test DailyBar class"""
        # Test Cython class
        md = datetime.date.today()
        daily_bar = PyDailyBar(ticker="AAPL", market_date=md, high_price=155.0, low_price=145.0, open_price=150.0, close_price=152.0, volume=1000.0, notional=150000.0, trade_count=50)

        self.assertEqual(daily_bar.ticker, "AAPL")
        self.assertEqual(daily_bar.market_date, md)
        self.assertEqual(daily_bar.high_price, 155.0)
        self.assertEqual(daily_bar.low_price, 145.0)
        self.assertEqual(daily_bar.open_price, 150.0)
        self.assertEqual(daily_bar.close_price, 152.0)
        self.assertEqual(daily_bar.volume, 1000.0)
        self.assertEqual(daily_bar.notional, 150000.0)
        self.assertEqual(daily_bar.trade_count, 50)
        self.assertEqual(daily_bar.bar_span.days, 1)  # Should be 1 day in seconds

        # Test serialization
        daily_bar2 = pickle.loads(pickle.dumps(daily_bar))

        self.assertEqual(daily_bar2.ticker, "AAPL")
        self.assertEqual(daily_bar2.high_price, 155.0)
        self.assertEqual(daily_bar2.bar_span.days, 1)


if __name__ == "__main__":
    print("Running TransactionData tests...")
    transaction_test = TransactionDataTest()
    transaction_test.run_test()

    print("\nRunning BarData tests...")
    bar_test = BarDataTest()
    bar_test.run_test()

    print("\nAll tests completed successfully!")
