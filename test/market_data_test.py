import pickle
import random
import time
import timeit
import unittest
import uuid

# install with `pip install PyAlgoEngine`
from algo_engine.base import TransactionSide, TransactionData, TransactionDirection as Direction, TransactionOffset as Offset, MarketDataBuffer, TickData, OrderData, BarData


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
        ntd = TransactionData(ticker="000016.SH", timestamp=1614556800.0, price=150.0, volume=100.0, side=1)

        print('\n---\n')
        print('native td init time per 10k: ', timeit.timeit(lambda: TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=1), number=10000))

        print('\n---\n')
        print('native td pickle time per 10k: ', timeit.timeit(lambda: pickle.loads(pickle.dumps(ntd)), number=10000))

        print('\n---\n')
        print('native td access ticker time per 10k', timeit.timeit(lambda: ntd.ticker, number=10000))

        print('\n---\n')
        print('native td access id time per 10k', timeit.timeit(lambda: ntd.transaction_id, number=10000))

        print('\n---\n')
        print('native td access data time per 10k', timeit.timeit(lambda: ntd.price, number=10000))

        print('\n---\n')
        print('native td access side time per 10k', timeit.timeit(lambda: ntd.side, number=10000))

        self.assertEqual(ntd.ticker, "000016.SH")
        self.assertEqual(ntd.timestamp, 1614556800.0)
        self.assertEqual(ntd.price, 150.0)
        self.assertEqual(ntd.volume, 100.0)
        self.assertEqual(ntd.side, 1)  # LongOpen
        self.assertEqual(ntd.notional, 150.0 * 100.0)  # price * volume

    def transaction_data_ids(self):
        """Test ID handling in TransactionData"""
        # Test with integer IDs
        td = TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=Direction.DIRECTION_LONG | Offset.OFFSET_CLOSE, transaction_id=12345, buy_id='abcde', sell_id=(_id := uuid.uuid4()))

        self.assertEqual(td.transaction_id, 12345)
        self.assertEqual(td.buy_id, 'abcde')
        self.assertEqual(td.sell_id, _id)
        self.assertEqual(td.side, TransactionSide.SIDE_LONG_CLOSE)

    def transaction_data_flow(self):
        """Test flow calculation in TransactionData"""
        # Test LongOpen (positive flow)
        td = TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=Direction.DIRECTION_LONG | Offset.OFFSET_CLOSE, transaction_id=12345, buy_id='abcde', sell_id=(_id := uuid.uuid4()))
        self.assertEqual(td.volume_flow, 100.0)  # positive for long

        # Test ShortOpen (negative flow)
        td = TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=Direction.DIRECTION_SHORT | Offset.OFFSET_OPEN, transaction_id=12345, buy_id='abcde', sell_id=(_id := uuid.uuid4()))
        self.assertEqual(td.volume_flow, -100.0)  # negative for short

    def transaction_data_pickle(self):
        """Test pickle serialization of Python wrapper"""
        td = TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=Direction.DIRECTION_LONG | Offset.OFFSET_CLOSE, transaction_id=12345, buy_id='abcde', sell_id=(_id := uuid.uuid4()), abc=111)
        td.efg = 234

        # Pickle and unpickle
        pickled = pickle.dumps(td)
        unpickled = pickle.loads(pickled)

        self.assertEqual(unpickled.ticker, "AAPL")
        self.assertEqual(unpickled.price, 150.0)
        self.assertEqual(unpickled.volume, 100.0)
        self.assertEqual(unpickled.side, TransactionSide.SIDE_LONG_CLOSE)
        self.assertEqual(unpickled.transaction_id, 12345)
        self.assertEqual(unpickled.abc, 111)
        self.assertEqual(unpickled.efg, 234)

    def transaction_side_enum(self):
        """Test TransactionSide enum in Python wrapper"""
        # Use the specific enum value directly instead of relying on the constructor
        side_enum = TransactionSide.SIDE_LONG_OPEN
        td = TransactionData(ticker="AAPL", timestamp=1614556800.0, price=150.0, volume=100.0, side=Direction.DIRECTION_LONG | Offset.OFFSET_OPEN, transaction_id=12345, buy_id='abcde', sell_id=(_id := uuid.uuid4()))

        # Check that side is returned as an IntEnum
        self.assertIsInstance(td.side, TransactionSide)
        self.assertEqual(td.side, side_enum)
        self.assertEqual(td.side.value, Direction.DIRECTION_LONG + Offset.OFFSET_OPEN)
        # Compare with the actual enum instance name rather than a string
        self.assertEqual(td.side.name, side_enum.name)


class MarketDataBufferTest(unittest.TestCase):
    def run_test(self):
        self.test_init()
        self.test_serialization()
        self.test_update_buffer()

    @classmethod
    def _tick_gen(cls, size: int, order_book_size: int = 5, buffer: MarketDataBuffer = None):
        ticker = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX"]

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(ticker)

            # Random price between 1 and 1000 with 2 decimal places
            price = round(random.uniform(100, 1000), 2)
            bid_price = price - round(random.uniform(0, 1), 2)
            ask_price = price + round(random.uniform(0, 1), 2)
            bid_volume = random.randint(1, 100)
            ask_volume = random.randint(1, 100)

            if order_book_size > 0:
                order_book = {'bid_price_1': bid_price, 'ask_price_1': ask_price, 'bid_volume_1': bid_volume, 'ask_volume_1': ask_volume}
                for lv in range(2, order_book_size):
                    order_book[f'bid_price_{lv}'] = order_book[f'bid_price_{lv - 1}'] - round(random.uniform(0, 1), 2)
                    order_book[f'ask_price_{lv}'] = order_book[f'ask_price_{lv - 1}'] + round(random.uniform(0, 1), 2)
                    order_book[f'bid_volume_{lv}'] = random.uniform(1, 100)
                    order_book[f'ask_volume_{lv}'] = random.uniform(1, 100)
                    order_book[f'bid_orders_{lv}'] = random.randint(1, 100)
                    order_book[f'ask_orders_{lv}'] = random.randint(1, 100)
            else:
                order_book = {}

            # Random volume between 1 and 10000
            ttl_volume = random.randint(1, 10000)
            ttl_notional = ttl_volume * (price + round(random.uniform(-5, 5), 2))
            ttl_trade = int(random.uniform(0, ttl_volume))

            # Random timestamp within the last 24 hours
            timestamp = ts - random.uniform(0, 86400)

            if buffer is not None:
                buffer.update(
                    dtype=32,
                    ticker=_ticker,
                    timestamp=timestamp,
                    last_price=price,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_volume=bid_volume,
                    ask_volume=ask_volume,
                    total_traded_volume=ttl_volume,
                    total_trade_notional=ttl_notional,
                    total_trade_count=ttl_trade,
                    **order_book
                )

            data = TickData(
                ticker=_ticker,
                timestamp=timestamp,
                last_price=price,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                total_traded_volume=ttl_volume,
                total_trade_notional=ttl_notional,
                total_trade_count=ttl_trade,
                **order_book
            )
            data_list.append(data)

        return data_list

    @classmethod
    def _td_gen(cls, size: int, buffer: MarketDataBuffer = None):
        ticker = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX"]

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(ticker)

            # Random price between 1 and 1000 with 2 decimal places
            price = round(random.uniform(1, 1000), 2)

            # Random volume between 1 and 10000
            volume = random.randint(1, 10000)

            # Random timestamp within the last 24 hours
            timestamp = ts - random.uniform(0, 86400)

            # Random direction and offset using bitwise OR
            # direction = random.choice(list(Direction))
            # offset = random.choice(list(Offset))
            side = random.choice(list(TransactionSide))

            if buffer is not None:
                buffer.update(
                    dtype=20,
                    ticker=_ticker,
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    side=side
                )

            data = TransactionData(
                ticker=_ticker,
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side
            )
            data_list.append(data)

        return data_list

    def _od_gen(self, size: int, buffer: MarketDataBuffer = None):
        ticker = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX"]

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(ticker)

            # Random price between 1 and 1000 with 2 decimal places
            price = round(random.uniform(1, 1000), 2)

            # Random volume between 1 and 10000
            volume = random.randint(1, 10000)

            # Random timestamp within the last 24 hours
            timestamp = ts - random.uniform(0, 86400)

            # Random direction and offset using bitwise OR
            direction = random.choice([Direction.DIRECTION_LONG, Direction.DIRECTION_SHORT])
            offset = random.choice([Offset.OFFSET_ORDER, Offset.OFFSET_CANCEL])

            if buffer is not None:
                buffer.update(
                    dtype=30,
                    ticker=_ticker,
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    side=direction | offset
                )

            data = OrderData(
                ticker=_ticker,
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=direction | offset
            )
            data_list.append(data)

        return data_list

    def _bar_gen(self, size: int, buffer: MarketDataBuffer = None):
        ticker = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX"]

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(ticker)

            # Random price between 1 and 1000 with 2 decimal places
            close_price = round(random.uniform(1, 1000), 2)
            open_price = close_price * random.uniform(0.9, 1.1)
            high_price = max(open_price, close_price) * random.uniform(1, 1.05)
            low_price = min(open_price, close_price) * random.uniform(0.95, 1.)

            # Random volume between 1 and 10000
            volume = random.uniform(100, 10000)
            notional = volume * random.uniform(low_price, high_price)
            n_trades = random.randint(1, int(volume / 10))

            # Random timestamp within the last 24 hours
            timestamp = ts - random.uniform(0, 86400)

            if buffer is not None:
                buffer.update(
                    dtype=40,
                    ticker=_ticker,
                    timestamp=timestamp,
                    open_price=open_price,
                    close_price=close_price,
                    high_price=high_price,
                    low_price=low_price,
                    volume=volume,
                    notional=notional,
                    trades=n_trades,
                    bar_span=5 * 60
                )

            data = BarData(
                ticker=_ticker,
                timestamp=timestamp,
                open_price=open_price,
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume,
                notional=notional,
                trades=n_trades,
                bar_span=5 * 60
            )
            data_list.append(data)

        return data_list

    def test_init(self):
        import ctypes
        from multiprocessing import RawArray
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            max_size=50
        )

        data_list = self._td_gen(10) + self._tick_gen(10) + self._bar_gen(10)
        for td in data_list:
            buf.push(td)

        sorted_data_list = sorted(data_list, key=lambda _td: _td.timestamp)

        for td, ctd in zip(buf, sorted_data_list):
            if isinstance(td, TransactionData):
                self.assertEqual(td.timestamp, ctd.timestamp)
                self.assertEqual(td.price, ctd.price)
                self.assertEqual(td.volume, ctd.volume)
                self.assertEqual(td.side, ctd.side)
            else:
                self.assertEqual(td.to_bytes(), ctd.to_bytes())

    def test_serialization(self):
        import ctypes
        from multiprocessing import RawArray
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            max_size=10
        )

        data_list = self._td_gen(1) + self._tick_gen(1) + self._od_gen(1) + self._bar_gen(1)

        for td in data_list:
            buf.push(td)

        data = buf.to_bytes()
        new_buf = MarketDataBuffer.from_buffer(data)

        print('--- buffer push test ---\n')
        for i, (td, ctd) in enumerate(zip(buf, new_buf)):
            if isinstance(td, TransactionData):
                self.assertEqual(td.timestamp, ctd.timestamp)
                self.assertEqual(td.price, ctd.price)
                self.assertEqual(td.volume, ctd.volume)
                self.assertEqual(td.side, ctd.side)
            else:
                self.assertEqual(td.ticker, ctd.ticker)
            print(f'buffer {i} {type(td).__name__} validated!')

    def test_update_buffer(self):
        import ctypes
        from multiprocessing import RawArray
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            max_size=50
        )
        data_list = []
        data_list += self._td_gen(1, buffer=buf)
        data_list += self._tick_gen(1, buffer=buf)
        data_list += self._od_gen(1, buffer=buf)
        data_list += self._bar_gen(1, buffer=buf)

        sorted_data_list = sorted(data_list, key=lambda _td: _td.timestamp)

        print('--- buffer update test ---\n')
        for i, (td, ctd) in enumerate(zip(buf, sorted_data_list)):
            if isinstance(td, TransactionData):
                self.assertEqual(td.timestamp, ctd.timestamp)
                self.assertEqual(td.price, ctd.price)
                self.assertEqual(td.volume, ctd.volume)
                self.assertEqual(td.side, ctd.side)

            if isinstance(ctd, TickData):
                self.assertEqual(td.ticker, ctd.ticker)
                self.assertEqual(td.timestamp, ctd.timestamp)
                self.assertEqual(td.bid_price, ctd.bid_price)
                self.assertEqual(td.ask_price, ctd.ask_price)
                self.assertEqual(td.bid_volume, ctd.bid_volume)
                self.assertEqual(td.ask_volume, ctd.ask_volume)
                self.assertEqual(td.total_traded_volume, ctd.total_traded_volume)
                self.assertEqual(td.total_traded_notional, ctd.total_traded_notional)
                self.assertEqual(td.total_trade_count, ctd.total_trade_count)
                self.assertEqual(td.bid.price, ctd.bid.price)
                self.assertEqual(td.ask.price, ctd.ask.price)
                self.assertEqual(td.bid.volume, ctd.bid.volume)
                self.assertEqual(td.ask.volume, ctd.ask.volume)

            self.assertEqual(td.to_bytes(), ctd.to_bytes())
            print(f'buffer {i} {type(td).__name__} validated!')


if __name__ == "__main__":
    print("Running TransactionData tests...")
    transaction_test = TransactionDataTest()
    transaction_test.run_test()

    print("\nRunning BarData tests...")
    # bar_test = BarDataTest()
    # bar_test.run_test()

    print("\nAll tests completed successfully!")
