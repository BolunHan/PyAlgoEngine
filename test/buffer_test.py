import collections
import ctypes
import multiprocessing
import random
import time
import unittest
from multiprocessing import RawArray
from typing import Literal

from algo_engine.base.candlestick import BarData
from algo_engine.base.market_data_buffer import MarketDataBuffer, MarketDataRingBuffer, MarketDataConcurrentBuffer
from algo_engine.base.tick import TickData
from algo_engine.base.transaction import TransactionData, OrderData, TransactionSide, TransactionDirection as Direction, TransactionOffset as Offset


class MockData(object):
    ticker = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "PYPL", "ADBE", "NFLX"]

    @classmethod
    def generate_tick_data(cls, size: int, order_book_size: int = 5, buffer: MarketDataBuffer = None):

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(cls.ticker)

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
    def generate_trade_data(cls, size: int, buffer: MarketDataBuffer = None):

        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(cls.ticker)

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

    @classmethod
    def generate_order_data(cls, size: int, buffer: MarketDataBuffer = None):
        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(cls.ticker)

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

    @classmethod
    def generate_bar_data(cls, size: int, buffer: MarketDataBuffer = None):
        ts = time.time()
        data_list = []

        for i in range(size):
            # Random ticker
            _ticker = random.choice(cls.ticker)

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

    @classmethod
    def gen_data(cls, dtype: Literal['TickData', 'OrderData', 'TransactionData', 'BarData', 'Random'] = 'Random', buffer: MarketDataBuffer = None):
        match dtype:
            case 'TickData':
                generator = cls.generate_tick_data
            case 'OrderData':
                generator = cls.generate_order_data
            case 'TransactionData':
                generator = cls.generate_trade_data
            case 'BarData':
                generator = cls.generate_bar_data
            case 'Random':
                generator = random.choice([cls.generate_tick_data, cls.generate_order_data, cls.generate_trade_data, cls.generate_bar_data])
            case _:
                raise TypeError(f'Invalid data type {dtype}.')

        return generator(size=1, buffer=buffer)[0]


class MarketDataBufferTest(unittest.TestCase, MockData):
    def run_test(self):
        self.test_init()
        self.test_serialization()
        self.test_update_buffer()

    def test_init(self):
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            capacity=50
        )

        data_list = self.generate_trade_data(10) + self.generate_tick_data(10) + self.generate_bar_data(10)
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
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            capacity=10
        )

        data_list = self.generate_trade_data(1) + self.generate_tick_data(1) + self.generate_order_data(1) + self.generate_bar_data(1)

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
        shm = RawArray(ctypes.c_byte, 1024 * 1024)

        buf = MarketDataBuffer(
            buffer=memoryview(shm),
            capacity=50
        )
        data_list = []
        data_list += self.generate_trade_data(1, buffer=buf)
        data_list += self.generate_tick_data(1, buffer=buf)
        data_list += self.generate_order_data(1, buffer=buf)
        data_list += self.generate_bar_data(1, buffer=buf)

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


class MarketDataRingBufferTest(unittest.TestCase, MockData):
    def run_test(self):
        self.test_normal_operation()
        # self.test_wrapped_operation()
        # self.test_edge_cases()
        # self.test_mixed_data_types()
        # self.test_serialization()

    def test_normal_operation(self):
        """Test buffer operation without wrap-around"""
        # Create buffer sized to hold exactly 5 TickData entries without wrapping
        buffer_size = 3000  # roughly 1.5 tick data
        shm = RawArray(ctypes.c_byte, buffer_size)

        buf = MarketDataRingBuffer(buffer=memoryview(shm))
        st = time.time()
        dq = collections.deque(maxlen=2)
        for i in range(100000):
            md = MarketDataBufferTest.gen_data()
            dq.append(md)
            buf.put(md)

            if len(dq) == dq.maxlen:
                self.assertEqual(len(buf), dq.maxlen)

                _md0 = dq.popleft()
                _md1 = buf.listen()

                try:
                    self.assertEqual(_md0.to_bytes(), _md1.to_bytes())
                    print(f'data validated: {_md1}.')
                except AssertionError:
                    print(f'true data {_md0}')
                    print(f'got data {_md1}')
        print(f'ts cost per 100k: {time.time() - st}s')


class MarketDataConcurrentBufferTest(unittest.TestCase, MockData):
    def run_test(self):
        # self.test_single_process()
        self.test_concurrent_access()

    def test_single_process(self):
        """Test buffer operation without wrap-around"""
        # Create buffer sized to hold exactly 5 TickData entries without wrapping
        buffer_size = 3000  # roughly 1.5 tick data
        shm = RawArray(ctypes.c_byte, buffer_size)

        buf = MarketDataConcurrentBuffer(buffer=memoryview(shm), n_workers=1)
        st = time.time()
        dq = collections.deque(maxlen=2)
        for i in range(100000):
            md = MarketDataBufferTest.gen_data()
            dq.append(md)
            buf.put(md)

            if len(dq) == dq.maxlen:
                self.assertEqual(len(buf), dq.maxlen)

                _md0 = dq.popleft()
                _md1 = buf.listen(worker_id=0)

                try:
                    self.assertEqual(_md0.to_bytes(), _md1.to_bytes())
                    print(f'data validated: {_md1}.')
                except AssertionError:
                    print(f'true data {_md0}')
                    print(f'got data {_md1}')
        print(f'ts cost per 1m: {time.time() - st}s')

    def test_concurrent_access(self):
        """Test concurrent access with multiple workers"""
        # Create a shared memory buffer large enough for testing
        buffer_size = 5000  # 10MB
        shm = RawArray(ctypes.c_byte, buffer_size)

        # Create the concurrent buffer
        buf = MarketDataConcurrentBuffer(
            buffer=memoryview(shm),
            # capacity=1000,
            n_workers=8
        )

        # Generate test data (1000 items)
        test_data = []
        test_data += self.generate_trade_data(250)
        test_data += self.generate_tick_data(250)
        test_data += self.generate_order_data(250)
        test_data += self.generate_bar_data(250)
        random.shuffle(test_data)

        # Create and start worker processes
        workers = []
        for worker_id in range(8):
            p = multiprocessing.Process(target=self.worker_func, args=(buf, worker_id, test_data))
            workers.append(p)

        # Start the producer process
        producer = multiprocessing.Process(target=self.producer_func, args=(buf, test_data))
        producer.start()
        for worker in workers:
            worker.start()

        # Wait for producer to finish
        producer.join()
        for worker in workers:
            worker.join()

    def producer_func(self, buf: MarketDataConcurrentBuffer, test_data: list):
        """Function that puts data into the buffer"""
        start_time = time.time()
        put_attempts = 0

        for data in test_data:
            while True:
                if buf.is_full():
                    print(f'buffer full, retrying {put_attempts}...')
                    put_attempts += 1
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue

                buf.put(data)
                break

        print(f"Producer finished after {put_attempts} retries")

    def worker_func(self, buf: MarketDataConcurrentBuffer, worker_id: int, test_data: list):
        """Function that gets data from the buffer and validates it"""
        count = 0
        success = True

        while True:
            try:
                data = buf.listen(worker_id=worker_id, timeout=1)
            except TimeoutError:
                if count == len(test_data):
                    break

                continue

            self.assertEqual(data.to_bytes(), test_data[count].to_bytes())
            print(f'worker {worker_id} report data validated: {data}.')

            count += 1
            if count == len(test_data):
                break


def main():
    t0 = MarketDataBufferTest()
    t0.run_test()

    t1 = MarketDataRingBufferTest()
    t1.run_test()

    t2 = MarketDataConcurrentBufferTest()
    t2.run_test()


if __name__ == '__main__':
    main()
