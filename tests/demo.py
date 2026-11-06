import random
import random
import time
from typing import Literal

from algo_engine.base.candlestick import BarData
from algo_engine.base.market_data_buffer import MarketDataBuffer
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
