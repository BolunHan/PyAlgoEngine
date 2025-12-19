import datetime
import time
import uuid

from algo_engine.base.c_market_data_ng.c_transaction import TransactionData, OrderData, TransactionSide, OrderType
from algo_engine.base.c_market_data_ng.c_tick import TickData, TickDataLite
from algo_engine.base.c_market_data_ng.c_candlestick import BarData, DailyBar

import random

ticker_pool = ['600010.SH', '000016.SZ', '000300.SH', '000905.SH', '399001.SZ', '399006.SZ', 'APPL', 'MSFT', 'GOOGL']
trade_side_pool = [TransactionSide.SIDE_LONG_OPEN, TransactionSide.SIDE_LONG_CLOSE, TransactionSide.SIDE_SHORT_OPEN, TransactionSide.SIDE_SHORT_CLOSE, TransactionSide.SIDE_NEUTRAL_OPEN, TransactionSide.SIDE_NEUTRAL_CLOSE]
order_side_pool = [TransactionSide.SIDE_BID, TransactionSide.SIDE_ASK]
order_type_pool = [OrderType.ORDER_LIMIT, OrderType.ORDER_MARKET, OrderType.ORDER_FAK, OrderType.ORDER_FOK, OrderType.ORDER_CANCEL]
bar_span_pool = [60, 300, 900, 3600]
date_span_pool = [1, 5, 21, 63, 252]


def random_transaction_data() -> TransactionData:
    ticker = random.choice(ticker_pool)
    side = random.choice(trade_side_pool)
    price = round(random.uniform(10, 500), 2)
    volume = random.randint(100, 10000)
    timestamp = time.time()

    td = TransactionData(
        ticker=ticker,
        timestamp=timestamp,
        price=price,
        volume=volume,
        side=side,
        buy_id=uuid.uuid4(),
        sell_id=uuid.uuid4(),
        transaction_id=uuid.uuid4()
    )
    return td


def random_order_data() -> OrderData:
    ticker = random.choice(ticker_pool)
    side = random.choice(order_side_pool)
    order_type = random.choice(order_type_pool)
    price = round(random.uniform(10, 500), 2)
    volume = random.randint(100, 10000)
    timestamp = time.time()

    od = OrderData(
        ticker=ticker,
        timestamp=timestamp,
        price=price,
        volume=volume,
        side=side,
        order_id=uuid.uuid4(),
        order_type=order_type
    )
    return od


def random_tick_data_lite() -> TickDataLite:
    ticker = random.choice(ticker_pool)
    timestamp = time.time()
    last_price = round(random.uniform(10, 500), 2)
    volume = random.randint(1000, 100000)
    bid_prices = [round(last_price - random.uniform(0.1, 1.0) * i, 2) for i in range(1, 6)]
    ask_prices = [round(last_price + random.uniform(0.1, 1.0) * i, 2) for i in range(1, 6)]
    bid_volumes = [random.randint(100, 10000) for _ in range(5)]
    ask_volumes = [random.randint(100, 10000) for _ in range(5)]

    tdl = TickDataLite(
        ticker=ticker,
        timestamp=timestamp,
        last_price=last_price,
        volume=volume,
        bid_price=bid_prices[0],
        ask_price=ask_prices[0],
        bid_volume=bid_volumes[0],
        ask_volume=ask_volumes[0]
    )
    return tdl


def random_tick_data() -> TickData:
    ticker = random.choice(ticker_pool)
    timestamp = time.time()
    last_price = round(random.uniform(10, 500), 2)
    volume = random.randint(1000, 100000)
    bid_prices = {f'bid_price_{i}': round(last_price - random.uniform(0.1, 1.0) * i, 2) for i in range(1, 11)}
    ask_prices = {f'ask_price_{i}': round(last_price + random.uniform(0.1, 1.0) * i, 2) for i in range(1, 11)}
    bid_volumes = {f'bid_volume_{i}': random.randint(100, 10000) for i in range(1, 11)}
    ask_volumes = {f'ask_volume_{i}': random.randint(100, 10000) for i in range(1, 11)}

    td = TickData(
        ticker=ticker,
        timestamp=timestamp,
        last_price=last_price,
        volume=volume,
        total_bid_volume=sum(bid_volumes.values()) + random.randint(100, 10000),
        total_ask_volume=sum(ask_volumes.values()) + random.randint(100, 10000),
        **bid_prices,
        **ask_prices,
        **bid_volumes,
        **ask_volumes
    )

    return td


def random_bar_data() -> BarData:
    ticker = random.choice(ticker_pool)
    timestamp = time.time()
    open_price = round(random.uniform(10, 500), 2)
    high_price = round(open_price + random.uniform(0.1, 10.0), 2)
    low_price = round(open_price - random.uniform(0.1, 10.0), 2)
    close_price = round(random.uniform(low_price, high_price), 2)
    volume = random.randint(1000, 100000)
    bar_span = random.choice(bar_span_pool)

    bd = BarData(
        ticker=ticker,
        timestamp=timestamp,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        bar_span=bar_span
    )
    return bd


def random_daily_bar() -> DailyBar:
    ticker = random.choice(ticker_pool)
    market_date = datetime.date.today() - datetime.timedelta(days=random.randint(0, 365))
    open_price = round(random.uniform(10, 500), 2)
    high_price = round(open_price + random.uniform(0.1, 10.0), 2)
    low_price = round(open_price - random.uniform(0.1, 10.0), 2)
    close_price = round(random.uniform(low_price, high_price), 2)
    volume = random.randint(1000, 100000)
    date_span = random.choice(date_span_pool)

    db = DailyBar(
        ticker=ticker,
        market_date=market_date,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        bar_span=date_span
    )
    return db


md_gen = [random_transaction_data, random_order_data, random_tick_data_lite, random_tick_data, random_bar_data, random_daily_bar]


def random_market_data():
    gen_func = random.choice(md_gen)
    return gen_func()
