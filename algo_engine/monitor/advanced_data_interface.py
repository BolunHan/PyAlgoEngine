import datetime
import json
import pickle
from ctypes import Structure, c_double
from multiprocessing import shared_memory
from typing import Self

from . import Monitor
from ..base import TradeData, MarketData, OrderData, TransactionData, TransactionSide, TickData, TickDataLite, BarData


class Entry(Structure):
    _fields_ = [
        ("price", c_double),  # Double precision floating point for price
        ("volume", c_double),  # Double precision floating point for volume
    ]

    def __repr__(self):
        return f"Entry(price={self.price}, volume={self.volume}, n_orders={self.n_orders})"


class PyOrderBook(object):
    def __init__(self, side: TransactionSide, data: dict[float, Entry] = None):
        self.side = TransactionSide(side)
        self.data: dict[float, Entry] = {}
        self.timestamp: float = 0

        if data:
            self.data.update(data)


class SyntheticOrderBookMonitor(Monitor):

    def __init__(self, **kwargs):

        super().__init__(
            name=kwargs.pop('name', 'Monitor.SyntheticOrderBook'),
            monitor_id=kwargs.pop('monitor_id', None)
        )

        self.bid: dict[str, PyOrderBook] = {}
        self.ask: dict[str, PyOrderBook] = {}
        self.tick_data: dict[str, TickData | TickDataLite] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, (TickData, TickDataLite)):
            self._on_tick_data(market_data)
        if isinstance(market_data, (TransactionData, TradeData)):
            self._on_trade_data(market_data)

    def _get_order_book(self, ticker: str) -> tuple[PyOrderBook, PyOrderBook]:
        if ticker in self.bid:
            bid = self.bid[ticker]
        else:
            bid = self.bid[ticker] = PyOrderBook(side=TransactionSide.SIDE_BID)

        if ticker in self.ask:
            ask = self.ask[ticker]
        else:
            ask = self.ask[ticker] = PyOrderBook(side=TransactionSide.SIDE_ASK)

        return bid, ask

    def _on_tick_data(self, tick_data: TickData):
        if isinstance(tick_data, TickDataLite):
            self.tick_data[tick_data.ticker] = tick_data
        else:
            self.tick_data[tick_data.ticker] = tick_data.lite

    def _on_trade_data(self, trade_data: TransactionData | TradeData):
        ticker = trade_data.ticker
        bid, ask = self._get_order_book(ticker)
        price = trade_data.price
        volume = trade_data.volume
        sign = trade_data.side.sign

        if sign == 1:
            entry = ask.data.get(price)

            if entry is None:
                return

            entry.volume -= volume
            if entry.volume <= 0:
                ask.data.pop(price, None)
        elif sign == -1:
            entry = bid.data.get(price)

            if entry is None:
                return

            entry.volume -= volume
            if entry.volume <= 0:
                bid.data.pop(price, None)

    def on_order_data(self, order_data: OrderData):
        ticker = order_data.ticker
        bid, ask = self._get_order_book(ticker)
        price = order_data.price
        volume = order_data.volume
        sign = order_data.side.sign

        if sign == 1:
            entry = ask.data.get(price)

            if entry is None:
                ask.data[price] = Entry(price=price, volume=volume)
            else:
                entry.volume += volume
        elif sign == -1:
            entry = bid.data.get(price)

            if entry is None:
                bid.data[price] = Entry(price=price, volume=volume)
            else:
                entry.volume += volume

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id,
            bid={ticker: {price: order_book.data[price].volume for price in sorted(order_book.data)} for ticker, order_book in self.bid.items()},
            ask={ticker: {price: order_book.data[price].volume for price in sorted(order_book.data)} for ticker, order_book in self.ask.items()},
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id'],
        )

        self.bid = {ticker: (book := PyOrderBook(side=TransactionSide.SIDE_BID, data={price: Entry(price=price, volume=volume) for price, volume in order_book.items()})) for ticker, order_book in json_dict['bid'].items()}
        self.bid = {ticker: (book := PyOrderBook(side=TransactionSide.SIDE_ASK, data={price: Entry(price=price, volume=volume) for price, volume in order_book.items()})) for ticker, order_book in json_dict['ask'].items()}
        return self

    def clear(self) -> None:
        self.bid.clear()
        self.ask.clear()
        self.tick_data.clear()

    @property
    def value(self) -> dict[str, TickData]:
        order_book = {}
        for ticker, tick_lite in self.tick_data.items():
            bid, ask = self._get_order_book(ticker)

            bid_price, bid_volume, ask_price, ask_volume = {}, {}, {}, {}

            for i, price in enumerate(sorted(bid.data, reverse=True), start=1):
                bid_price[f'bid_price_{i}'] = price
                bid_price[f'bid_volume_{i}'] = bid.data[price].volume

            for i, price in enumerate(sorted(ask.data), start=1):
                bid_price[f'ask_price_{i}'] = price
                bid_price[f'ask_volume_{i}'] = ask.data[price].volume

            data = TickData(
                ticker=tick_lite.ticker,
                timestamp=tick_lite.timestamp,
                last_price=tick_lite.last_price,
                total_traded_volume=tick_lite.total_traded_volume,
                total_traded_notional=tick_lite.total_traded_notional,
                total_trade_count=tick_lite.total_trade_count,
                **bid_price, **bid_volume, **ask_price, **ask_volume
            )

            order_book[ticker] = data

        return order_book


class MinuteBarMonitor(Monitor):

    def __init__(self, interval: float = 60., **kwargs):
        self.interval = interval

        super().__init__(
            name=kwargs.pop('name', 'Monitor.MinuteBarMonitor'),
            monitor_id=kwargs.pop('monitor_id', None)
        )

        self._minute_bar_data: dict[str, BarData] = {}
        self._last_bar_data: dict[str, BarData] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        self._update_last_bar(market_data=market_data, interval=self.interval)
        # self._update_active_bar(market_data=market_data, interval=self.interval)

    def _update_last_bar(self, market_data: MarketData, interval: float):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = market_data.market_time
        timestamp = market_data.timestamp

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            # update bar_data
            if ticker in self._minute_bar_data:
                self._last_bar_data[ticker] = self._minute_bar_data[ticker]

            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                timestamp=int(timestamp // interval + 1) * interval,
                start_timestamp=int(timestamp // interval) * interval,
                bar_span=datetime.timedelta(seconds=interval),
                high_price=market_price,
                low_price=market_price,
                open_price=market_price,
                close_price=market_price,
                volume=0.,
                notional=0.,
                trade_count=0
            )
        else:
            bar_data = self._minute_bar_data[ticker]

        if isinstance(market_data, TradeData):
            bar_data['volume'] += market_data.volume
            bar_data['notional'] += market_data.notional
            bar_data['trade_count'] += 1

        bar_data['close_price'] = market_price
        bar_data['high_price'] = max(bar_data.high_price, market_price)
        bar_data['low_price'] = min(bar_data.low_price, market_price)

    def _update_active_bar(self, market_data: MarketData, interval: float):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = market_data.market_time
        timestamp = market_data.timestamp

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                start_timestamp=timestamp - interval,
                timestamp=timestamp,
                bar_span=datetime.timedelta(seconds=interval),
                high_price=market_price,
                low_price=market_price,
                open_price=market_price,
                close_price=market_price,
                volume=0.,
                notional=0.,
                trade_count=0
            )
            bar_data.history = []

        else:
            bar_data = self._minute_bar_data[ticker]

        history: list[TradeData] = getattr(bar_data, 'history')
        bar_data['start_timestamp'] = timestamp - interval

        if isinstance(market_data, TradeData):
            history.append(market_data)

        while True:
            if history[0].market_time >= bar_data.bar_start_time:
                break
            else:
                history.pop(0)

        bar_data['volume'] = sum([_.volume for _ in history])
        bar_data['notional'] = sum([_.notional for _ in history])
        bar_data['trade_count'] = len([_.notional for _ in history])
        bar_data['close_price'] = market_price
        bar_data['open_price'] = history[0].market_price
        bar_data['high_price'] = max([_.market_price for _ in history])
        bar_data['low_price'] = min([_.market_price for _ in history])

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id,
            interval=self.interval,
            minute_bar_data={k: v.to_bytes() for k, v in self._minute_bar_data.items()},
            last_bar_data={k: v.to_bytes() for k, v in self._last_bar_data.items()},
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id'],
            interval=json_dict['interval'],
        )

        self._minute_bar_data = {k: BarData.from_bytes(v) for k, v in json_dict['minute_bar_data'].items()}
        self._last_bar_data = {k: BarData.from_bytes(v) for k, v in json_dict['last_bar_data'].items()}
        return self

    def from_shm(self, name: str = None) -> None:
        if name is None:
            name = f'{self.monitor_id}.json'

        shm = shared_memory.SharedMemory(name=name)
        json_dict = pickle.loads(bytes(shm.buf))

        self.clear()

        self._minute_bar_data.update({k: BarData.from_bytes(v) for k, v in json_dict['minute_bar_data'].items()})
        self._last_bar_data.update({k: BarData.from_bytes(v) for k, v in json_dict['last_bar_data'].items()})

    def clear(self) -> None:
        self._minute_bar_data.clear()
        self._last_bar_data.clear()

    @property
    def value(self) -> dict[str, BarData]:
        return self._last_bar_data
