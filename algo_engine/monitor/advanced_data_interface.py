import datetime
import json
import pickle
from multiprocessing import shared_memory
from typing import Self

from . import Monitor
from ..base import TradeData, OrderBook, MarketData, BarData, TransactionData


class SyntheticOrderBookMonitor(Monitor):

    def __init__(self, **kwargs):

        super().__init__(
            name=kwargs.pop('name', 'Monitor.SyntheticOrderBook'),
            monitor_id=kwargs.pop('monitor_id', None)
        )

        self.order_book: dict[str, OrderBook] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, TradeData):
            self.on_trade_data(trade_data=market_data)

    def on_trade_data(self, trade_data: TradeData):
        ticker = trade_data.ticker

        if order_book := self.order_book.get(ticker):
            if order_book.market_time <= trade_data.market_time:
                side = trade_data.side
                price = trade_data.price
                book = order_book.ask if side.sign > 0 else order_book.bid
                listed_volume = book.at_price(price).volume if price in book else 0.
                traded_volume = trade_data.volume
                book.update(price=price, volume=max(0, listed_volume - traded_volume))

    def on_transaction_data(self, transaction_data: TransactionData):
        pass

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id,
            order_book={k: v.to_json(fmt='dict') for k, v in self.order_book.items()},
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
            keep_order_log=json_dict['keep_order_log']
        )

        self.order_book = {k: MarketData.from_json(v) for k, v in json_dict['order_book'].items()}
        return self

    def from_shm(self, name: str = None) -> None:
        if name is None:
            name = f'{self.monitor_id}.json'

        shm = shared_memory.SharedMemory(name=name)
        json_dict = pickle.loads(bytes(shm.buf))

        self.clear()

        self.order_book.update({k: MarketData.from_json(v) for k, v in json_dict['order_book'].items()})

    def clear(self) -> None:
        self.order_book.clear()

    @property
    def value(self) -> dict[str, OrderBook]:
        return self.order_book


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
            minute_bar_data={k: v.to_json(fmt='dict') for k, v in self._minute_bar_data.items()},
            last_bar_data={k: v.to_json(fmt='dict') for k, v in self._last_bar_data.items()},
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

        self._minute_bar_data = {k: MarketData.from_json(v) for k, v in json_dict['minute_bar_data'].items()}
        self._last_bar_data = {k: MarketData.from_json(v) for k, v in json_dict['last_bar_data'].items()}
        return self

    def from_shm(self, name: str = None) -> None:
        if name is None:
            name = f'{self.monitor_id}.json'

        shm = shared_memory.SharedMemory(name=name)
        json_dict = pickle.loads(bytes(shm.buf))

        self.clear()

        self._minute_bar_data.update({k: MarketData.from_json(v) for k, v in json_dict['minute_bar_data'].items()})
        self._last_bar_data.update({k: MarketData.from_json(v) for k, v in json_dict['last_bar_data'].items()})

    def clear(self) -> None:
        self._minute_bar_data.clear()
        self._last_bar_data.clear()

    @property
    def value(self) -> dict[str, BarData]:
        return self._last_bar_data
