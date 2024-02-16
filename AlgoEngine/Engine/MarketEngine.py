import abc
import datetime
import functools
import inspect
import json
import pickle
import uuid
from collections import defaultdict
from multiprocessing import shared_memory
from typing import Iterable, Self

from PyQuantKit import TickData, TradeData, OrderBook, MarketData, Progress, TransactionSide, BarData, TransactionData

from . import LOGGER

__all__ = ['MDS', 'MarketDataService', 'MarketDataMonitor', 'MonitorManager', 'SyntheticOrderBookMonitor', 'MinuteBarMonitor', 'Profile', 'ProgressiveReplay', 'SimpleReplay', 'Replay']
LOGGER = LOGGER.getChild('MarketEngine')


class MarketDataMonitor(object, metaclass=abc.ABCMeta):
    """
    this is a template for market data monitor

    A data monitor is a module that process market data and generate custom index

    When MDS receive an update of market data, the __call__ function of this monitor is triggered.

    Note: all the market_data, of all subscribed ticker will be fed into monitor. It should be assumed that a storage for multiple ticker is required.
    To access the monitor, use `monitor = MDS[monitor_id]`
    To access the index generated by the monitor, use `monitor.value`
    To indicate that the monitor is ready to use set `monitor.is_ready = True`

    The implemented monitor should be initialized and use `MDS.add_monitor(monitor)` to attach onto the engine
    """

    def __init__(self, name: str, monitor_id: str = None):
        self.name: str = name
        self.monitor_id: str = uuid.uuid4().hex if monitor_id is None else monitor_id
        self.enabled: bool = True

    @abc.abstractmethod
    def __call__(self, market_data: MarketData, **kwargs):
        ...

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    @abc.abstractmethod
    def to_json(self, fmt='str') -> dict | str:
        ...

    @classmethod
    @abc.abstractmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        ...

    def to_shm(self, name: str = None) -> str:
        """
        Put the data of the monitor into python shared memory.
        This function is designed to facilitate multiprocessing.
        Some monitor is not advised to be handled concurrently,
        In which case, raise a NotImplementedError.

        The function is expected to put all data into a sharable list,
        and return the name of the list, which can be set by the given name.
        Default name = self.monitor_id

        Note that this method HAVE NO LOCK, use with caution.
        """
        if name is None:
            name = f'{self.monitor_id}.json'

        data = pickle.dumps(self.to_json(fmt='dict'))
        size = len(data)

        try:
            shm = shared_memory.SharedMemory(name=name)

            if shm.size != size:
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileNotFoundError as _:
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)

        shm.buf[:size] = data
        shm.close()
        return name

    def from_shm(self, name: str = None) -> None:
        """
        retrieve the data and update the monitor from shared memory.
        This function is designed to facilitate multiprocessing.
        """
        return

    @abc.abstractmethod
    def clear(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def value(self) -> dict[str, float] | float:
        ...

    @property
    def is_ready(self) -> bool:
        return True


class MonitorManager(object):
    """
    manage market data monitor

    state codes for the manager
    0: idle
    1: working
    -1: terminating
    """

    def __init__(self):
        self.monitor: dict[str, MarketDataMonitor] = {}

    def __call__(self, market_data: MarketData):
        for monitor_id in self.monitor:
            self._work(monitor_id=monitor_id, market_data=market_data)

    def add_monitor(self, monitor: MarketDataMonitor):
        self.monitor[monitor.monitor_id] = monitor

    def pop_monitor(self, monitor_id: str):
        self.monitor.pop(monitor_id)

    def _work(self, monitor_id: str, market_data: MarketData):
        monitor = self.monitor.get(monitor_id)
        if monitor is not None and monitor.enabled:
            monitor.__call__(market_data)

    def clear(self):
        self.monitor.clear()


class SyntheticOrderBookMonitor(MarketDataMonitor):

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


class MinuteBarMonitor(MarketDataMonitor):

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


class Profile(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            session_start: datetime.time | None = None,
            session_end: datetime.time | None = None,
            session_break: tuple[datetime.time, datetime.time] | None = None
    ):
        self.session_start: datetime.time | None = session_start
        self.session_end: datetime.time | None = session_end
        self.session_break: tuple[datetime.time, datetime.time] | None = session_break

    @abc.abstractmethod
    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        ...

    @abc.abstractmethod
    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        ...


class DefaultProfile(Profile):
    def __init__(self):
        super().__init__(
            session_start=datetime.time(0),
            session_end=None,
            session_break=None
        )

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        if start_time is not None and isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time)

        if end_time is not None and isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time)

        if start_time is None or end_time is None:
            return datetime.timedelta(seconds=0)

        if start_time > end_time:
            return datetime.timedelta(seconds=0)

        return end_time - start_time

    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        return True


class CN_Profile(Profile):
    def __init__(self):
        super().__init__(
            session_start=datetime.time(9, 30),
            session_end=datetime.time(15, 0),
            session_break=(datetime.time(11, 30), datetime.time(13, 0))
        )

        self._trade_calendar = {}

    @functools.lru_cache
    def trade_calendar(self, start_date: datetime.date, end_date: datetime.date, market='XSHG', tz='UTC') -> list[datetime.date]:
        import pandas as pd

        if market in self._trade_calendar:
            trade_calendar = self._trade_calendar[market]
        else:
            import exchange_calendars
            trade_calendar = self._trade_calendar[market] = exchange_calendars.get_calendar(market)

        calendar = trade_calendar.sessions_in_range(
            pd.Timestamp(start_date, tz=tz),
            pd.Timestamp(end_date, tz=tz)
        )

        # noinspection PyTypeChecker
        result = list(pd.to_datetime(calendar).date)

        return result

    @functools.lru_cache
    def is_trade_day(self, market_date: datetime.date, market='XSHG', tz='UTC') -> bool:
        if market in self._trade_calendar:
            trade_calendar = self._trade_calendar[market]
        else:
            import exchange_calendars
            trade_calendar = self._trade_calendar[market] = exchange_calendars.get_calendar(market)

        return trade_calendar.is_session(market_date)

    def trade_days_between(self, start_date: datetime.date, end_date: datetime.date = datetime.date.today(), **kwargs) -> int:
        """
        Returns the number of trade days between the given date, which is the pre-open of the start_date to the pre-open of the end_date.
        :param start_date: the given trade date
        :param end_date: the given trade date
        :return: integer number of days
        """
        assert start_date <= end_date, "The end date must not before the start date"

        if start_date == end_date:
            offset = 0
        else:
            market_date_list = self.trade_calendar(start_date=start_date, end_date=end_date, **kwargs)
            if not market_date_list:
                offset = 0
            else:
                last_trade_date = market_date_list[-1]
                offset = len(market_date_list)

                if last_trade_date == end_date:
                    offset -= 1

        return offset

    @classmethod
    def time_to_seconds(cls, t: datetime.time):
        return (t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1000

    def trade_time_between(self, start_time: datetime.datetime | datetime.time | float | int, end_time: datetime.datetime | datetime.time | float | int, fmt='timedelta', **kwargs):
        if start_time is None or end_time is None:
            if fmt == 'timestamp':
                return 0.
            elif fmt == 'timedelta':
                return datetime.timedelta(0)
            else:
                raise NotImplementedError(f'Invalid fmt {fmt}, should be "timestamp" or "timedelta"')

        session_start = kwargs.pop('session_start', self.session_start)
        session_break = kwargs.pop('session_break', self.session_break)
        session_end = kwargs.pop('session_end', self.session_end)
        session_length_0 = datetime.timedelta(seconds=self.time_to_seconds(session_break[0]) - self.time_to_seconds(session_start))
        session_length_1 = datetime.timedelta(seconds=self.time_to_seconds(session_end) - self.time_to_seconds(session_break[1]))
        session_length = session_length_0 + session_length_1
        implied_date = datetime.date.today()

        if isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time)
            implied_date = start_time.date()

        if isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time)
            implied_date = end_time.date()

        if isinstance(start_time, datetime.time):
            start_time = datetime.datetime.combine(implied_date, start_time)

        if isinstance(end_time, datetime.time):
            end_time = datetime.datetime.combine(implied_date, end_time)

        offset = datetime.timedelta()

        market_time = start_time.time()

        # calculate the timespan from start_time to session_end
        if market_time <= session_start:
            offset += session_length
        elif session_start < market_time <= session_break[0]:
            offset += datetime.datetime.combine(start_time.date(), session_break[0]) - start_time
            offset += session_length_1
        elif session_break[0] < market_time <= session_break[1]:
            offset += session_length_1
        elif session_break[1] < market_time <= session_end:
            offset += datetime.datetime.combine(start_time.date(), session_end) - start_time
        else:
            offset += datetime.timedelta(0)

        offset -= session_length

        market_time = end_time.time()

        # calculate the timespan from session_start to end_time
        if market_time <= session_start:
            offset += datetime.timedelta(0)
        elif session_start < market_time <= session_break[0]:
            offset += end_time - datetime.datetime.combine(end_time.date(), session_start)
        elif session_break[0] < market_time <= session_break[1]:
            offset += session_length_0
        elif session_break[1] < market_time <= session_end:
            offset += end_time - datetime.datetime.combine(end_time.date(), session_break[1])
            offset += session_length_0
        else:
            offset += session_length

        # calculate market_date difference
        if start_time.date() != end_time.date():
            offset += session_length * self.trade_days_between(start_date=start_time.date(), end_date=end_time.date(), **kwargs)

        if fmt == 'timestamp':
            return offset.total_seconds()
        elif fmt == 'timedelta':
            return offset
        else:
            raise NotImplementedError(f'Invalid fmt {fmt}, should be "timestamp" or "timedelta"')

    def in_trade_session(self, market_time: datetime.datetime | float | int = None) -> bool:
        if market_time is None:
            market_time = datetime.datetime.now()

        if isinstance(market_time, (float, int)):
            market_time = datetime.datetime.fromtimestamp(market_time)

        market_date = market_time.date()
        market_time = market_time.time()

        if not self.is_trade_day(market_date=market_date):
            return False

        if market_time < datetime.time(9, 30):
            return False

        if datetime.time(11, 30) < market_time < datetime.time(13, 00):
            return False

        if market_time > datetime.time(15, 00):
            return False

        return True


class MarketDataService(object):
    def __init__(self, profile: Profile = None, **kwargs):
        self.profile = DefaultProfile() if profile is None else profile
        self.synthetic_orderbook = kwargs.pop('synthetic_orderbook', False)
        self.cache_history = kwargs.pop('cache_history', False)

        self._market_price = {}
        self._market_history = defaultdict(dict)
        self._market_time: datetime.datetime | None = None
        self._timestamp: float | None = None

        self._order_book: dict[str, OrderBook] = {}
        self._tick_data: dict[str, TickData] = {}
        self._trade_data: dict[str, TradeData] = {}
        self._monitor: dict[str, MarketDataMonitor] = {}
        self._monitor_manager = MonitorManager()

        if self.synthetic_orderbook:
            # init synthetic orderbook monitor
            _ = SyntheticOrderBookMonitor(mds=self)
            self.add_monitor(monitor=_)
            # override current orderbook
            self._order_book = _.order_book

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

    def __getitem__(self, monitor_id: str) -> MarketDataMonitor:
        return self.monitor[monitor_id]

    def add_monitor(self, monitor: MarketDataMonitor):
        self.monitor[monitor.monitor_id] = monitor
        self.monitor_manager.add_monitor(monitor)

    def pop_monitor(self, monitor: MarketDataMonitor = None, monitor_id: str = None, monitor_name: str = None):
        if monitor_id is not None:
            pass
        elif monitor_name is not None:
            for _ in list(self.monitor.values()):
                if _.name == monitor_name:
                    monitor_id = _.monitor_id
            if monitor is None:
                LOGGER.error(f'monitor_name {monitor_name} not registered.')
        elif monitor is not None:
            monitor_id = monitor.monitor_id
        else:
            LOGGER.error('must assign a monitor, or monitor_id, or monitor_name to pop.')
            return None

        self.monitor.pop(monitor_id)
        self.monitor_manager.pop_monitor(monitor_id)

    def init_cn_override(self):
        self.profile = CN_Profile()

    def _on_trade_data(self, trade_data: TradeData):
        ticker = trade_data.ticker

        if ticker not in self._trade_data:
            LOGGER.info(f'MDS confirmed {ticker} TradeData subscribed!')

        self._trade_data[ticker] = trade_data

    def _on_tick_data(self, tick_data: TickData):
        ticker = tick_data.ticker

        if ticker not in self._tick_data:
            LOGGER.info(f'MDS confirmed {ticker} TickData subscribed!')

        self._tick_data[ticker] = tick_data
        # self._order_book[ticker] = tick_data.order_book

    def _on_order_book(self, order_book):
        ticker = order_book.ticker

        if ticker not in self._order_book:
            LOGGER.info(f'MDS confirmed {ticker} OrderBook subscribed!')

        self._order_book[ticker] = order_book

    def on_market_data(self, market_data: MarketData):
        ticker = market_data.ticker
        market_time = market_data.market_time
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        self._market_price[ticker] = market_price
        self._market_time = market_time
        self._timestamp = timestamp

        if self.cache_history:
            self._market_history[ticker][market_time] = market_price

        if isinstance(market_data, TradeData):
            self._on_trade_data(trade_data=market_data)
        elif isinstance(market_data, TickData):
            self._on_tick_data(tick_data=market_data)
        elif isinstance(market_data, OrderBook):
            self._on_order_book(order_book=market_data)

        self.monitor_manager.__call__(market_data=market_data)

    def get_order_book(self, ticker: str) -> OrderBook | None:
        return self._order_book.get(ticker, None)

    def get_queued_volume(self, ticker: str, side: TransactionSide | str | int, prior: float, posterior: float = None) -> float:
        """
        get queued volume prior / posterior to given price, NOT COUNTING GIVEN PRICE!
        :param ticker: the given ticker
        :param side: the given trade side
        :param prior: the given price
        :param posterior: optional the given posterior price
        :return: the summed queued volume, in float.
        """
        order_book = self.get_order_book(ticker=ticker)

        if order_book is None:
            queued_volume = float('nan')
        else:
            trade_side = TransactionSide(side)

            if trade_side.sign > 0:
                book = order_book.bid
            elif trade_side < 0:
                book = order_book.ask
            else:
                raise ValueError(f'Invalid side {side}')

            queued_volume = book.loc_volume(p0=prior, p1=posterior)
        return queued_volume

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        return self.profile.trade_time_between(start_time=start_time, end_time=end_time, **kwargs)

    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        return self.profile.in_trade_session(market_time=market_time)

    def clear(self):
        # self._market_price.clear()
        # self._market_time = None
        # self._timestamp = None

        self._market_history.clear()
        self._order_book.clear()
        self.monitor.clear()
        self.monitor_manager.clear()

    @property
    def market_price(self) -> dict[str, float]:
        result = self._market_price
        return result

    @property
    def market_history(self) -> dict[str, dict[datetime.datetime, float]]:
        result = self._market_history
        return result

    @property
    def market_time(self) -> datetime.datetime | None:
        if self._market_time is None:
            if self._timestamp is None:
                return None
            else:
                return datetime.datetime.fromtimestamp(self._timestamp)
        else:
            return self._market_time

    @property
    def market_date(self) -> datetime.date | None:
        if self.market_time is None:
            return None

        return self._market_time.date()

    @property
    def timestamp(self) -> float | None:
        if self._timestamp is None:
            if self._market_time is None:
                return None
            else:
                return self._market_time.timestamp()
        else:
            return self._timestamp

    @property
    def session_start(self) -> datetime.time | None:
        return self.profile.session_start

    @property
    def session_end(self) -> datetime.time | None:
        return self.profile.session_end

    @property
    def session_break(self) -> tuple[datetime.time, datetime.time] | None:
        return self.profile.session_break

    @property
    def monitor(self) -> dict[str, MarketDataMonitor]:
        return self._monitor

    @property
    def monitor_manager(self) -> MonitorManager:
        return self._monitor_manager

    @monitor_manager.setter
    def monitor_manager(self, manager: MonitorManager):
        self._monitor_manager.clear()

        self._monitor_manager = manager

        for monitor in self.monitor.values():
            self._monitor_manager.add_monitor(monitor=monitor)


class Replay(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __next__(self): ...

    @abc.abstractmethod
    def __iter__(self): ...


class SimpleReplay(Replay):
    def __init__(self, **kwargs):
        self.eod = kwargs.pop('eod', None)
        self.bod = kwargs.pop('bod', None)

        self.replay_task = []
        self.task_progress = 0
        self.task_date = None
        self.progress = Progress(tasks=1, **kwargs)

    def load(self, data):
        if isinstance(data, dict):
            self.replay_task.extend(list(data.values()))
        else:
            self.replay_task.extend(data)

    def reset(self):
        self.replay_task.clear()
        self.task_progress = 0
        self.task_date = None
        self.progress.reset()

    def next_task(self):
        if self.task_progress < len(self.replay_task):
            market_data = self.replay_task[self.task_progress]
            market_time = market_data.market_time

            if isinstance(market_time, datetime.datetime):
                market_date = market_time.date()
            else:
                market_date = market_time

            if market_date != self.task_date:
                if callable(self.eod) and self.task_date:
                    self.eod(self.task_date)

                self.task_date = market_date
                self.progress.prompt = f'Replay {market_date:%Y-%m-%d}:'

                if callable(self.bod):
                    self.bod(market_date)

            self.progress.done_tasks = self.task_progress / len(self.replay_task)

            if (not self.progress.tick_size) or self.progress.progress >= self.progress.tick_size + self.progress.last_output:
                self.progress.output()

            self.task_progress += 1
        else:
            raise StopIteration()

        return market_data

    def __next__(self):
        try:
            return self.next_task()
        except StopIteration:
            if not self.progress.is_done:
                self.progress.done_tasks = 1
                self.progress.output()

            self.reset()
            raise StopIteration()

    def __iter__(self):
        return self


class ProgressiveReplay(Replay):
    """
    progressively loading and replaying market data

    requires arguments
    loader: a data loading function. Expect loader = Callable(market_date: datetime.date, ticker: str, dtype: str| type) -> dict[any, MarketData]
    start_date & end_date: the given replay period
    or calendar: the given replay calendar.

    accepts kwargs:
    ticker / tickers: the given symbols to replay, expect a str| list[str]
    dtype / dtypes: the given dtype(s) of symbol to replay, expect a str | type, list[str | type]. default = all, which is (TradeData, TickData, OrderBook)
    subscription / subscribe: the given ticker-dtype pair to replay, expect a list[dict[str, str | type]]
    """

    def __init__(
            self,
            loader,
            **kwargs
    ):
        self.loader = loader
        self.start_date: datetime.date | None = kwargs.pop('start_date', None)
        self.end_date: datetime.date | None = kwargs.pop('end_date', None)
        self.calendar: list[datetime.date] | None = kwargs.pop('calendar', None)

        self.eod = kwargs.pop('eod', None)
        self.bod = kwargs.pop('bod', None)

        self.replay_subscription = {}
        self.replay_calendar = []
        self.replay_task = []

        self.date_progress = 0
        self.task_progress = 0
        self.progress = Progress(tasks=1, **kwargs)

        tickers: list[str] = kwargs.pop('ticker', kwargs.pop('tickers', []))
        dtypes: list[str | type] = kwargs.pop('dtype', kwargs.pop('dtypes', [TradeData, OrderBook, TickData]))

        if not all([arg_name in inspect.getfullargspec(loader).args for arg_name in ['market_date', 'ticker', 'dtype']]):
            raise TypeError('loader function has 3 requires args, market_date, ticker and dtype.')

        if isinstance(tickers, str):
            tickers = [tickers]
        elif isinstance(tickers, Iterable):
            tickers = list(tickers)
        else:
            raise TypeError(f'Invalid ticker {tickers}, expect str or list[str]')

        if isinstance(dtypes, str) or inspect.isclass(dtypes):
            dtypes = [dtypes]
        elif isinstance(dtypes, Iterable):
            dtypes = list(dtypes)
        else:
            raise TypeError(f'Invalid dtype {dtypes}, expect str or list[str]')

        for ticker in tickers:
            for dtype in dtypes:
                self.add_subscription(ticker=ticker, dtype=dtype)

        subscription = kwargs.pop('subscription', kwargs.pop('subscribe', []))

        if isinstance(subscription, dict):
            subscription = [subscription]

        for sub in subscription:
            self.add_subscription(**sub)

        self.reset()

    def add_subscription(self, ticker: str, dtype: type | str):
        if isinstance(dtype, str):
            pass
        elif inspect.isclass(dtype):
            dtype = dtype.__name__
        else:
            raise ValueError(f'Invalid dtype {dtype}, expect str or class.')

        topic = f'{ticker}.{dtype}'
        self.replay_subscription[topic] = (ticker, dtype)

    def remove_subscription(self, ticker: str, dtype: type | str):
        if isinstance(dtype, str):
            pass
        else:
            dtype = dtype.__name__

        topic = f'{ticker}.{dtype}'
        self.replay_subscription.pop(topic, None)

    def reset(self):
        if self.calendar is None:
            md = self.start_date
            self.replay_calendar.clear()

            while md <= self.end_date:
                self.replay_calendar.append(md)
                md += datetime.timedelta(days=1)

        elif callable(self.calendar):
            self.replay_calendar = self.calendar(start_date=self.start_date, end_date=self.end_date)
        else:
            self.replay_calendar = self.calendar

        self.date_progress = 0
        self.progress.reset()

    def next_trade_day(self):
        if self.date_progress < len(self.replay_calendar):
            market_date = self.replay_calendar[self.date_progress]
            self.progress.prompt = f'Replay {market_date:%Y-%m-%d} ({self.date_progress + 1} / {len(self.replay_calendar)}):'
            for topic in self.replay_subscription:
                ticker, dtype = self.replay_subscription[topic]
                LOGGER.info(f'{self} loading {market_date} {ticker} {dtype}')
                data = self.loader(market_date=market_date, ticker=ticker, dtype=dtype)

                if isinstance(data, dict):
                    self.replay_task.extend(list(data.values()))
                elif isinstance(data, (list, tuple)):
                    self.replay_task.extend(data)

            self.date_progress += 1
        else:
            raise StopIteration()

        self.replay_task.sort(key=lambda x: x.market_time)

    def next_task(self):
        if self.task_progress < len(self.replay_task):
            data = self.replay_task[self.task_progress]
            self.task_progress += 1
        else:
            if self.eod is not None and self.date_progress:
                self.eod(market_date=self.replay_calendar[self.date_progress - 1], replay=self)

            self.replay_task.clear()
            self.task_progress = 0

            if self.bod is not None and self.date_progress < len(self.replay_calendar):
                self.bod(market_date=self.replay_calendar[self.date_progress], replay=self)

            self.next_trade_day()

            data = self.next_task()

        if self.replay_task and self.replay_calendar:
            current_progress = (self.date_progress - 1 + (self.task_progress / len(self.replay_task))) / len(self.replay_calendar)
            self.progress.done_tasks = current_progress
        else:
            self.progress.done_tasks = 1

        if (not self.progress.tick_size) \
                or self.progress.progress >= self.progress.tick_size + self.progress.last_output \
                or self.progress.is_done:
            self.progress.output()

        return data

    def __next__(self):
        try:
            return self.next_task()
        except StopIteration:
            if not self.progress.is_done:
                self.progress.done_tasks = 1
                self.progress.output()

            self.reset()
            raise StopIteration()

    def __iter__(self):
        self.reset()
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}{{id={id(self)}, from={self.start_date}, to={self.end_date}}}'


MDS = MarketDataService()
