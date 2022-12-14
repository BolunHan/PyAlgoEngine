from __future__ import annotations

import abc
import datetime
import functools
import threading
import uuid
from collections import defaultdict

from PyQuantKit import TickData, TradeData, OrderBook, MarketData, Progress, TransactionSide, BarData, TransactionData

from . import LOGGER

__all__ = ['MDS', 'MarketDataService', 'MarketDataMonitor', 'Profile', 'ProgressiveReplay', 'SimpleReplay', 'Replay']
LOGGER = LOGGER.getChild('MarketEngine')


class MarketDataMonitor(object, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: str = None, mds: MarketDataService = None):
        self.name = name
        self.monitor_id = uuid.uuid4().hex if monitor_id is None else monitor_id
        self.mds = MDS if mds is None else mds

    @abc.abstractmethod
    def __call__(self, market_data: MarketData, **kwargs): ...

    @abc.abstractmethod
    @property
    def value(self): ...

    @abc.abstractmethod
    @property
    def is_ready(self) -> bool: ...


class SyntheticOrderBookMonitor(MarketDataMonitor):
    def __init__(self, keep_order_log: bool = False, **kwargs):
        self.keep_order_log = keep_order_log

        super().__init__(
            name=kwargs.pop('name', 'Monitor.SyntheticOrderBook'),
            monitor_id=kwargs.pop('monitor_id', None),
            mds=kwargs.pop('mds', None),
        )

        self._is_ready = True
        self._value = {}
        self.order_book = {}
        self.order_log = {}

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, TradeData):
            self.on_trade_data(trade_data=market_data)

    def on_trade_data(self, trade_data: TradeData):
        ticker = trade_data.ticker

        if order_book := self.order_book.get(ticker):
            if order_book.market_time <= trade_data.market_time:
                side: TransactionSide = trade_data.side
                price = trade_data.price
                book = order_book.ask if side.sign > 0 else order_book.bid
                listed_volume = book.at_price(price).volume if price in book else 0.
                traded_volume = trade_data.volume
                book.update_entry(price=price, volume=max(0, listed_volume - traded_volume))

        if self.keep_order_log:
            self._update_order_log(trade_data=trade_data)

    def on_transaction_data(self, transaction_data: TransactionData):
        pass

    def _update_order_log(self, trade_data: TradeData):
        side: TransactionSide = trade_data.side
        price = trade_data.price
        traded_volume = trade_data.volume

        for order_id in list(self.order_log):
            order_log = self.order_log.get(order_id)

            if order_log is None:
                continue

            if side.sign > 0:
                if order_log.side.sign < 0:
                    if price == order_log.price:
                        order_log.volume -= traded_volume
                    elif price > order_log.price:
                        order_log.volume = 0.
                else:
                    if price <= order_log.price:
                        order_log.volume = 0.
            elif side.sign < 0:
                if order_log.side.sign > 0:
                    if price == order_log.price:
                        order_log.volume -= traded_volume
                    elif price < order_log.price:
                        order_log.volume = 0.
                else:
                    if price >= order_log.price:
                        order_log.volume = 0.

            if order_log.volume <= 0:
                self.order_log.pop(order_id)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def value(self) -> dict[str, OrderBook]:
        return self.order_book


class MinuteBarMonitor(MarketDataMonitor):
    def __init__(self, interval: float = 60., **kwargs):
        self.interval = interval

        super().__init__(
            name=kwargs.pop('name', 'Monitor.SyntheticOrderBook'),
            monitor_id=kwargs.pop('monitor_id', None),
            mds=kwargs.pop('mds', None),
        )

        self._minute_bar_data: dict[str, BarData] = {}
        self._last_bar_data: dict[str, BarData] = {}

        self._is_ready = True
        self._value = {}

    def __call__(self, market_data: MarketData, **kwargs):
        self._update_last_bar(market_data=market_data, interval=self.interval)
        # self._update_active_bar(market_data=market_data, interval=self.interval)

    def _update_last_bar(self, market_data: MarketData, interval: float):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = market_data.market_time

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            # update bar_data
            if ticker in self._minute_bar_data:
                self._last_bar_data[ticker] = self._minute_bar_data[ticker]

            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                bar_start_time=datetime.datetime.fromtimestamp(self.mds.timestamp // interval * interval),
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
            bar_data.volume += market_data.volume
            bar_data.notional += market_data.notional
            bar_data.trade_count += 1

        bar_data.close_price = market_price
        bar_data.high_price = max(bar_data.high_price, market_price)
        bar_data.low_price = min(bar_data.low_price, market_price)

    def _update_active_bar(self, market_data: MarketData, interval: float):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = MarketData.market_time

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                bar_start_time=datetime.datetime.fromtimestamp(self.mds.timestamp - interval),
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
        bar_data.bar_start_time = datetime.datetime.fromtimestamp(self.mds.timestamp - interval)

        if isinstance(market_data, TradeData):
            history.append(market_data)

        while True:
            if history[0].market_time >= bar_data.bar_start_time:
                break
            else:
                history.pop(0)

        bar_data.volume = sum([_.volume for _ in history])
        bar_data.notional = sum([_.notional for _ in history])
        bar_data.trade_count = len([_.notional for _ in history])
        bar_data.close_price = market_price
        bar_data.open_price = history[0].market_price
        bar_data.high_price = max([_.market_price for _ in history])
        bar_data.low_price = min([_.market_price for _ in history])

    @property
    def is_ready(self) -> bool:
        return self._is_ready

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

        if market in self.trade_calendar:
            trade_calendar = self._trade_calendar[market]
        else:
            import trading_calendars
            trade_calendar = self._trade_calendar[market] = trading_calendars.get_calendar(market)

        calendar = trade_calendar.sessions_in_range(
            pd.Timestamp(start_date, tz=tz),
            pd.Timestamp(end_date, tz=tz)
        )

        # noinspection PyTypeChecker
        result = list(pd.to_datetime(calendar).date)

        return result

    @functools.lru_cache
    def is_trade_day(self, market_date: datetime.date, market='XSHG', tz='UTC') -> bool:
        import pandas as pd

        if market in self.trade_calendar:
            trade_calendar = self._trade_calendar[market]
        else:
            import trading_calendars
            trade_calendar = self._trade_calendar[market] = trading_calendars.get_calendar(market)

        return trade_calendar.is_session(pd.Timestamp(market_date, tz=tz))

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

        self.lock = threading.Lock()

        if self.synthetic_orderbook:
            # init synthetic orderbook monitor
            _ = SyntheticOrderBookMonitor(mds=self)
            self.add_monitor(monitor=_)
            # override current orderbook
            self._order_book = _.order_book

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

    def add_monitor(self, monitor: MarketDataMonitor):
        self._monitor[monitor.monitor_id] = monitor

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
        self._order_book[ticker] = tick_data.order_book

    def _on_order_book(self, order_book):
        ticker = order_book.ticker

        if ticker not in self._order_book:
            LOGGER.info(f'MDS confirmed {ticker} OrderBook subscribed!')

        self._order_book[ticker] = order_book

    def on_market_data(self, market_data: MarketData):
        self.lock.acquire()
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

        for monitor_id in self._monitor:
            monitor = self._monitor.get(monitor_id)

            if monitor is None:
                continue

            monitor.__call__(market_data)

        self.lock.release()

    def get_order_book(self, ticker: str) -> OrderBook:
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

            queued_volume = book.loc(prior=prior, posterior=posterior)
        return queued_volume

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        return self.profile.trade_time_between(start_time=start_time, end_time=end_time, **kwargs)

    def in_trade_session(self, market_time: datetime.datetime | float) -> bool:
        return self.profile.in_trade_session(market_time=market_time)

    def clear(self):
        self.lock.acquire()
        # self._market_price.clear()
        # self._market_time = None
        # self._timestamp = None

        self._market_history.clear()
        self._order_book.clear()
        self._monitor.clear()
        self.lock.release()

    @property
    def market_price(self) -> dict[str, float]:
        self.lock.acquire()
        result = self._market_price
        self.lock.release()
        return result

    @property
    def market_history(self) -> dict[str, dict[datetime.datetime, float]]:
        self.lock.acquire()
        result = self._market_history
        self.lock.release()
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
        self.start_date: datetime.date = kwargs.pop('start_date', None)
        self.end_date: datetime.date = kwargs.pop('end_date', None)
        self.calendar: list[datetime.date] = kwargs.pop('calendar', None)

        self.eod = kwargs.pop('eod', None)
        self.bod = kwargs.pop('bod', None)

        self.replay_subscription = {}
        self.replay_calendar = []
        self.replay_task = []

        self.date_progress = 0
        self.task_progress = 0
        self.progress = Progress(tasks=1, **kwargs)

        tickers = kwargs.pop('ticker', kwargs.pop('tickers', []))
        dtypes = kwargs.pop('dtype', kwargs.pop('dtypes', [TradeData, OrderBook, TickData]))

        if not isinstance(tickers, list):
            tickers = [tickers]

        if not isinstance(dtypes, list):
            dtypes = [dtypes]

        for ticker in tickers:
            for dtype in dtypes:
                self.add_subscription(ticker=ticker, dtype=dtype)

        subscription = kwargs.pop('subscription', kwargs.pop('subscribe', []))

        if not isinstance(subscription, list):
            subscription = [subscription]

        for sub in subscription:
            self.add_subscription(**sub)

        self.reset()

    def add_subscription(self, ticker: str, dtype: type | str):
        if isinstance(dtype, str):
            pass
        else:
            dtype = dtype.__name__

        topic = f'{ticker}.{dtype}'
        self.replay_subscription[topic] = (ticker, dtype)

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
        self.replay_task.clear()
        self.task_progress = 0

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
                self.eod(market_date=self.replay_calendar[self.date_progress - 1])

            if self.bod is not None and self.date_progress < len(self.replay_calendar):
                self.bod(market_date=self.replay_calendar[self.date_progress])

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
