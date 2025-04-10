import abc
import datetime
import inspect
import operator
from collections.abc import Mapping, Sequence, Iterator
from typing import Iterable, Protocol

from . import LOGGER
from ..base import Progress, TickData, TransactionData, TradeData, OrderData, MarketData, MarketDataBuffer

LOGGER = LOGGER.getChild('Replay')


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


class DataLoader(Protocol):
    def __call__(self, market_date: datetime.date, ticker: str, dtype: str) -> Mapping[float, MarketData] | Sequence[MarketData] | MarketDataBuffer:
        ...


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
            loader: DataLoader,
            **kwargs
    ):
        self.loader = loader
        self.market_date: datetime.date | None = kwargs.pop('market_date', None)
        self.start_date: datetime.date | None = kwargs.pop('start_date', None)
        self.end_date: datetime.date | None = kwargs.pop('end_date', None)
        self.calendar: list[datetime.date] | None = kwargs.pop('calendar', None)

        self.eod = kwargs.pop('eod', None)
        self.bod = kwargs.pop('bod', None)

        self.replay_subscription = {}
        self.replay_calendar = []
        self.replay_task: Iterator | None = None
        self.replay_task_length: int = 0
        self.replay_status = {}

        self.date_progress = 0
        self.task_progress = 0
        self.progress = Progress(tasks=1, **kwargs)

        tickers: list[str] = kwargs.pop('ticker', kwargs.pop('tickers', []))
        dtypes: list[str | type] = kwargs.pop('dtype', kwargs.pop('dtypes', [TradeData, TransactionData, OrderData, TickData]))

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
            self.replay_calendar = [self.start_date + datetime.timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        else:
            self.replay_calendar = self.calendar

        if self.market_date is None:
            self.market_date = self.replay_calendar[0] if self.replay_calendar else self.start_date
        else:
            date_to_replay = [_ for _ in self.replay_calendar if _ >= self.market_date]
            self.market_date = date_to_replay[0] if date_to_replay else self.end_date

        self.replay_status = {market_date: 'skipped' if market_date < self.market_date else 'idle' for market_date in self.replay_calendar}

        self.task_progress = 0
        self.replay_task_length = 0
        self.replay_task = None
        self.date_progress = sum([1 for _ in self.replay_calendar if _ < self.market_date])
        self.progress.reset()

        if self.date_progress:
            self.progress.done_tasks = self.date_progress / len(self.replay_calendar)

    def next_trade_day(self):
        if self.date_progress >= len(self.replay_calendar):
            raise StopIteration()

        self.market_date = market_date = self.replay_calendar[self.date_progress]
        self.replay_status[market_date] = 'started'
        self.progress.prompt = f'Replay {market_date:%Y-%m-%d} ({self.date_progress + 1} / {len(self.replay_calendar)}):'

        for topic in self.replay_subscription:
            ticker, dtype = self.replay_subscription[topic]
            LOGGER.info(f'{self} loading {market_date} {ticker} {dtype}...')
            data = self.loader(market_date=market_date, ticker=ticker, dtype=dtype)
            if isinstance(data, Mapping):
                data = [data[ts] for ts in sorted(data)]  # expect to be a mapping of ts and data
                self.replay_task = iter(data)
                self.replay_task_length = len(data)
            elif isinstance(data, Sequence):
                data = sorted(data, key=operator.attrgetter('timestamp', 'ticker', '__class__.__name__'))
                self.replay_task = iter(data)
                self.replay_task_length = len(data)
            elif isinstance(data, MarketDataBuffer):
                data.sort()
                self.replay_task = iter(data)
                self.replay_task_length = len(data)
            else:
                raise TypeError(f'Invalid return type of dataloader, expect list, tuple, dict or MarketDataBuffer, got {type(data)}.')

        LOGGER.info(f'{market_date} data loaded! {self.replay_task_length:,} entries.')
        self.date_progress += 1

    def next_task(self):
        try:
            data = next(self.replay_task)
            self.task_progress += 1
        except StopIteration:
            if self.eod is not None and self.replay_status[self.market_date] == 'started':
                self.eod(market_date=self.market_date, replay=self)
                self.replay_status[self.market_date] = 'done'

            self.replay_task = None
            self.task_progress = 0

            if self.bod is not None and self.date_progress < len(self.replay_calendar):
                self.bod(market_date=self.replay_calendar[self.date_progress], replay=self)

            # this is by designed, to load the new data after the bod is done.
            self.next_trade_day()

            #  the bod process should be moved here!

            data = self.next_task()

        if self.replay_task_length and self.replay_calendar:
            current_progress = (self.date_progress - 1 + (self.task_progress / self.replay_task_length)) / len(self.replay_calendar)
            self.progress.done_tasks = current_progress
        else:
            self.progress.done_tasks = 1

        if (not self.progress.tick_size) \
                or self.progress.progress >= self.progress.tick_size + self.progress.last_output \
                or self.progress.is_done:
            self.progress.output()

        return data

    def __next__(self) -> MarketData:
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
