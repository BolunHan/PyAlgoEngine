import abc
import datetime
import threading
import uuid
from collections import defaultdict
from typing import Dict, List, Union, Optional, Tuple

from PyQuantKit import TickData, TradeData, OrderBook, MarketData, Progress, TransactionSide, BarData

from . import LOGGER

__all__ = ['MDS', 'MarketDataService', 'ProgressiveReplay', 'SimpleReplay', 'Replay']
LOGGER = LOGGER.getChild('MarketEngine')


class MarketDataService(object):
    class OrderLog(object):
        def __init__(self, ticker: str, price: float, volume: float, side: TransactionSide):
            self.ticker = ticker
            self.price = price
            self.volume = volume
            self.side = side

    def __init__(self, **kwargs):
        self.synthetic_orderbook = kwargs.pop('synthetic_orderbook', False)
        self.cache_history = kwargs.pop('cache_history', False)
        self.session_start: Optional[datetime.time] = None
        self.session_end: Optional[datetime.time] = None
        self.session_break: Tuple[datetime.time, :datetime.time] = None

        self._market_price = {}
        self._market_history = defaultdict(dict)
        self._market_time: Optional[datetime.datetime] = None
        self._timestamp: Optional[float] = None

        self._order_book: Dict[str, OrderBook] = {}
        self._tick_data: Dict[str, TickData] = {}
        self._trade_data: Dict[str, TradeData] = {}
        self._minute_bar_data: Dict[str, BarData] = {}
        self._last_bar_data: Dict[str, BarData] = {}

        self._order_log: Dict[str, MarketDataService.OrderLog] = {}
        self.lock = threading.Lock()

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

    def init_cn_override(self):
        from FactorGraph.Data import StockLib

        self.trade_time_between = StockLib.QUERY.trade_time_between
        self.in_trade_session = StockLib.QUERY.in_trade_session
        self.session_start = datetime.time(9, 30)
        self.session_end = datetime.time(15, 0)
        self.session_break = [datetime.time(11, 30), datetime.time(13, 0)]

    def _on_trade_data(self, trade_data: TradeData):
        ticker = trade_data.ticker

        if ticker not in self._trade_data:
            LOGGER.info(f'MDS confirmed {ticker} TradeData subscribed!')

        self._trade_data[ticker] = trade_data

        if self.synthetic_orderbook and (order_book := self._order_book.get(ticker)):

            if order_book.market_time <= trade_data.market_time:
                side: TransactionSide = trade_data.side
                price = trade_data.price
                book = order_book.ask if side.sign > 0 else order_book.bid
                listed_volume = book.at_price(price).volume if price in book else 0.
                traded_volume = trade_data.volume
                book.update_entry(price=price, volume=max(0, listed_volume - traded_volume))

                for order_id in list(self._order_log):
                    order_log = self._order_log.get(order_id)

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
                        self._order_log.pop(order_id)

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

    def _update_last_bar(self, market_data: MarketData, interval: float = 60.):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = market_data.market_time

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            # update bar_data
            if ticker in self._minute_bar_data:
                self._last_bar_data[ticker] = self._minute_bar_data[ticker]

            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                bar_start_time=datetime.datetime.fromtimestamp(self._timestamp // interval * interval),
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

    def _update_active_bar(self, market_data: MarketData, interval: float = 60.):
        ticker = market_data.ticker
        market_price = market_data.market_price
        market_time = MarketData.market_time

        if ticker not in self._minute_bar_data or market_time >= self._minute_bar_data[ticker].bar_end_time:
            bar_data = self._minute_bar_data[ticker] = BarData(
                ticker=ticker,
                bar_start_time=datetime.datetime.fromtimestamp(self._timestamp - interval),
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

        history: List[TradeData] = getattr(bar_data, 'history')
        bar_data.bar_start_time = datetime.datetime.fromtimestamp(self._timestamp - interval)

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

        self._update_last_bar(market_data=market_data)
        # self._update_active_bar(market_data=market_data)

        self.lock.release()

    def get_order_book(self, ticker: str) -> OrderBook:
        return self._order_book.get(ticker, None)

    def get_queued_volume(self, ticker: str, side: Union[TransactionSide, str, int], prior: float, posterior: float = None) -> float:
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

    def trade_time_between(self, start_time: Union[datetime.datetime, float], end_time: Union[datetime.datetime, float], **kwargs) -> datetime.timedelta:
        if start_time is not None and isinstance(start_time, (float, int)):
            start_time = datetime.datetime.fromtimestamp(start_time)

        if end_time is not None and isinstance(end_time, (float, int)):
            end_time = datetime.datetime.fromtimestamp(end_time)

        if start_time is None or end_time is None:
            return datetime.timedelta(seconds=0)

        if start_time > end_time:
            return datetime.timedelta(seconds=0)

        return end_time - start_time

    def in_trade_session(self, market_time: Union[datetime.datetime, float]) -> bool:
        return True

    def new_order_at(self, ticker: str, price: float, side: Union[TransactionSide, str, int]) -> str:
        order_id = uuid.uuid4().hex
        trade_side = TransactionSide(side)
        order_book = self._order_book.get(ticker)
        order_log = self.OrderLog(ticker=ticker, price=price, side=trade_side, volume=0.)

        if order_book is None:
            LOGGER.error(f'No available order book for {ticker}. MDS must be connected to the feed before using it.')
        else:
            if trade_side.sign:
                if trade_side.sign > 0:
                    book = order_book.bid
                else:
                    book = order_book.ask

                entry = book.get(price=price)
                if entry is not None:
                    order_log.volume = entry.volume
            else:
                LOGGER.error(f'Invalid trade side {side} for {ticker}')

            self._order_log[order_id] = order_log

        return order_id

    def where_is_order(self, order_id: str) -> float:
        """
        get queued volume prior to given order_id, NOT COUNTING PRIOR! use .get_queued_volume() to get queued volume before given price
        :param order_id: the given order id
        :return: float
        """
        if order_id in self._order_log:
            order_log = self._order_log[order_id]
            queued_volume = order_log.volume
        else:
            queued_volume = 0.

        return queued_volume

    def clear(self):
        self.lock.acquire()
        # self._market_price.clear()
        # self._market_time = None
        # self._timestamp = None

        self._market_history.clear()
        self._order_book.clear()
        self._order_log.clear()
        self._minute_bar_data.clear()
        self._last_bar_data.clear()
        self.lock.release()

    @property
    def market_price(self) -> Dict[str, float]:
        self.lock.acquire()
        result = self._market_price
        self.lock.release()
        return result

    @property
    def market_history(self) -> Dict[str, Dict[datetime.datetime, float]]:
        self.lock.acquire()
        result = self._market_history
        self.lock.release()
        return result

    @property
    def market_time(self) -> Optional[datetime.datetime]:
        if self._market_time is None:
            if self._timestamp is None:
                return None
            else:
                return datetime.datetime.fromtimestamp(self._timestamp)
        else:
            return self._market_time

    @property
    def market_date(self) -> Optional[datetime.date]:
        if self.market_time is None:
            return None

        return self._market_time.date()

    @property
    def timestamp(self) -> Optional[float]:
        if self._timestamp is None:
            if self._market_time is None:
                return None
            else:
                return self._market_time.timestamp()
        else:
            return self._timestamp


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
    loader: a data loading function. Expect loader = Callable(market_date: datetime.date, ticker: str, dtype: Union[str, type]) -> Dict[any, MarketData]
    start_date & end_date: the given replay period
    or calendar: the given replay calendar.

    accepts kwargs:
    ticker / tickers: the given symbols to replay, expect a Union[str, List[str]]
    dtype / dtypes: the given dtype(s) of symbol to replay, expect a Union[Union[str, type], List[Union[str, type]]]. default = all, which is (TradeData, TickData, OrderBook)
    subscription / subscribe: the given ticker-dtype pair to replay, expect a List[Dict[str, Union[str, type]]]
    """

    def __init__(
            self,
            loader,
            **kwargs
    ):
        self.loader = loader
        self.start_date: datetime.date = kwargs.pop('start_date', None)
        self.end_date: datetime.date = kwargs.pop('end_date', None)
        self.calendar: List[datetime.date] = kwargs.pop('calendar', None)

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

    def add_subscription(self, ticker: str, dtype: Union[type, str]):
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
