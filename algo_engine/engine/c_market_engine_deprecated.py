import abc
import datetime
import uuid
from collections import defaultdict
from math import inf

from . import LOGGER, Singleton
from ..base import MarketData, TickData, TransactionSide, TransactionDirection, DataType, InternalData
from ..profile import PROFILE, Profile

LOGGER = LOGGER.getChild('MarketEngine')

__all__ = ['MDS', 'MarketDataService', 'MarketDataMonitor', 'MonitorManager']


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


class MonitorManager(object, metaclass=Singleton):
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

    def add_monitor(self, monitor: MarketDataMonitor, **kwargs):
        self.monitor[monitor.monitor_id] = monitor

    def pop_monitor(self, monitor_id: str, **kwargs) -> MarketDataMonitor:
        return self.monitor.pop(monitor_id)

    def _work(self, monitor_id: str, market_data: MarketData):
        monitor = self.monitor.get(monitor_id)
        if monitor is not None and monitor.enabled:
            monitor.__call__(market_data)

    def start(self):
        pass

    def stop(self):
        pass

    def clear(self):
        self.monitor.clear()

    @property
    def values(self) -> dict[str, float]:
        values = {}

        for monitor in self.monitor.values():
            values.update(monitor.value)

        return values


class MarketDataService(object, metaclass=Singleton):
    def __init__(self, profile: Profile = None, **kwargs):
        self.profile = PROFILE if profile is None else profile
        self.cache_history = kwargs.pop('cache_history', False)

        self._market_price = {}
        self._market_history = defaultdict(dict)
        self._market_time: datetime.datetime | None = None
        self._timestamp: float | None = None

        self._market_data: dict[int, dict[str, MarketData]] = {}
        self._monitor: dict[str, MarketDataMonitor] = {}
        self._monitor_manager = MonitorManager()

    def __call__(self, **kwargs):
        if 'market_data' in kwargs:
            self.on_market_data(market_data=kwargs['market_data'])

    def __getitem__(self, monitor_id: str) -> MarketDataMonitor:
        return self.monitor[monitor_id]

    def add_monitor(self, monitor: MarketDataMonitor, **kwargs):
        self.monitor[monitor.monitor_id] = monitor
        self.monitor_manager.add_monitor(monitor, **kwargs)
        # remove the mds attr from the monitor as it is misleading
        # when using the multiprocessing the state of mds in child process is not complete.
        # thus using it will causes problem.
        # an alternative is to create a shared contexts monitor.
        # monitor.mds = self

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

    def on_internal_data(self, internal_data: InternalData):
        self.monitor_manager.__call__(market_data=internal_data)

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

        ticker = market_data.ticker
        dtype = market_data.dtype
        if dtype in self._market_data:
            snapshot = self._market_data[dtype]
        elif dtype == DataType.DTYPE_INTERNAL:
            raise TypeError('Internal data must be passed in using on_internal_data method')
        else:
            snapshot = self._market_data[dtype] = {}

        if ticker not in snapshot:
            LOGGER.info(f'MDS confirmed {ticker} {market_data.__class__.__name__} subscribed!')

        snapshot[ticker] = market_data

        self.monitor_manager.__call__(market_data=market_data)

    def get_queued_volume(self, ticker: str, side: TransactionSide | TransactionDirection, p_min: float, p_max: float = None) -> float:
        """
        get queued volume prior / posterior to given price, NOT COUNTING GIVEN PRICE!
        :param ticker: the given ticker
        :param side: the given trade side
        :param p_min: the given price
        :param p_max: optional the given posterior price
        :return: the summed queued volume, in float. 0 if not available.
        """
        tick_data: TickData = self._market_data[DataType.DTYPE_TICK.value].get(ticker)

        if tick_data is None:
            return 0.

        sign = side.sign

        if sign > 0:
            book = tick_data.bid
        elif sign < 0:
            book = tick_data.ask
        else:
            raise ValueError(f'Invalid side {side}')

        if p_min is None:
            p_min = -inf

        if p_max is None:
            p_max = inf

        return book.loc_volume(p0=p_min, p1=p_max)

    def trade_time_between(self, start_time: datetime.datetime | float, end_time: datetime.datetime | float, **kwargs) -> datetime.timedelta:
        return self.profile.trade_time_between(start_time=start_time, end_time=end_time, **kwargs)

    def is_market_session(self, market_time: datetime.datetime | float | int) -> bool:
        return self.profile.is_market_session(timestamp=market_time)

    def clear(self):
        # self._market_price.clear()
        # self._market_time = None
        # self._timestamp = None

        self._market_history.clear()
        self._market_data.clear()
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
                return datetime.datetime.fromtimestamp(self._timestamp, tz=self.profile.time_zone)
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


MDS = MarketDataService()
