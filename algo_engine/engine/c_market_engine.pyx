import abc
import uuid

from cpython.datetime cimport datetime
from libc.math cimport NAN, isnan
from libc.stdint cimport uint8_t, uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

from . import LOGGER
from ..base.c_market_data cimport _MetaInfo, DataType, _MarketDataVirtualBase, _InternalBuffer, _TransactionDataBuffer, _OrderDataBuffer, _TickDataLiteBuffer, _TickDataBuffer, _CandlestickBuffer
from ..base.c_market_data import MarketData
from ..profile cimport C_PROFILE


C_MDS = MarketDataService()
MDS = C_MDS


class MarketDataMonitor(object, metaclass=abc.ABCMeta):
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



cdef class MonitorManager:

    def __cinit__(self):
        self.monitor = {}

    cdef void c_on_market_data(self, _MarketDataBuffer* data_ptr):
        for monitor_id in self.monitor:
            self._work(monitor_id=monitor_id, market_data=_MarketDataVirtualBase.c_ptr_to_data(data_ptr))

    def __call__(self, market_data: MarketData):
        for monitor_id in self.monitor:
            self._work(monitor_id=monitor_id, market_data=market_data)

    cpdef void add_monitor(self, object monitor):
        self.monitor[monitor.monitor_id] = monitor

    cpdef void pop_monitor(self, str monitor_id):
        self.monitor.pop(monitor_id)

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


cdef class MarketDataService:
    def __cinit__(self, object profile=None, size_t max_subscription=65535):
        self.profile = C_PROFILE
        self.max_subscription = max_subscription
        self.mapping = {}

        self._monitor = {}
        self._monitor_manager = MonitorManager()
        self._n = <size_t> len(self.mapping)

        self._timestamp: float
        self._market_price = <double*> malloc(self.max_subscription * sizeof(double))
        self._subscription_trade_data = <bint*> malloc(self.max_subscription * sizeof(bint))
        self._subscription_order_data = <bint*> malloc(self.max_subscription * sizeof(bint))
        self._subscription_tick_data_lite = <bint*> malloc(self.max_subscription * sizeof(bint))
        self._subscription_tick_data = <bint*> malloc(self.max_subscription * sizeof(bint))
        self._subscription_bar_data = <bint*> malloc(self.max_subscription * sizeof(bint))

        memset(<void*> self._market_price, 0, self.max_subscription * sizeof(double))
        memset(<void*> self._subscription_trade_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_order_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_tick_data_lite, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_tick_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_bar_data, 0, self.max_subscription * sizeof(bint))

        if (self._market_price == NULL
                or self._subscription_trade_data == NULL
                or self._subscription_order_data == NULL
                or self._subscription_tick_data_lite == NULL
                or self._subscription_tick_data == NULL
                or self._subscription_bar_data == NULL):
            raise MemoryError("Could not allocate memory for slice")

    def __dealloc__(self):
        if self._market_price != NULL:
            free(self._market_price)

        if self._subscription_trade_data != NULL:
            free(self._subscription_trade_data)

        if self._subscription_order_data != NULL:
            free(self._subscription_order_data)

        if self._subscription_tick_data_lite != NULL:
            free(self._subscription_tick_data_lite)

        if self._subscription_tick_data != NULL:
            free(self._subscription_tick_data)

        if self._subscription_bar_data != NULL:
            free(self._subscription_bar_data)

    cdef void c_on_internal_data(self, _MarketDataBuffer* data_ptr):
        self._monitor_manager.c_on_market_data(data_ptr=data_ptr)

    cdef void c_on_market_data(self, _MarketDataBuffer* data_ptr):
        cdef _MetaInfo* meta_info = <_MetaInfo*> data_ptr
        cdef uint8_t dtype = meta_info.dtype
        cdef bytes ticker_bytes = meta_info.ticker
        cdef double timestamp = meta_info.timestamp
        cdef double market_price

        cdef size_t idx
        cdef _InternalBuffer* internal_data
        cdef _TransactionDataBuffer* transaction_data
        cdef _OrderDataBuffer* order_data
        cdef _TickDataLiteBuffer* tick_data_lite
        cdef _TickDataBuffer* tick_data
        cdef _CandlestickBuffer* bar_data
        cdef bint* subscription

        if dtype == DataType.DTYPE_INTERNAL:
            raise TypeError('Internal data must be passed in using on_internal_data method')

        if ticker_bytes in self.mapping:
            idx = self.mapping[ticker_bytes]
        else:
            idx = self.mapping[ticker_bytes] = self._n
            self._n += 1

        if dtype == DataType.DTYPE_TRANSACTION:
            transaction_data = <_TransactionDataBuffer*> data_ptr
            market_price = transaction_data.price
            subscription = self._subscription_trade_data
        elif dtype == DataType.DTYPE_ORDER:
            order_data = <_OrderDataBuffer*> data_ptr
            market_price = order_data.price
            subscription = self._subscription_order_data
        elif dtype == DataType.DTYPE_TICK_LITE:
            tick_data_lite = <_TickDataLiteBuffer*> data_ptr
            market_price = tick_data_lite.last_price
            subscription = self._subscription_tick_data_lite
        elif dtype == DataType.DTYPE_TICK:
            tick_data = <_TickDataBuffer*> data_ptr
            market_price = tick_data.lite.last_price
            subscription = self._subscription_tick_data
        elif dtype == DataType.DTYPE_BAR:
            bar_data = <_CandlestickBuffer*> data_ptr
            market_price = bar_data.close_price
            subscription = self._subscription_bar_data
        else:
            raise ValueError(f'Unknown data type {dtype}')

        self._market_price[idx] = market_price
        self._timestamp = timestamp
        if not subscription[idx]:
            subscription[idx] = True
            LOGGER.info(f'MDS confirmed {ticker_bytes.decode("utf-8")} {_MarketDataVirtualBase.c_dtype_name(dtype)} subscribed!')

        self._monitor_manager.c_on_market_data(data_ptr=data_ptr)

    # --- python interface ---
    def __len__(self):
        return self._n

    def __call__(self, object market_data):
        self.on_market_data(market_data=market_data)

    def __getitem__(self, monitor_id: str) -> MarketDataMonitor:
        return self._monitor[monitor_id]

    cpdef void on_internal_data(self, object internal_data):
        cdef uintptr_t data_addr = internal_data._data_addr
        self.c_on_internal_data(data_ptr=<_MarketDataBuffer*> data_addr)

    cpdef void on_market_data(self, object market_data):
        cdef uintptr_t data_addr = market_data._data_addr
        self.c_on_market_data(data_ptr=<_MarketDataBuffer*> data_addr)

    cpdef double get_market_price(self, str ticker):
        cdef bytes ticker_bytes = ticker.encode('utf-8')

        if ticker_bytes not in self.mapping:
            return NAN

        cdef size_t idx = self.mapping[ticker_bytes]
        return self._market_price[idx]

    def add_monitor(self, monitor: MarketDataMonitor, **kwargs):
        self._monitor[monitor.monitor_id] = monitor
        self._monitor_manager.add_monitor(monitor, **kwargs)
        # remove the mds attr from the monitor as it is misleading
        # when using the multiprocessing the state of mds in child process is not complete.
        # thus using it will causes problem.
        # an alternative is to create a shared contexts monitor.
        # monitor.mds = self

    def pop_monitor(self, monitor: MarketDataMonitor = None, monitor_id: str = None, monitor_name: str = None):
        if monitor_id is not None:
            pass
        elif monitor_name is not None:
            for _ in list(self._monitor.values()):
                if _.name == monitor_name:
                    monitor_id = _.monitor_id
            if monitor is None:
                LOGGER.error(f'monitor_name {monitor_name} not registered.')
        elif monitor is not None:
            monitor_id = monitor.monitor_id
        else:
            LOGGER.error('must assign a monitor, or monitor_id, or monitor_name to pop.')
            return None

        self._monitor.pop(monitor_id)
        self._monitor_manager.pop_monitor(monitor_id)

    def clear(self):
        self.mapping.clear()
        self._n = 0

        self._monitor.clear()
        self._monitor_manager.clear()

        self._timestamp = NAN

        memset(<void*> self._market_price, 0, self.max_subscription * sizeof(double))
        memset(<void*> self._subscription_trade_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_order_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_tick_data_lite, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_tick_data, 0, self.max_subscription * sizeof(bint))
        memset(<void*> self._subscription_bar_data, 0, self.max_subscription * sizeof(bint))

    @property
    def market_price(self) -> dict[str, float]:
        cdef size_t idx
        cdef bytes ticker_bytes
        cdef dict result = {}

        for idx, ticker_bytes in self.mapping.items():
            result[ticker_bytes.decode('utf-8')] = self._market_price[idx]

        return result

    @property
    def market_time(self) -> datetime | None:
        if isnan(self._timestamp):
            return None
        return _MarketDataVirtualBase.c_to_dt(self._timestamp)

    @property
    def market_date(self) -> datetime.date | None:
        if isnan(self._timestamp):
            return None
        return _MarketDataVirtualBase.c_to_dt(self._timestamp).date()

    @property
    def timestamp(self) -> float | None:
        if isnan(self._timestamp):
            return None
        return self._timestamp

    @property
    def monitor(self) -> dict[str, MarketDataMonitor]:
        return self._monitor

    @property
    def monitor_manager(self) -> object:
        return self._monitor_manager

    @monitor_manager.setter
    def monitor_manager(self, object manager):
        self._monitor_manager.clear()

        self._monitor_manager = <MonitorManager> manager

        for monitor in self._monitor.values():
            self._monitor_manager.add_monitor(monitor=monitor)
