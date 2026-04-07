import abc
import uuid

from cpython.unicode cimport PyUnicode_FromString, PyUnicode_AsUTF8
from libc.math cimport NAN, isnan
from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy, strdup

from . import LOGGER
from ..base.c_market_data_ng.c_market_data cimport MarketData, md_data_type, c_md_get_price, c_md_dtype_name
from ..exchange_profile.c_exchange_profile cimport PROFILE, ExchangeProfile


class MarketDataMonitor(object, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: int = 0):
        self.name: str = name
        self.monitor_id: int = monitor_id or uuid.uuid4().int
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

    cdef void c_on_market_data(self, const md_variant* data_ptr):
        self.__call__(market_data=MarketData.c_from_header(<md_variant*> data_ptr, False))

    def __call__(self, MarketData market_data):
        for monitor_id in self.monitor:
            self._work(monitor_id=monitor_id, market_data=market_data)

    def __contains__(self, uint64_t monitor_id):
        return monitor_id in self.monitor

    cpdef void add_monitor(self, object monitor):
        self.monitor[monitor.monitor_id] = monitor

    cpdef void pop_monitor(self, str monitor_id):
        self.monitor.pop(monitor_id)

    cpdef void clear_monitors(self):
        self.monitor.clear()

    def _work(self, str monitor_id, MarketData market_data):
        monitor = self.monitor.get(monitor_id)
        if monitor is not None and monitor.enabled:
            monitor.__call__(market_data)

    def start(self):
        pass

    def stop(self):
        pass

    def clear(self):
        self.monitor.clear()

    property values:
        def __get__(self):
            cdef dict values = {}
            cdef object monitor
            for monitor in self.monitor.values():
                values.update(monitor.value)
            return values


cdef class MarketDataService:
    def __cinit__(self, ExchangeProfile profile=None):
        self.monitor_manager = MonitorManager()
        self.subscription_capacity = 0
        self.subscription_mapping = {}
        self.subscription_status = NULL  # Will be extended latter

        self.profile = PROFILE if profile is None else profile
        self.timestamp = NAN
        self.monitor = {}

    def __dealloc__(self):
        self.clear()

    cdef void c_subscription_buffer_extend(self):
        cdef size_t new_cap = self.subscription_capacity * 2
        if not new_cap:
            new_cap = 65535

        cdef mds_subscription* buffer = <mds_subscription*> calloc(new_cap, sizeof(mds_subscription))
        if not buffer:
            raise MemoryError('Failed to extend mds_subscription buffer')
        if self.subscription_status and self.subscription_capacity:
            memcpy(buffer, self.subscription_status, self.subscription_capacity * sizeof(mds_subscription))
        free(self.subscription_status)
        self.subscription_status = buffer
        self.subscription_capacity = new_cap

    cdef void c_update_subscription(self, const md_variant* md):
        cdef const char* ticker = md.meta_info.ticker
        cdef md_data_type dtype = md.meta_info.dtype
        cdef PyObject* py_idx = PyDict_GetItemString(<PyObject*> self.subscription_mapping, ticker)
        cdef size_t idx = 0
        if py_idx:
            idx = PyLong_AsSize_t(py_idx)
        else:
            idx = self.n_subscribed
            if idx >= self.subscription_capacity:
                self.c_subscription_buffer_extend()
            PyDict_SetItemString(<PyObject*> self.subscription_mapping, ticker, PyLong_FromSize_t(idx))
            LOGGER.info(f'MDS confirmed [{PyUnicode_FromString(ticker)}] <{PyUnicode_FromString(c_md_dtype_name(dtype))}> subscribed!')
            (self.subscription_status + idx).ticker = strdup(ticker)
            self.n_subscribed += 1
        cdef mds_subscription* subscription = self.subscription_status + idx
        subscription.last_update = md.meta_info.timestamp
        subscription.last_price = c_md_get_price(md)
        subscription.n_feeds += 1
        if dtype == md_data_type.DTYPE_TRANSACTION:
            subscription.is_subscribed_td = True
        elif dtype == md_data_type.DTYPE_ORDER:
            subscription.is_subscribed_od = True
        elif dtype == md_data_type.DTYPE_TICK_LITE:
            subscription.is_subscribed_tk_lite = True
        elif dtype == md_data_type.DTYPE_TICK:
            subscription.is_subscribed_tk = True
        elif dtype == md_data_type.DTYPE_BAR:
            subscription.is_subscribed_cs = True
        else:
            raise ValueError(f'Unknown md_data_type {dtype}')

    cdef void c_on_internal_data(self, const md_internal* md):
        self.monitor_manager.c_on_market_data(<const md_variant*> md)

    cdef void c_on_market_data(self, const md_variant* md):
        cdef double timestamp = md.meta_info.timestamp
        cdef md_data_type dtype = md.meta_info.dtype
        if dtype == md_data_type.DTYPE_INTERNAL:
            raise TypeError('Internal data must be passed in using on_internal_data method')
        self.c_update_subscription(md)
        self.timestamp = timestamp
        self.monitor_manager.c_on_market_data(md)

    # === Python Interfaces ===

    def __len__(self):
        return self.n_subscribed

    def __call__(self, MarketData market_data):
        self.c_on_market_data(market_data.header)

    def __getitem__(self, str monitor_id):
        return self.monitor[monitor_id]

    cpdef void on_internal_data(self, InternalData internal_data):
        self.c_on_internal_data(<const md_internal*> internal_data.header)

    cpdef void on_market_data(self, MarketData market_data):
        self.c_on_market_data(<const md_variant*> market_data.header)

    cpdef double get_market_price(self, str ticker):
        cdef const char* key = PyUnicode_AsUTF8(ticker)
        cdef PyObject* py_idx = PyDict_GetItemString(<PyObject*> self.subscription_mapping, key)
        if not py_idx:
            return NAN
        cdef size_t idx = PyLong_AsSize_t(py_idx)
        cdef mds_subscription* subscription = self.subscription_status + idx
        if subscription.n_feeds == 0:
            return NAN
        return subscription.last_price

    def add_monitor(self, monitor: MarketDataMonitor, **kwargs):
        self.monitor[monitor.monitor_id] = monitor
        self.monitor_manager.add_monitor(monitor, **kwargs)
        # remove the mds attr from the monitor as it is misleading
        # when using the multiprocessing the state of mds in child process is not complete.
        # thus using it will cause problem.
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

    def clear(self):
        cdef size_t i
        cdef mds_subscription* subscription
        for i in range(self.n_subscribed):
            subscription = self.subscription_status + i
            if subscription.ticker:
                free(<void*> subscription.ticker)
        self.n_subscribed = 0
        free(self.subscription_status)
        self.subscription_status = NULL
        self.subscription_capacity = 0

        self.subscription_mapping.clear()
        self.monitor.clear()
        self.monitor_manager.clear()
        self.timestamp = NAN

    property market_price:
        def __get__(self):
            cdef dict out = {}
            cdef size_t idx
            cdef mds_subscription* subscription
            for idx in range(self.n_subscribed):
                subscription = self.subscription_status + idx
                if subscription.n_feeds:
                    PyDict_SetItemString(<PyObject*> out, subscription.ticker, PyFloat_FromDouble(subscription.last_price))
            return out

    property market_time:
        def __get__(self):
            if isnan(self.timestamp):
                return None
            return self.profile.c_timestamp_to_datetime(self.timestamp)

    property market_date:
        def __get__(self):
            if isnan(self.timestamp):
                return None
            # with custom profile set, c_ex_profile_session_date_from_unix is not the proper way to convert timestamp to date
            # as the tzinfo might be different
            return self.profile.c_timestamp_to_datetime(self.timestamp).date()

    property subscriptions:
        def __get__(self):
            return self.subscription_mapping


cdef MarketDataService MDS = MarketDataService()
globals()['MDS'] = MDS
