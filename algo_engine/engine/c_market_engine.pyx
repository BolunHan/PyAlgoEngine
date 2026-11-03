import abc
import uuid
from typing import Any

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.ref cimport Py_XDECREF
from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromString, PyUnicode_FromStringAndSize
from libc.math cimport NAN
from libc.string cimport memcpy, strlen

from cbase.bytemap cimport bytemap_entry, c_bytemap_ex_len, c_bytemap_gen_seq_id

from algo_engine.base.c_allocator_protocol cimport MD_DEFAULT_ALLOCATOR
from algo_engine.base.c_market_data.c_market_data cimport FilterMode, c_md_dtype_name
from algo_engine.exchange_profile.c_exchange_profile cimport PROFILE

from . import LOGGER


class MarketDataMonitorProtocol(object, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: Any = None):
        self.name = name
        self.monitor_id = monitor_id or uuid.uuid4()
        self.enabled = True

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


cdef class MarketDataMonitor(CCPType):
    def __init__(self, str name, md_filter_flag filter_mode=<md_filter_flag> 0, int16_t priority=0, object monitor_id=None):
        self.name = name

        self.header = c_mds_monitor_new(PyUnicode_AsUTF8(name), priority, filter_mode, MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True

        if monitor_id is None:
            c_mds_monitor_gen_uid(
                self.header,
                self.header.name,
                strlen(self.header.name) if self.header.name else 0
            )
        else:
            self.c_set_monitor_id(monitor_id)

        self.ccp_bind(&self.header)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header:
            c_mds_monitor_free(self.header)
            self.header = NULL

    cdef void c_set_monitor_id(self, object monitor_id):
        cdef bytes id_bytes

        if isinstance(monitor_id, uuid.UUID):
            id_bytes = monitor_id.bytes
        elif isinstance(monitor_id, int):
            id_bytes = uuid.UUID(int=monitor_id).bytes
        elif isinstance(monitor_id, bytes) and len(monitor_id) == 16:
            id_bytes = monitor_id
        else:
            raise TypeError(
                f'<{self.__class__.__name__}> requires a uuid.UUID, int uid or 16-byte id, '
                f'got {type(monitor_id).__name__}.'
            )

        memcpy(&self.header.monitor_id[0], <const char*> id_bytes, 16)

    @staticmethod
    cdef MarketDataMonitor c_from_header(mds_monitor* header, bint owner):
        cdef MarketDataMonitor instance = MarketDataMonitor.__new__(MarketDataMonitor)
        instance.header = header
        instance.owner = owner

        if header:
            instance.name = PyUnicode_FromString(header.name) if header.name else None
            instance.ccp_bind(&instance.header)

        return instance

    def __call__(self, MarketData market_data, **kwargs):
        pass

    cpdef void clear(self):
        pass

    property monitor_id:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return uuid.UUID(bytes=PyBytes_FromStringAndSize(<const char*> self.header.monitor_id, 16))

    property enabled:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.enabled

        def __set__(self, bint value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.enabled = value

    property is_ready:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.is_ready

        def __set__(self, bint value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.is_ready = value

    property priority:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return self.header.priority

        def __set__(self, int16_t value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.priority = value

    property md_filter:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return FilterMode(self.header.md_filter)

        def __set__(self, md_filter_flag value):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            self.header.md_filter = value

    property value:
        def __get__(self):
            return None


MarketDataMonitorProtocol.register(MarketDataMonitor)


cdef class MarketDataMonitorMap(BoundByteMap):
    @staticmethod
    cdef MarketDataMonitorMap c_from_header(bytemap* header, bint owner=False):
        if owner:
            raise PermissionError('Must not assign the monitor registry ownership to <MarketDataMonitorMap>')

        cdef MarketDataMonitorMap instance = MarketDataMonitorMap.__new__(MarketDataMonitorMap)
        instance.owner = False
        instance.seq_id = c_bytemap_gen_seq_id(<void*> instance)
        instance.c_bind(header)
        return instance

    cdef const char* c_serialize_key(self, object obj, size_t* key_len):
        if not isinstance(obj, uuid.UUID):
            raise TypeError(
                f'<{self.__class__.__name__}> requires a uuid.UUID key, '
                f'got {type(obj).__name__}.'
            )

        self.ws_key = obj.bytes
        if key_len:
            key_len[0] = 16
        return <const char*> self.ws_key

    cdef object c_deserialize_key(self, const char* key, size_t key_len):
        if key_len != 16:
            raise ValueError(f'<{self.__class__.__name__}> monitor id key must be 16 bytes, got {key_len}.')
        return uuid.UUID(bytes=PyBytes_FromStringAndSize(key, key_len))

    cdef const char* c_serialize_value(self, object obj, size_t* value_len):
        if not isinstance(obj, MarketDataMonitor):
            raise TypeError(
                f'<{self.__class__.__name__}> requires a MarketDataMonitor value, '
                f'got {type(obj).__name__}.'
            )

        self.ws_value = <void*> (<MarketDataMonitor> obj).header
        if value_len:
            value_len[0] = sizeof(void*)
        return <const char*> &self.ws_value

    cdef object c_deserialize_value(self, const char* value, size_t value_len):
        cdef void**       value_ptr = <void**> value
        cdef mds_monitor* monitor   = <mds_monitor*> value_ptr[0]

        if not monitor:
            return None
        return MarketDataMonitor.c_from_header(monitor, False)


cdef class MonitorManager(CCPType):
    def __init__(self):
        self.header = c_mds_monitor_manager_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.owner = True
        self.ccp_bind(&self.header)

        cdef int ret_code = c_mds_monitor_manager_register_callback(
            self.header,
            self.c_callback_adaptor,
            <void*> <PyObject*> self,
            &self.callback_id
        )

        if ret_code != mds_ret_code.MDS_OK:
            c_mds_monitor_manager_free(self.header)
            self.header = NULL
            self.owner = False
            raise RuntimeError(f'Failed to register callback for <{self.__class__.__name__}>, err code: {ret_code}')

        self.monitor = MarketDataMonitorMap.c_from_header(self.header.monitors, False)

    def __dealloc__(self):
        if not self.owner:
            return

        if self.header and self.callback_id:
            c_mds_monitor_manager_unregister_callback(self.header, self.callback_id)
            self.callback_id = 0

        if self.header:
            c_mds_monitor_manager_free(self.header)
            self.header = NULL

    # === Cython Internal Binding ===

    @staticmethod
    cdef inline void c_callback_adaptor(mds_monitor_manager* manager, const md_variant* market_data, void* user_data) noexcept with gil:
        cdef MonitorManager self = <MonitorManager> <object> <PyObject*> user_data
        cdef MarketData py_md = MarketData.c_from_header(market_data, False)
        cdef PyObject* res = NULL
        cdef PyObject* exc = NULL
        cdef MarketDataMonitor monitor
        cdef object monitor_id

        for monitor_id in self.monitor:
            monitor = self.monitor.get(monitor_id)
            if monitor is None or not monitor.enabled:
                continue

            res = PyObject_CallFunctionObjArgs(<PyObject*> monitor, <PyObject*> py_md, NULL)
            if not res:
                exc = PyErr_GetRaisedException()
                if exc:
                    PyErr_DisplayException(exc)
                    Py_XDECREF(exc)
            else:
                Py_XDECREF(res)

    cdef mds_monitor* c_get_monitor(self, object monitor_id):
        cdef MarketDataMonitor monitor = self.monitor.get(monitor_id)
        if monitor is None or not isinstance(monitor, MarketDataMonitor):
            return NULL
        return monitor.header

    # === Market Data Feed Skeleton ===

    cpdef void feed_monitor(self, object monitor_id, MarketData market_data):
        cdef object monitor = self.monitor.get(monitor_id)
        if monitor is not None and monitor.enabled:
            monitor.__call__(market_data)

    cdef void c_on_market_data(self, const md_variant* market_data):
        cdef MarketData py_md = MarketData.c_from_header(market_data, False)
        cdef object monitor_id
        for monitor_id in self.monitor:
            self.feed_monitor(monitor_id, py_md)

    cpdef void on_market_data(self, MarketData market_data):
        cdef object monitor_id
        for monitor_id in self.monitor:
            self.feed_monitor(monitor_id, market_data)

    def __call__(self, MarketData market_data):
        cdef object monitor_id
        for monitor_id in self.monitor:
            self.feed_monitor(monitor_id, market_data)

    # === Monitor Management ===

    cpdef void add_monitor(self, object monitor):
        self.monitor[monitor.monitor_id] = monitor

    cpdef void pop_monitor(self, object monitor_id):
        self.monitor.pop(monitor_id)

    cpdef void clear_monitors(self):
        self.monitor.clear()

    def __contains__(self, object monitor_id):
        return monitor_id in self.monitor

    # === Lifecycle Management ===

    cpdef void start(self):
        pass

    cpdef void stop(self):
        pass

    cpdef void clear(self):
        self.clear_monitors()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # === Monitor Value Protocol Skeleton ===

    cpdef dict get_values(self):
        cdef dict values = {}
        cdef object monitor
        cdef object monitor_value
        for monitor in self.monitor.values():
            monitor_value = monitor.value
            if monitor_value is None:
                continue
            if isinstance(monitor_value, dict):
                values.update(monitor_value)
            else:
                values[monitor.name] = monitor_value
        return values

    property values:
        def __get__(self):
            cdef dict values = {}
            cdef object monitor
            cdef object monitor_value
            for monitor in self.monitor.values():
                monitor_value = monitor.value
                if monitor_value is None:
                    continue
                if isinstance(monitor_value, dict):
                    values.update(monitor_value)
                else:
                    values[monitor.name] = monitor_value
            return values

    property n_monitors:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_bytemap_ex_len(self.header.monitors)


cdef class MarketDataService:
    def __init__(self, ExchangeProfile profile=None):
        self.profile = PROFILE if profile is None else profile

        self.header = c_mds_new(MD_DEFAULT_ALLOCATOR)
        if not self.header:
            raise MemoryError(f'Failed to allocate memory for <{self.__class__.__name__}>.')

        self.monitor = MarketDataMonitorMap.c_from_header(self.header.monitors)
        self.owner = True
        self.monitor_manager = MonitorManager()
        c_mds_set_manager(self.header, self.monitor_manager.header)

    def __dealloc__(self):
        if self.owner and self.header:
            c_mds_free(self.header)
            self.header = NULL

    # === Cython Internal Binding ===

    cdef void c_on_internal_data(self, const md_internal* md):
        cdef int ret_code = c_mds_on_internal_data(self.header, md)
        if ret_code != mds_ret_code.MDS_OK:
            raise RuntimeError(f'c_mds_on_internal_data failed with err code: {ret_code}')

    cdef void c_on_market_data(self, const md_variant* md):
        cdef int ret_code = c_mds_on_market_data(self.header, md)

        if ret_code >= 0:
            if ret_code == mds_ret_code.MDS_OK_SUBSCRIBED:
                LOGGER.info(
                    f'MDS confirmed [{PyUnicode_FromString(md.meta_info.ticker)}] '
                    f'<{PyUnicode_FromString(c_md_dtype_name(md.meta_info.dtype))}> subscribed!'
                )
            return
        elif ret_code == mds_ret_code.MDS_ERR_INTERNAL_DATA:
            raise TypeError('Internal data must be passed in using on_internal_data method')
        elif ret_code == mds_ret_code.MDS_ERR_UNKNOWN_DTYPE:
            raise ValueError(f'Unknown md_data_type {md.meta_info.dtype}')
        else:
            raise RuntimeError(f'c_mds_on_market_data failed with err code: {ret_code}')

    # === Python Interfaces ===

    def __len__(self):
        if not self.header:
            raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
        return c_bytemap_ex_len(self.header.subscriptions)

    def __call__(self, MarketData market_data):
        self.c_on_market_data(market_data.header)

    def __getitem__(self, object monitor_id):
        return self.monitor[monitor_id]

    cpdef void on_internal_data(self, InternalData internal_data):
        self.c_on_internal_data(<const md_internal*> internal_data.header)

    cpdef void on_market_data(self, MarketData market_data):
        self.c_on_market_data(<const md_variant*> market_data.header)

    cpdef double get_market_price(self, str ticker):
        cdef const char* key = PyUnicode_AsUTF8(ticker)
        return self.c_get_market_price(key)

    cdef double c_get_market_price(self, const char* ticker):
        return c_mds_get_market_price(self.header, ticker)

    cpdef void set_manager(self, MonitorManager manager):
        # Resolve the new C base first: it may allocate (the base manager's dummy
        # block), and a failure must not leave the current registry half-moved.
        cdef mds_monitor_manager* base = NULL
        if manager is not None:
            base = manager.header

        cdef dict tmp_monitor = dict(self.monitor)

        for monitor_id in list(tmp_monitor):
            self.pop_monitor(monitor_id=monitor_id)

        c_mds_set_manager(self.header, base)
        self.monitor_manager = manager

        for monitor_id, monitor in tmp_monitor.items():
            self.add_monitor(monitor)

    cpdef void add_monitor(self, object monitor):
        cdef object monitor_id

        # Step 1: Add monitor to the monitor manager, in this step the monitor_id can be altered.
        if self.monitor_manager is not None:
            self.monitor_manager.add_monitor(monitor)

        monitor_id = monitor.monitor_id

        # Step 2: Register in the service registry under the resolved id.
        if monitor_id in self.monitor:
            if self.monitor_manager is not None:
                self.monitor_manager.pop_monitor(monitor_id)
            raise KeyError(f'monitor_id {monitor_id} already exists.')
        self.monitor[monitor_id] = monitor

    cpdef void pop_monitor(
            self,
            object monitor=None,
            object monitor_id=None,
            str monitor_name=None
    ):
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
            return

        self.monitor.pop(monitor_id)
        if self.monitor_manager is not None:
            self.monitor_manager.pop_monitor(monitor_id)

    cpdef void clear(self):
        if self.header:
            c_mds_clear(self.header)

        self.monitor.clear()
        if self.monitor_manager is not None:
            self.monitor_manager.clear()

    # === Python Properties ===

    property n_subscribed:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_bytemap_ex_len(self.header.subscriptions)

    property n_monitors:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            return c_mds_n_monitors(self.header)

    property timestamp:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.header.ts.date.year:
                return NAN
            return self.header.ts.unix_ts

    property market_price:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict out = {}
            cdef bytemap_entry* entry = self.header.subscriptions.first
            cdef const mds_subscription* subscription

            while entry:
                subscription = <const mds_subscription*> entry.value
                out[PyUnicode_FromStringAndSize(entry.key, entry.key_length)] = subscription.last_price
                entry = entry.next
            return out

    property market_time:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.header.ts.date.year:
                return None
            return self.profile.c_timestamp_to_datetime(self.header.ts.unix_ts)

    property market_date:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')
            if not self.header.ts.date.year:
                return None
            # with custom profile set, c_ex_profile_session_date_from_unix is not the proper way to convert timestamp to date
            # as the tzinfo might be different
            return self.profile.c_timestamp_to_datetime(self.header.ts.unix_ts).date()

    property subscriptions:
        def __get__(self):
            if not self.header:
                raise RuntimeError(f'<{self.__class__.__name__}> not initialized!')

            cdef dict out = {}
            cdef bytemap_entry* entry = self.header.subscriptions.first
            cdef size_t index = 0

            while entry:
                out[PyUnicode_FromStringAndSize(entry.key, entry.key_length)] = index
                index += 1
                entry = entry.next
            return out


cdef MarketDataService MDS = MarketDataService()
globals()['MDS'] = MDS
