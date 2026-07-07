from cpython.object cimport PyObject
from libcpp cimport bool as c_bool

from algo_engine.base.c_market_data.c_internal cimport InternalData
from algo_engine.base.c_market_data.c_market_data cimport MarketData, md_internal, md_variant
from algo_engine.exchange_profile.c_exchange_profile cimport ExchangeProfile


cdef extern from "Python.h":
    PyObject* PyDict_GetItemString(PyObject* py_dict, const char* key)
    int PyDict_SetItemString(PyObject* py_dict, const char* key, PyObject* val)
    PyObject* PyLong_FromSize_t(size_t v)
    size_t PyLong_AsSize_t(PyObject* pylong)
    PyObject* PyFloat_FromDouble(double v)


cdef class MonitorManager:
    cdef readonly dict monitor

    cpdef void feed_monitor(self, object monitor_id, MarketData market_data)

    cdef void c_on_market_data(self, const md_variant* market_data)

    cpdef void on_market_data(self, MarketData market_data)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, object monitor_id)

    cpdef void clear_monitors(self)

    cpdef void start(self)

    cpdef void stop(self)

    cpdef void clear(self)

    cpdef dict get_values(self)


cdef struct mds_subscription:
    const char* ticker
    double last_update
    double last_price
    size_t n_feeds
    c_bool is_subscribed_td
    c_bool is_subscribed_od
    c_bool is_subscribed_tk_lite
    c_bool is_subscribed_tk
    c_bool is_subscribed_cs


cdef class MarketDataService:
    cdef size_t subscription_capacity
    cdef dict subscription_mapping
    cdef mds_subscription* subscription_status

    cdef readonly ExchangeProfile profile
    cdef readonly double timestamp
    cdef readonly size_t n_subscribed
    cdef readonly dict monitor
    cdef readonly MonitorManager monitor_manager

    cdef void c_subscription_buffer_extend(self)

    cdef void c_update_subscription(self, const md_variant* md)

    cdef void c_on_internal_data(self, const md_internal* md)

    cdef void c_on_market_data(self, const md_variant* md)

    cpdef void on_internal_data(self, InternalData internal_data)

    cpdef void on_market_data(self, MarketData market_data)

    cpdef double get_market_price(self, str ticker)

    cpdef void set_manager(self, MonitorManager manager)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, object monitor=?, object monitor_id=?, str monitor_name=?)

    cpdef void clear(self)


cdef MarketDataService MDS
