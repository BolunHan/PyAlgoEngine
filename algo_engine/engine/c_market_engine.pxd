from libcpp cimport bool as c_bool
from cpython.object cimport PyObject

from ..base.c_market_data.c_market_data cimport MarketData, md_variant, md_internal
from ..base.c_market_data.c_internal cimport InternalData
from ..exchange_profile.c_exchange_profile cimport ExchangeProfile


cdef extern from "Python.h":
    PyObject* PyDict_GetItemString(PyObject* py_dict, const char* key)
    int PyDict_SetItemString(PyObject* py_dict, const char* key, PyObject* val)
    PyObject* PyLong_FromSize_t(size_t v)
    size_t PyLong_AsSize_t(PyObject* pylong)
    PyObject* PyFloat_FromDouble(double v)


cdef class MonitorManager:
    cdef readonly dict monitor

    cdef void c_on_market_data(self, const md_variant* md)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, str monitor_id)

    cpdef void clear_monitors(self)


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
    cdef MonitorManager monitor_manager

    cdef void c_subscription_buffer_extend(self)

    cdef void c_update_subscription(self, const md_variant* md)

    cdef void c_on_internal_data(self, const md_internal* md)

    cdef void c_on_market_data(self, const md_variant* md)

    cpdef void on_internal_data(self, InternalData internal_data)

    cpdef void on_market_data(self, MarketData market_data)

    cpdef double get_market_price(self, str ticker)


cdef MarketDataService MDS
