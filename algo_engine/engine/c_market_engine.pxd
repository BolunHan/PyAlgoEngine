from ..base.c_market_data cimport _MarketDataBuffer
from ..profile cimport ProfileDispatcher


cdef MarketDataService C_MDS


cdef class MonitorManager:
    cdef public dict monitor

    cdef void c_on_market_data(self, _MarketDataBuffer* data_ptr)

    cpdef void add_monitor(self, object monitor)

    cpdef void pop_monitor(self, str monitor_id)


cdef class MarketDataService:
    cdef public ProfileDispatcher profile
    cdef public size_t max_subscription

    cdef dict mapping
    cdef dict _monitor
    cdef MonitorManager _monitor_manager

    cdef size_t _n
    cdef double _timestamp
    cdef double* _market_price
    cdef bint* _subscription_trade_data
    cdef bint* _subscription_order_data
    cdef bint* _subscription_tick_data_lite
    cdef bint* _subscription_tick_data
    cdef bint* _subscription_bar_data

    cdef void c_on_internal_data(self, _MarketDataBuffer* data_ptr)

    cdef void c_on_market_data(self, _MarketDataBuffer* data_ptr)

    cpdef void on_internal_data(self, object internal_data)

    cpdef void on_market_data(self, object market_data)

    cpdef double get_market_price(self, str ticker)
