# cython: language_level=3
from .market_data cimport MarketData, _OrderBookBuffer


cdef class TickDataLite(MarketData):
    pass


cdef class OrderBook:
    cdef _OrderBookBuffer* _data
    cdef public int side
    cdef public bint sorted
    cdef int _iter_index
    cdef bint _owner

    cpdef double loc_volume(self, double p0, double p1)

    cpdef bytes to_bytes(self)

    cpdef void sort(self)

cdef class TickData(TickDataLite):
    cdef OrderBook _bid_book
    cdef OrderBook _ask_book

    cpdef void parse(self, dict kwargs)

    cpdef TickDataLite lite(self)
