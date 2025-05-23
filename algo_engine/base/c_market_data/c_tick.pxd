# cython: language_level=3
from libc.stdint cimport uint8_t, uintptr_t

from .c_market_data cimport _MarketDataBuffer, _OrderBookBuffer, _TickDataLiteBuffer, _TickDataBuffer


cdef class TickDataLite:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef public uintptr_t _data_addr
    cdef _TickDataLiteBuffer _data

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef TickDataLite c_from_bytes(bytes data)


cdef class OrderBook:
    cdef _OrderBookBuffer* _data
    cdef bint _owner
    cdef size_t _iter_index
    cdef public uint8_t side
    cdef public bint sorted

    cdef double c_loc_volume(self, double p0, double p1)

    @staticmethod
    cdef OrderBook c_from_bytes(bytes data, uint8_t side=*)

    cdef bytes c_to_bytes(self)

    cdef void c_sort(self)


cdef class TickData:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef public uintptr_t _data_addr
    cdef _TickDataBuffer _data
    cdef OrderBook _bid_book
    cdef OrderBook _ask_book

    cdef _init_order_book(self)

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef TickData c_from_bytes(bytes data)

    cpdef void parse(self, dict kwargs)

    cpdef TickDataLite lite(self)
