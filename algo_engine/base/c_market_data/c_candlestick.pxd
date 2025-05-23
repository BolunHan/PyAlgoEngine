# cython: language_level=3
from libc.stdint cimport uintptr_t

from .c_market_data cimport _MarketDataBuffer, _CandlestickBuffer


cdef class BarData:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef public uintptr_t _data_addr
    cdef _CandlestickBuffer _data

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef BarData c_from_bytes(bytes data)

