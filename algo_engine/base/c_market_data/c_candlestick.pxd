# cython: language_level=3
from .c_market_data cimport MarketData, _CandlestickBuffer


cdef class BarData(MarketData):
    cdef _CandlestickBuffer _data

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef BarData c_from_bytes(bytes data)

