# cython: language_level=3
from .market_data cimport MarketData


cdef class BarData(MarketData):
    pass

