from libc.stdint cimport uint32_t

from .c_market_data cimport market_data_t, MarketData


cdef class InternalData(MarketData):
    cdef uint32_t code


cdef inline object internal_from_header(market_data_t* market_data, bint owner):
    cdef InternalData instance = InternalData.__new__(InternalData)
    instance.header = market_data
    instance.owner = owner
    return instance
