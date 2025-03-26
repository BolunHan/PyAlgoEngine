# cython: language_level=3

from .market_data cimport MarketData, _ID

cdef class TransactionData(MarketData):
    @staticmethod
    cdef void _set_id(_ID* id_ptr, object id_value)

    @staticmethod
    cdef object _get_id(_ID* id_ptr)

cdef class OrderData(MarketData):
    pass

cdef class TradeData(TransactionData):
    pass