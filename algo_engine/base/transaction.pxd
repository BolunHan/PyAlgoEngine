# cython: language_level=3

from .market_data cimport MarketData, _ID


# Declare TransactionHelper class
cdef class TransactionHelper:
    @staticmethod
    cdef int get_opposite(int side)

    @staticmethod
    cdef int get_sign(int side)

    @staticmethod
    cdef int get_direction(int side)

    @staticmethod
    cdef int get_offset(int side)

    @staticmethod
    cdef const char* get_side_name(int side)

    @staticmethod
    cdef const char* get_order_type_name(int order_type)

    @staticmethod
    cdef const char* get_direction_name(int side)

    @staticmethod
    cdef const char* get_offset_name(int side)


cdef class TransactionData(MarketData):
    @staticmethod
    cdef void _set_id(_ID* id_ptr, object id_value)

    @staticmethod
    cdef object _get_id(_ID* id_ptr)

cdef class OrderData(MarketData):
    pass

cdef class TradeData(TransactionData):
    pass