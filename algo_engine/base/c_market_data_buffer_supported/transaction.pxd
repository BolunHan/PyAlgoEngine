# cython: language_level=3
from libc.stdint cimport uint8_t, int8_t

from .market_data cimport MarketData, _ID


# Declare TransactionHelper class
cdef class TransactionHelper:
    @staticmethod
    cdef uint8_t get_opposite(uint8_t side)

    @staticmethod
    cdef int8_t get_sign(uint8_t side)

    @staticmethod
    cdef uint8_t get_direction(uint8_t side)

    @staticmethod
    cdef uint8_t get_offset(uint8_t side)

    @staticmethod
    cdef const char* get_side_name(uint8_t side)

    @staticmethod
    cdef const char* get_order_type_name(uint8_t order_type)

    @staticmethod
    cdef const char* get_direction_name(uint8_t side)

    @staticmethod
    cdef const char* get_offset_name(uint8_t side)


cdef class TransactionData(MarketData):
    @staticmethod
    cdef void _set_id(_ID* id_ptr, object id_value)

    @staticmethod
    cdef object _get_id(_ID* id_ptr)

    @staticmethod
    cdef bint _id_equal(const _ID* id1, const _ID* id2)


cdef class OrderData(MarketData):
    pass

cdef class TradeData(TransactionData):
    pass