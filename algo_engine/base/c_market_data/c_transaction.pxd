# cython: language_level=3
from libc.stdint cimport uint8_t, int8_t, uintptr_t

from .c_market_data cimport _MarketDataBuffer, _ID, _TransactionDataBuffer, _OrderDataBuffer


cdef class TransactionHelper:

    @staticmethod
    cdef void set_id(_ID* id_ptr, object id_value)

    @staticmethod
    cdef object get_id(_ID* id_ptr)

    @staticmethod
    cdef bint compare_id(const _ID* id1, const _ID* id2)

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


cdef class TransactionData:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef public uintptr_t _data_addr
    cdef _TransactionDataBuffer _data

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef TransactionData c_from_bytes(bytes data)


cdef class OrderData:
    cdef dict __dict__
    cdef _MarketDataBuffer* _data_ptr
    cdef public uintptr_t _data_addr
    cdef _OrderDataBuffer _data

    cdef bytes c_to_bytes(self)

    @staticmethod
    cdef OrderData c_from_bytes(bytes data)


cdef class TradeData(TransactionData):
    pass